from dataclasses import replace
import json
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

from traffic_routing import Demand, ExperimentConfig, Network, Road, run_experiment
from traffic_routing.simulation import World
from traffic_routing.swarm import AdamAscent, FEATURES, SwarmPolicy, route_features, softmax


class SwarmTests(unittest.TestCase):
    def setUp(self):
        self.net = Network([Road("a", "O", "A", 2, 10), Road("b", "A", "D", 1, 1), Road("c", "O", "D", 5, 2)])
        self.demand = Demand("new", "O", "D", 0)

    def test_softmax_stability_and_validation(self):
        self.assertEqual(softmax([10000, 10000]), [.5, .5])
        self.assertAlmostEqual(sum(softmax([-1000, 0, 1000])), 1)
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                softmax(values)
        for weights in ([0], [0, 0, 0, 0, float("nan")]):
            with self.assertRaises(ValueError):
                SwarmPolicy(weights)

    def test_commitment_board_sees_future_vehicles_and_ablation_hides_them(self):
        world = World(self.net)
        for i in range(6):
            world.add(Demand(f"old{i}", "O", "D", 0), ("a", "b"), 0)
        world.serve(0)
        paths = self.net.paths("O", "D")
        enabled = route_features(self.net, world, paths)
        hidden = route_features(self.net, world, paths, coordination=False)
        self.assertGreater(enabled[0][2], 0)
        self.assertGreater(enabled[0][2], hidden[0][2])
        self.assertAlmostEqual(hidden[0][2], 6/10/3)
        self.assertEqual(enabled[0][1], hidden[0][1])

    def test_policy_gradient_matches_finite_difference(self):
        world = World(self.net)
        weights = [-.2, .1, -.3, .4, -.5]
        agent = SwarmPolicy(weights, stochastic=True, seed=12)
        decision = agent.choose(self.demand, world, 8)
        paths = self.net.paths("O", "D", 8)
        features = route_features(self.net, world, paths)
        action = paths.index(decision.path)
        analytic = agent.log_policy_gradient()
        for j in range(len(weights)):
            plus, minus = list(weights), list(weights)
            plus[j] += 1e-5
            minus[j] -= 1e-5
            def logp(w):
                return math.log(softmax([sum(a*b for a, b in zip(w, row)) for row in features])[action])
            numeric = (logp(plus)-logp(minus))/2e-5
            self.assertAlmostEqual(analytic[j], numeric, places=6)

    def test_actual_rewards_teach_policy_to_prefer_shorter_route(self):
        weights = [0.0]*len(FEATURES)
        optimizer = AdamAscent(len(weights), .1)
        for episode in range(32):
            agent = SwarmPolicy(weights, stochastic=True, seed=episode)
            result = run_experiment(self.net, [self.demand], "rl_swarm",
                                    ExperimentConfig(measurement_window=6), rl_agent=agent)
            # Action-independent baseline = 3 ticks, reward = -trip time.
            advantage = 3-result["metrics"]["average_travel_time"]
            optimizer.update(weights, [advantage*g for g in agent.log_policy_gradient()])
        learned = SwarmPolicy(weights)
        learned.choose(self.demand, World(self.net), 8)
        self.assertGreater(learned.last_probabilities[0], .7)
        self.assertLess(weights[0], 0)

    def test_model_roundtrip_and_invalid_schema(self):
        model = SwarmPolicy([-1, -2, -3, -4, -5], metadata={"trained": True})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.json"
            model.save(path)
            loaded = SwarmPolicy.load(path)
            self.assertEqual(loaded.weights, model.weights)
            self.assertEqual(loaded.metadata, model.metadata)
            self.assertFalse(loaded.stochastic)
            payload = json.loads(path.read_text());payload["features"][0] = "wrong"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                SwarmPolicy.load(path)

    def test_frozen_inference_is_reproducible_and_does_not_update_weights(self):
        model = SwarmPolicy([-1, -1, -1, -1, -1])
        original = list(model.weights)
        demand = [replace(self.demand, id=f"v{i}") for i in range(12)]
        first = run_experiment(self.net, demand, "rl_swarm", rl_agent=model, capture_trace=True)
        second = run_experiment(self.net, list(reversed(demand)), "rl_swarm", rl_agent=model, capture_trace=True)
        self.assertEqual(first, second)
        self.assertEqual(model.weights, original)
        self.assertEqual(model.episode_gradients, [])
        self.assertEqual(first["metrics"]["completed"], len(demand))
        with self.assertRaises(ValueError):
            run_experiment(self.net, demand, "rl_swarm")

    def test_single_action_and_zero_length_routes_have_zero_gradient(self):
        agent = SwarmPolicy(stochastic=True)
        decision = agent.choose(Demand("x", "O", "O", 0), World(self.net), 8)
        self.assertEqual(decision.path, ())
        self.assertEqual(agent.log_policy_gradient(), [0]*len(FEATURES))

    def test_bundled_checkpoint_selection_and_held_out_seed_separation(self):
        folder = Path(__file__).resolve().parent.parent / "map_demo/rl"
        model = SwarmPolicy.load(folder / "swarm-model.json")
        meta = model.metadata
        training = set(range(meta["training_seeds"][0], meta["training_seeds"][1]+1))
        validation, test = set(meta["validation_seeds"]), set(meta["test_seeds"])
        self.assertFalse(training & validation or training & test or validation & test)
        self.assertNotIn(meta["demo_seed"], training | validation | test)
        history = json.loads((folder / "training-history.json").read_text())
        validated = [row for row in history if "validation_mean_trip_ticks" in row]
        best = min(validated, key=lambda row: row["validation_mean_trip_ticks"])
        self.assertEqual(model.weights, best["weights"])
        self.assertEqual(meta["selected_episode"], best["episodes"])
        self.assertNotEqual(model.weights, meta["initial_weights"])
        report = json.loads((folder / "evaluation.json").read_text())
        digest = hashlib.sha256((folder / "swarm-model.json").read_bytes()).hexdigest()
        self.assertEqual(report["checkpoint_sha256"], digest)
        self.assertEqual({row["seed"] for row in report["episodes"]}, test)


if __name__ == "__main__":
    unittest.main()
