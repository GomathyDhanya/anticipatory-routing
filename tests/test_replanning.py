import unittest

from traffic_routing.experiment import ExperimentConfig, run_experiment
from traffic_routing.model import Demand, Network, Road
from traffic_routing.routing import Policy
from traffic_routing.swarm import SwarmPolicy


class ReplanningTests(unittest.TestCase):
    def setUp(self):
        self.network = Network([
            Road("oa", "O", "A", 3, 10), Road("ad", "A", "D", 1, 1),
            Road("ab", "A", "B", 1, 10), Road("bd", "B", "D", 2, 10),
            Road("ao", "A", "O", 1, 10),
        ])
        self.demands = [Demand("car", "O", "D", 0)] + [
            Demand(f"bg-{i:02}", "A", "D", 2, False) for i in range(12)]

    def run_policy(self, policy, replan):
        return run_experiment(self.network, self.demands, policy,
                              ExperimentConfig(replan=replan, measurement_window=30),
                              forecast=[], capture_trace=True,
                              rl_agent=SwarmPolicy([0, -1, 0, 0, 0]))

    def test_reactive_avoids_queue_that_formed_after_departure(self):
        fixed = self.run_policy(Policy.REACTIVE, False)
        dynamic = self.run_policy(Policy.REACTIVE, True)
        fixed_trip = next(t for t in fixed["trips"] if t["id"] == "car")
        dynamic_trip = next(t for t in dynamic["trips"] if t["id"] == "car")
        self.assertEqual(fixed_trip["path"], ("oa", "ad"))
        self.assertEqual(dynamic_trip["path"], ("oa", "ab", "bd"))
        self.assertLess(dynamic_trip["arrival"], fixed_trip["arrival"])
        self.assertEqual(dynamic["metrics"]["route_changes"], 1)
        self.assertEqual([v[0] for v in dynamic["trajectories"]["car"]], list(dynamic_trip["path"]))

    def test_all_policies_conserve_vehicles_and_record_legal_complete_traces(self):
        for policy in Policy:
            with self.subTest(policy=policy):
                result = self.run_policy(policy, True)
                self.assertEqual(result["metrics"]["completed"], len(self.demands))
                by_id = {d.id: d for d in self.demands}
                self.assertGreater(result["metrics"]["replanning_decisions"], 0)
                for trip in result["trips"]:
                    self.network.validate_path(by_id[trip["id"]], trip["path"])
                    visits = result["trajectories"][trip["id"]]
                    self.assertEqual([v[0] for v in visits], list(trip["path"]))
                    self.assertTrue(all(q <= enter < leave for _, q, enter, leave in visits))
                    self.assertEqual(visits[-1][3], trip["arrival"])

    def test_static_reconsiders_but_keeps_free_flow_route(self):
        result = self.run_policy(Policy.STATIC, True)
        self.assertEqual(result["metrics"]["route_changes"], 0)
        self.assertEqual(next(t for t in result["trips"] if t["id"] == "car")["path"], ("oa", "ad"))

    def test_replanning_is_deterministic(self):
        self.assertEqual(self.run_policy(Policy.COOPERATIVE, True),
                         self.run_policy(Policy.COOPERATIVE, True))

    def test_rl_collects_learning_gradients_for_departure_and_junction(self):
        agent = SwarmPolicy(stochastic=True, seed=42)
        result = run_experiment(self.network, self.demands, Policy.RL_SWARM,
                                ExperimentConfig(replan=True, measurement_window=30),
                                rl_agent=agent)
        self.assertEqual(len(agent.episode_gradients), 1 + result["metrics"]["replanning_decisions"])
        self.assertGreater(len(agent.episode_gradients), 1)
