from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from traffic_routing import Demand, ExperimentConfig, Network, Planner, Policy, Road, run_experiment
from traffic_routing.experiment import demo_scenario
from traffic_routing.report import write_report
from traffic_routing.simulation import World


class NetworkTests(unittest.TestCase):
    def test_path_order_and_parallel_roads(self):
        network = Network([Road("slow", "A", "B", 3, 1), Road("fast", "A", "B", 1, 1),
                           Road("bc", "B", "C", 1, 1), Road("ac", "A", "C", 4, 1),
                           Road("ba", "B", "A", 1, 1)])
        self.assertEqual(network.paths("A", "C"), (("fast", "bc"), ("ac",), ("slow", "bc")))
        self.assertEqual(network.paths("A", "A"), ((),))
        with self.assertRaises(ValueError):
            network.paths("C", "A")
        with self.assertRaises(ValueError):
            network.paths("missing", "A")

    def test_bad_road_parameters(self):
        for free_flow, capacity in ((0, 1), (1, 0), (-1, 1), (1.5, 1), (True, 1)):
            with self.subTest(free_flow=free_flow, capacity=capacity), self.assertRaises(ValueError):
                Road("a", "A", "B", free_flow, capacity)
        with self.assertRaises(ValueError):
            Network([Road("a", "A", "B", 1, 1), Road("a", "B", "C", 1, 1)])

    def test_route_validation(self):
        net = Network([Road("ab", "A", "B", 1, 1), Road("bc", "B", "C", 1, 1)])
        demand = Demand("x", "A", "C", 0)
        for path in ((), ("bc",), ("ab",), ("bad",)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                World(net).add(demand, path, 0)


class SimulatorTests(unittest.TestCase):
    def test_fifo_capacity_and_exact_delay(self):
        net = Network([Road("ab", "A", "B", 3, 2)])
        demand = [Demand(str(i), "A", "B", 0) for i in range(5)]
        result = run_experiment(net, demand, "static", ExperimentConfig(measurement_window=10))
        self.assertEqual([trip["arrival"] for trip in result["trips"]], [3, 3, 4, 4, 5])
        self.assertEqual(result["metrics"]["total_delay"], 4)
        self.assertEqual(result["metrics"]["window_throughput"], .5)

    def test_no_extra_junction_tick(self):
        net = Network([Road("ab", "A", "B", 2, 1), Road("bc", "B", "C", 3, 1)])
        result = run_experiment(net, [Demand("x", "A", "C", 0)], "static")
        self.assertEqual(result["trips"][0]["arrival"], 5)
        self.assertEqual(result["trips"][0]["delay"], 0)

    def test_queue_admission_matches_independent_single_edge_oracle(self):
        rng = random.Random(45)
        for capacity in (1, 2, 5):
            demands = [Demand(f"v{i:03}", "A", "B", rng.randrange(10)) for i in range(50)]
            net = Network([Road("ab", "A", "B", 3, capacity)])
            slots = {}
            expected = {}
            for d in sorted(demands, key=lambda v: (v.departure, v.id)):
                entry = d.departure
                while slots.get(entry, 0) == capacity:
                    entry += 1
                slots[entry] = slots.get(entry, 0) + 1
                expected[d.id] = entry + 3
            result = run_experiment(net, demands, "static")
            self.assertEqual({trip["id"]: trip["arrival"] for trip in result["trips"]}, expected)

    def test_zero_length_trip_and_empty_demand(self):
        net = Network([Road("ab", "A", "B", 1, 1)])
        for policy in (p for p in Policy if p != Policy.RL_SWARM):
            result = run_experiment(net, [Demand("x", "A", "A", 2)], policy)
            self.assertEqual(result["metrics"]["average_travel_time"], 0)
            self.assertEqual(result["metrics"]["completed"], 1)
            empty = run_experiment(net, [], policy)
            self.assertEqual(empty["metrics"]["completed"], 0)
            self.assertEqual(empty["metrics"]["peak_utilization"], 0)

    def test_fixed_window_does_not_hide_unfinished_trips(self):
        net = Network([Road("ab", "A", "B", 5, 1)])
        result = run_experiment(net, [Demand("x", "A", "B", 0)], "static",
                                ExperimentConfig(measurement_window=2))
        self.assertEqual(result["metrics"]["window_completed"], 0)
        self.assertEqual(result["metrics"]["completed"], 1)
        self.assertEqual(result["metrics"]["last_arrival"], 5)
        with self.assertRaises(RuntimeError):
            run_experiment(net, [Demand("x", "A", "B", 0)], "static",
                           ExperimentConfig(measurement_window=2, max_ticks=2))

    def test_clone_is_independent(self):
        net = Network([Road("ab", "A", "B", 1, 1)])
        world = World(net)
        world.add(Demand("x", "A", "B", 0), ("ab",), 0)
        clone = world.clone()
        clone.serve(0)
        clone.release(1)
        self.assertEqual(list(world.queues["ab"]), ["x"])
        self.assertIsNone(world.active["x"].exit_at)
        self.assertNotIn("x", world.finished)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.net = Network([Road("oa", "O", "A", 2, 10), Road("ad", "A", "D", 1, 1),
                            Road("od", "O", "D", 6, 2)])

    def test_future_commitments_change_route_before_congestion_is_observed(self):
        world = World(self.net)
        for i in range(6):
            world.add(Demand(f"old{i}", "O", "D", 0), ("oa", "ad"), 0)
        world.serve(0)
        world.release(1)
        self.assertEqual(len(world.queues["ad"]), 0)
        planner = Planner(self.net)
        new = Demand("new", "O", "D", 1)
        for policy in (Policy.STATIC, Policy.REACTIVE, Policy.PREDICTIVE):
            self.assertEqual(planner.choose(new, policy, world, world, 1, []).path, ("oa", "ad"))
        decision = planner.choose(new, Policy.ANTICIPATORY, world, world, 1, [])
        self.assertEqual(decision.path, ("od",))
        self.assertEqual(decision.predicted_travel_time, 6)
        self.assertEqual(len(world.active), 6, "Planning must not mutate the live simulation")

    def test_exogenous_forecast_changes_predictive_route(self):
        world = World(self.net)
        forecast = [Demand(f"bg{i}", "A", "D", 2, False) for i in range(6)]
        new = Demand("new", "O", "D", 1)
        planner = Planner(self.net)
        self.assertEqual(planner.choose(new, "reactive", world, world, 1, forecast).path, ("oa", "ad"))
        self.assertEqual(planner.choose(new, "predictive", world, world, 1, forecast).path, ("od",))
        no_horizon = Planner(self.net, forecast_horizon=0)
        self.assertEqual(no_horizon.choose(new, "predictive", world, world, 1, forecast).path, ("oa", "ad"))

    def test_cooperative_accounts_for_cost_imposed_on_other_vehicles(self):
        # The new vehicle can save one tick by reaching AD ahead of three
        # committed vehicles, but doing so adds one tick to EACH of their trips.
        net = Network([Road("oa", "O", "A", 1, 10), Road("ad", "A", "D", 1, 1),
                       Road("od", "O", "D", 3, 1), Road("xa", "X", "A", 2, 10)])
        world = World(net)
        for i in range(3):
            world.add(Demand(f"z-old{i}", "X", "D", 0), ("xa", "ad"), 0)
        world.serve(0)
        world.release(1)
        planner = Planner(net)
        demand = Demand("a-new", "O", "D", 1)
        selfish = planner.choose(demand, "anticipatory", world, world, 1, [])
        cooperative = planner.choose(demand, "cooperative", world, world, 1, [])
        self.assertEqual(selfish.path, ("oa", "ad"))
        self.assertEqual(selfish.predicted_travel_time, 2)
        self.assertEqual(cooperative.path, ("od",))
        self.assertEqual(cooperative.predicted_travel_time, 3)

    def test_all_policies_match_in_uncongested_network(self):
        demand = [Demand("x", "O", "D", 0)]
        for policy in (p for p in Policy if p != Policy.RL_SWARM):
            result = run_experiment(self.net, demand, policy)
            self.assertEqual(result["metrics"]["average_travel_time"], 3)

    def test_assignment_order_is_deterministic_and_input_order_independent(self):
        network, demand = demo_scenario(7)
        demand = demand[:30]
        for policy in (p for p in Policy if p != Policy.RL_SWARM):
            first = run_experiment(network, demand, policy)
            second = run_experiment(network, list(reversed(demand)), policy)
            self.assertEqual(first, second)

    def test_demo_conserves_vehicles_and_commitments_help(self):
        network, demand = demo_scenario(7)
        results = {p: run_experiment(network, demand, p) for p in Policy if p != Policy.RL_SWARM}
        for result in results.values():
            self.assertEqual(result["metrics"]["completed"], len(demand))
            self.assertTrue(all(t["delay"] >= 0 for t in result["trips"]))
        self.assertLess(results[Policy.ANTICIPATORY]["metrics"]["total_travel_time"],
                        results[Policy.REACTIVE]["metrics"]["total_travel_time"])

    def test_forecast_can_be_disabled_or_include_false_positives(self):
        demand = [Demand("x", "O", "D", 0)]
        for forecast in ([], [Demand("hypothetical", "A", "D", 2, False)]):
            result = run_experiment(self.net, demand, "anticipatory", forecast=forecast)
            self.assertEqual(result["metrics"]["completed"], 1)

    def test_rejects_bad_inputs_and_duplicate_forecasts(self):
        d = Demand("x", "O", "D", 0)
        with self.assertRaises(ValueError):
            run_experiment(self.net, [d, d], "static")
        with self.assertRaises(ValueError):
            run_experiment(self.net, [d], "static", forecast=[d])
        bg = replace(d, controlled=False)
        with self.assertRaises(ValueError):
            run_experiment(self.net, [bg], "predictive", forecast=[replace(bg, departure=1)])
        with self.assertRaises(ValueError):
            run_experiment(self.net, [bg], "predictive", forecast=[bg, bg])
        with self.assertRaises(ValueError):
            run_experiment(self.net, [replace(d, departure=120)], "static")
        for kwargs in ({"alpha": -1}, {"gamma": float("nan")}, {"alpha": 0}, {"k": 0}):
            with self.assertRaises(ValueError):
                Planner(self.net, **kwargs)

    def test_rollout_timeout_is_explicit(self):
        world = World(self.net)
        with self.assertRaises(RuntimeError):
            Planner(self.net, rollout_limit=1).choose(Demand("x", "O", "D", 0),
                                                      "anticipatory", world, world, 0, [])


class OutputTests(unittest.TestCase):
    def test_report_roundtrip_and_html_escaping(self):
        net = Network([Road("<script>", "A", "B", 1, 1)])
        result = run_experiment(net, [Demand("x", "A", "B", 0)], "static")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            write_report([result], path, {"label": "<script>"})
            document = (path / "report.html").read_text()
            self.assertNotIn("<script>", document)
            self.assertIn("&lt;script&gt;", document)
            self.assertEqual(json.loads((path / "results.json").read_text())["results"][0]["metrics"], result["metrics"])
            self.assertTrue((path / "metrics.csv").is_file())

    def test_cli_accepts_custom_scenario_and_rejects_invalid_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            scenario = path / "scenario.json"
            scenario.write_text(json.dumps({"roads": [{"id": "ab", "source": "A", "target": "B", "free_flow": 1, "capacity": 1}],
                                             "demands": [{"id": "x", "origin": "A", "destination": "B", "departure": 0}]}))
            command = [sys.executable, "-m", "traffic_routing", "--scenario", str(scenario), "--output", str(path / "report")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((path / "report" / "report.html").is_file())
            result = subprocess.run(command + ["--candidates", "0"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
