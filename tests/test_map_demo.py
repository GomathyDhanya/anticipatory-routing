import collections
import json
from pathlib import Path
import random
import unittest

from traffic_routing import Demand, Network, Road, run_experiment
from traffic_routing.map_demo import load_corridors


class MapDemoTests(unittest.TestCase):
    def test_yen_paths_match_exhaustive_search_on_small_graphs(self):
        rng = random.Random(14)
        for trial in range(20):
            roads = [Road(f"r{a}{b}", str(a), str(b), rng.randrange(1, 5), 1)
                     for a in range(6) for b in range(6) if a != b and rng.random() < .38]
            if not roads:
                continue
            net = Network(roads)
            if "0" not in net.nodes or "5" not in net.nodes:
                continue
            exhaustive = []
            def visit(node, path, visited):
                if node == "5":
                    exhaustive.append(path)
                    return
                for road in net.outgoing[node]:
                    if road.target not in visited:
                        visit(road.target, path + (road.id,), visited | {road.target})
            visit("0", (), {"0"})
            if exhaustive:
                expected = sorted(exhaustive, key=lambda p: (net.travel_time(p), p))[:8]
                self.assertEqual(list(net.paths("0", "5", 8)), expected, f"trial {trial}")
            else:
                with self.assertRaises(ValueError):
                    net.paths("0", "5", 8)

    def test_optional_trace_does_not_change_results(self):
        net = Network([Road("a", "A", "B", 2, 1), Road("b", "B", "C", 3, 1)])
        demand = [Demand(f"v{i}", "A", "C", 0) for i in range(4)]
        traced = run_experiment(net, demand, "anticipatory", capture_trace=True)
        history = traced.pop("trajectories")
        self.assertEqual(traced, run_experiment(net, demand, "anticipatory"))
        self.assertEqual(history["v0"], [["a", 0, 0, 2], ["b", 2, 2, 5]])
        self.assertEqual(history["v3"], [["a", 0, 3, 5], ["b", 5, 5, 8]])

    def test_sourced_corridors_are_connected_and_share_geometry(self):
        net, features, origin, destination, coords, source_routes = load_corridors()
        self.assertEqual(len(source_routes), 3)
        paths = net.paths(origin, destination, 6)
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            net.validate_path(Demand("x", origin, destination, 0), path)
        directed_segments = []
        for road in net.roads.values():
            geometry = features[road.id]["geometry"]
            self.assertEqual(geometry[0], coords[road.source])
            self.assertEqual(geometry[-1], coords[road.target])
            directed_segments.extend((tuple(a), tuple(b)) for a, b in zip(geometry, geometry[1:]))
        self.assertEqual(len(directed_segments), len(set(directed_segments)), "Shared roads must not create duplicate capacity")

    def test_cached_replays_match_capacity_timing_metrics_and_conservation(self):
        payload = json.loads((Path(__file__).resolve().parent.parent / "map_demo/data/demo.json").read_text())
        for scene in payload["scenes"]:
            self.assertIn("rl_swarm", scene["policies"])
            self.assertEqual(set(scene["replanning"]), set(scene["policies"]))
            departures = None
            runs = list(scene["policies"].items()) + [(name + " replan", run) for name, run in scene["replanning"].items()]
            for name, run in runs:
                with self.subTest(scene=scene["id"], policy=name):
                    signature = [(v["id"], v["departure"], v["controlled"]) for v in run["vehicles"]]
                    if departures is None:
                        departures = signature
                    self.assertEqual(signature, departures)
                    self.assertEqual(len(signature), scene["vehicles"])
                    admissions = collections.Counter()
                    total_time = total_delay = 0
                    for vehicle in run["vehicles"]:
                        previous = vehicle["departure"]
                        for edge, queued, entered, exited in vehicle["visits"]:
                            self.assertEqual(previous, queued)
                            self.assertLessEqual(queued, entered)
                            self.assertEqual(exited-entered, payload["roads"][edge]["free_flow"])
                            admissions[edge, entered] += 1
                            total_delay += entered-queued
                            previous = exited
                        self.assertEqual(previous, vehicle["arrival"])
                        total_time += vehicle["arrival"]-vehicle["departure"]
                    self.assertEqual(total_delay, run["metrics"]["total_delay"])
                    self.assertEqual(total_time, run["metrics"]["total_travel_time"])
                    self.assertTrue(all(count <= scene["capacities"][edge] for (edge, tick), count in admissions.items()))
                    for tick, (completed, queued, active) in enumerate(run["series"]):
                        self.assertEqual(completed, sum(v["arrival"] <= tick for v in run["vehicles"]))
                        self.assertEqual(active, sum(v["departure"] <= tick < v["arrival"] for v in run["vehicles"]))
                        count = sum(q <= tick < enter for v in run["vehicles"] for edge, q, enter, exit_at in v["visits"])
                        self.assertEqual(queued, count)


if __name__ == "__main__":
    unittest.main()
