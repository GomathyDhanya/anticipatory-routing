import csv
import json
from pathlib import Path
import tempfile
import unittest

from traffic_routing.experiment import ExperimentConfig, run_experiment
from traffic_routing.model import Demand, Network, Road
from traffic_routing.report import write_report
from traffic_routing.routing import Policy
from traffic_routing.swarm import SwarmPolicy


class ReportModesTests(unittest.TestCase):
    def test_all_policy_mode_pairs_survive_report_and_csv(self):
        network = Network([Road("od", "O", "D", 1, 1)])
        results = [run_experiment(network, [Demand("car", "O", "D", 0)], policy,
                                  ExperimentConfig(replan=mode), rl_agent=SwarmPolicy())
                   for mode in (False, True) for policy in Policy]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(results, root, {})
            document = (root / "report.html").read_text()
            self.assertIn("RL swarm · Junction planning", document)
            self.assertIn("RL swarm · Departure planning", document)
            self.assertEqual(document.count('<polyline '), 24)
            self.assertEqual(document.count('stroke-dasharray="7 4"'), 12)
            with (root / "metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({(r["policy"], r["planning_mode"]) for r in rows},
                             {(p.value, m) for p in Policy for m in ("departure", "junction")})
            self.assertEqual(len(json.loads((root / "results.json").read_text())["results"]), 12)
