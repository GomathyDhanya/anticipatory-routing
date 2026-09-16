"""Build full comparison reports from the cached real-road replay, without rerunning it."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from .experiment import ExperimentConfig
from .report import write_report


def build(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent.parent / "map_demo"
    payload = json.loads((root / "data/demo.json").read_text())
    for scene in payload["scenes"]:
        results = []
        for mode, key in ((False, "policies"), (True, "replanning")):
            for policy, run in scene[key].items():
                config = ExperimentConfig(seed=19, candidates=6, forecast_horizon=30,
                                          measurement_window=300, max_ticks=2000, replan=mode)
                routes = Counter(" → ".join(v["initialPath"]) for v in run["vehicles"] if v["controlled"])
                results.append({"policy": policy, "config": asdict(config), "metrics": run["metrics"],
                                "route_counts": dict(routes), "replans": run["replans"],
                                "vehicles": run["vehicles"],
                                "series": [{"tick": tick, "completed": row[0], "queued": row[1], "active": row[2]}
                                           for tick, row in enumerate(run["series"])]})
        scenario = {"label": "Milpitas → Sunnyvale · " + scene["label"],
                    "units": f"1 tick = {payload['tickSeconds']} seconds; traffic is simulated",
                    "roads": payload["roads"], "capacities": scene["capacities"],
                    "bottleneck": scene["bottleneck"], "sources": payload["sources"],
                    "rl": payload["rl"]}
        write_report(results, root / "reports" / scene["id"], scenario)


if __name__ == "__main__":
    build()
