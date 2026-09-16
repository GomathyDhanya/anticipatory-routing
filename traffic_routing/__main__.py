from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from .experiment import ExperimentConfig, demo_scenario, run_experiment
from .model import Demand, Network, Road
from .report import write_report
from .routing import Policy
from .swarm import SwarmPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare commitment-aware traffic routing policies.")
    parser.add_argument("--scenario", type=Path, help="JSON containing roads, demands, and optional forecast")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--forecast-horizon", type=int, default=30)
    parser.add_argument("--measurement-window", type=int, default=120)
    parser.add_argument("--max-ticks", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=1.0, help="Cooperative travel-time weight")
    parser.add_argument("--beta", type=float, default=0.0, help="Cooperative delay weight")
    parser.add_argument("--gamma", type=float, default=0.0, help="Cooperative overload weight")
    parser.add_argument("--policies", nargs="+", choices=[p.value for p in Policy])
    parser.add_argument("--rl-model", type=Path, help="Trained JSON checkpoint for rl_swarm")
    parser.add_argument("--replan", action="store_true", help="Reconsider remaining routes at junctions")
    parser.add_argument("--compare-planning", action="store_true", help="Report both departure and junction planning")
    args = parser.parse_args()
    try:
        forecast = None
        rl_agent = SwarmPolicy.load(args.rl_model) if args.rl_model else None
        if args.scenario:
            data = json.loads(args.scenario.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or set(data) - {"roads", "demands", "forecast"}:
                raise ValueError("Scenario must be an object with roads, demands, and optional forecast")
            network = Network([Road(**row) for row in data["roads"]])
            demand = [Demand(**row) for row in data["demands"]]
            if "forecast" in data:
                forecast = [Demand(**row) for row in data["forecast"]]
        else:
            network, demand = demo_scenario(args.seed)
        config = ExperimentConfig(seed=args.seed, candidates=args.candidates,
                                  forecast_horizon=args.forecast_horizon,
                                  measurement_window=args.measurement_window,
                                  max_ticks=args.max_ticks, alpha=args.alpha,
                                  beta=args.beta, gamma=args.gamma, replan=args.replan)
        scenario = {"roads": [asdict(r) for r in network.roads.values()],
                    "demands": [asdict(d) for d in demand]}
        if forecast is not None:
            scenario["forecast"] = [asdict(d) for d in forecast]
        results = []
        print(f"{'Policy':<15} {'Mean trip':>10} {'Total delay':>12} {'P95':>8}", flush=True)
        policies = args.policies or [p for p in Policy if rl_agent is not None or p != Policy.RL_SWARM]
        modes = (False, True) if args.compare_planning else (args.replan,)
        for policy, replan in ((p, mode) for mode in modes for p in dict.fromkeys(policies)):
            result = run_experiment(network, demand, policy, replace(config, replan=replan), forecast, rl_agent=rl_agent)
            results.append(result)
            m = result["metrics"]
            print(f"{policy:<15} {m['average_travel_time']:>10.2f} {m['total_delay']:>12} {m['p95_travel_time']:>8}", flush=True)
        write_report(results, args.output, scenario)
        print(f"\nReport: {(args.output / 'report.html').resolve()}")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
