"""Reproducible policy-gradient training and held-out corridor evaluation.

python3 -m traffic_routing.train_swarm --episodes 480
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import random
import statistics

from .experiment import ExperimentConfig, run_experiment
from .map_demo import ROOT, TICK_SECONDS, load_corridors
from .model import Demand, Network, Road, positive_integer
from .routing import Policy
from .swarm import AdamAscent, FEATURES, SwarmPolicy

CONFIG = ExperimentConfig(candidates=6, forecast_horizon=30, measurement_window=300, max_ticks=2000)


def scenario(base, features, origin, destination, seed):
    rng = random.Random(seed)
    kind = seed % 3
    count = rng.randrange(48, 97) if kind == 0 else rng.randrange(174, 247)
    spread = rng.randrange(36, 61) if kind == 0 else rng.randrange(12, 25)
    bottleneck = max((e for e, f in features.items() if f["is_highway"] and f["capacity"] < 24),
                     key=lambda e: features[e]["meters"])
    net = Network([Road(r.id, r.source, r.target, r.free_flow,
                        1 if kind == 2 and r.id == bottleneck else r.capacity) for r in base.roads.values()])
    demand = [Demand(f"car-{i:04}", origin, destination, rng.randrange(spread), i % 6 != 0) for i in range(count)]
    return net, demand, ("light", "surge", "restriction")[kind]


def validation_cost(weights, scenarios):
    costs = []
    for network, demand, kind in scenarios:
        result = run_experiment(network, demand, Policy.RL_SWARM, CONFIG, rl_agent=SwarmPolicy(weights))
        costs.append(result["metrics"]["average_travel_time"])
    return statistics.mean(costs)


def train(episodes=480, batch_size=8, output=ROOT / "rl"):
    positive_integer(episodes, "episodes")
    positive_integer(batch_size, "batch_size")
    if episodes > 9000:
        raise ValueError("Use at most 9000 episodes to preserve seed-set separation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base, features, origin, destination, _, _ = load_corridors()
    validation_seeds = list(range(20_000, 20_006))
    test_seeds = list(range(30_000, 30_012))
    validation = [scenario(base, features, origin, destination, seed) for seed in validation_seeds]
    weights = [0.0] * len(FEATURES)
    optimizer = AdamAscent(len(weights), learning_rate=.18)
    history = []
    # Keep the best learned checkpoint on validation only. Test seeds and the
    # seed-19 visual demos are never used to update or select weights.
    best_weights, best_cost, best_episode = None, float("inf"), 0
    for first in range(0, episodes, batch_size):
        gradients, returns = [], []
        for episode in range(first, min(first+batch_size, episodes)):
            seed = 10_000 + episode
            net, demand, kind = scenario(base, features, origin, destination, seed)
            shortest = net.travel_time(net.paths(origin, destination, 1)[0])
            # Self-critical baseline: deterministic rollout under current frozen
            # parameters, independent of this episode's sampled actions.
            baseline = run_experiment(net, demand, Policy.RL_SWARM, CONFIG,
                                      rl_agent=SwarmPolicy(weights))["metrics"]["average_travel_time"]
            agent = SwarmPolicy(weights, stochastic=True, seed=seed+100_000)
            result = run_experiment(net, demand, Policy.RL_SWARM, CONFIG, rl_agent=agent)
            cost = result["metrics"]["average_travel_time"]
            advantage = (baseline-cost)/max(1, shortest)
            gradients.append([advantage*g for g in agent.log_policy_gradient()])
            returns.append(cost)
        gradient = [statistics.mean(g[j] for g in gradients) for j in range(len(weights))]
        optimizer.update(weights, gradient)
        completed = min(first+batch_size, episodes)
        row = {"episodes": completed, "sampled_mean_trip_ticks": statistics.mean(returns), "weights": list(weights)}
        if completed % (batch_size*5) == 0 or completed == episodes:
            score = validation_cost(weights, validation)
            row["validation_mean_trip_ticks"] = score
            if score < best_cost:
                best_cost, best_weights, best_episode = score, list(weights), completed
            print(f"episodes={completed} train={statistics.mean(returns)*TICK_SECONDS/60:.2f} min "
                  f"validation={score*TICK_SECONDS/60:.2f} min", flush=True)
        history.append(row)
    metadata = {"training_episodes": episodes, "selected_episode": best_episode,
                "training_seeds": [10_000, 10_000+episodes-1], "validation_seeds": validation_seeds,
                "test_seeds": test_seeds, "demo_seed": 19, "initial_weights": [0.0]*len(FEATURES),
                "reward": "negative mean completed trip time across ALL vehicles, divided by shortest free-flow time",
                "baseline": "self-critical deterministic rollout of current frozen policy",
                "optimizer": "Adam ascent, learning_rate=0.18, batch=8 by default, gradient norm clip=10",
                "batch_size": batch_size, "inference": "deterministic argmax; no learning during evaluation",
                "validation_mean_trip_ticks": best_cost, "training_config": asdict(CONFIG),
                "scope": "Milpitas–Sunnyvale corridor; held-out demand seeds, not held-out geography",
                "coordination": "shared parameters + current/future route-commitment board; no message-passing network"}
    model = SwarmPolicy(best_weights, metadata=metadata)
    model.save(output / "swarm-model.json")
    (output / "training-history.json").write_text(json.dumps(history, indent=2))
    evaluate(output, model)
    return model


def evaluate(output, model=None):
    output = Path(output)
    model = model or SwarmPolicy.load(output / "swarm-model.json")
    best_weights, metadata = model.weights, model.metadata
    test_seeds = metadata["test_seeds"]
    base, features, origin, destination, _, _ = load_corridors()
    rows = []
    for seed in test_seeds:
        net, demand, kind = scenario(base, features, origin, destination, seed)
        row = {"seed": seed, "scenario": kind, "vehicles": len(demand), "policies": {}}
        for label, policy, agent in [
            ("reactive", Policy.REACTIVE, None),
            ("anticipatory", Policy.ANTICIPATORY, None),
            ("cooperative", Policy.COOPERATIVE, None),
            ("rl_swarm", Policy.RL_SWARM, SwarmPolicy(best_weights)),
            ("rl_no_commitments", Policy.RL_SWARM, SwarmPolicy(best_weights, coordination=False)),
            ("untrained_uniform", Policy.RL_SWARM, SwarmPolicy(stochastic=True, seed=seed+200_000)),
        ]:
            result = run_experiment(net, demand, policy, CONFIG, rl_agent=agent)
            row["policies"][label] = result["metrics"]
        rows.append(row)
        print(f"held-out seed={seed} {kind} RL={row['policies']['rl_swarm']['average_travel_time']*TICK_SECONDS/60:.2f} min", flush=True)
    summary = {}
    for label in rows[0]["policies"]:
        values = [row["policies"][label]["average_travel_time"]*TICK_SECONDS/60 for row in rows]
        summary[label] = {"mean_minutes_across_seeds": statistics.mean(values),
                          "stdev_minutes_across_seeds": statistics.stdev(values)}
    report = {"checkpoint": "swarm-model.json", "metadata": metadata, "summary": summary, "episodes": rows,
              "checkpoint_sha256": hashlib.sha256((output / "swarm-model.json").read_bytes()).hexdigest()}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "rl")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the frozen checkpoint without training or selection")
    args = parser.parse_args()
    try:
        if args.evaluate_only:
            evaluate(args.output)
        else:
            train(args.episodes, args.batch_size, args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
