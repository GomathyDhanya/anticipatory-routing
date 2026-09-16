from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import random
import statistics

from .model import Demand, Network, Road, positive_integer
from .routing import Planner, Policy
from .simulation import World
from .swarm import SwarmPolicy


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    candidates: int = 8
    forecast_horizon: int = 30
    measurement_window: int = 120
    max_ticks: int = 10_000
    severe_utilization: float = 1.5
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    replan: bool = False

    def __post_init__(self) -> None:
        if type(self.replan) is not bool:
            raise ValueError("replan must be a boolean")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ("candidates", "measurement_window", "max_ticks"):
            positive_integer(getattr(self, name), name)
        positive_integer(self.forecast_horizon, "forecast_horizon", allow_zero=True)
        if not math.isfinite(self.severe_utilization) or self.severe_utilization <= 1:
            raise ValueError("severe_utilization must be finite and greater than one")


def demo_scenario(seed: int = 7) -> tuple[Network, list[Demand]]:
    """Three corridors and downstream bottlenecks; deterministic seeded demand."""
    network = Network([
        Road("oa", "O", "A", 3, 8), Road("ad", "A", "D", 3, 1),
        Road("ob", "O", "B", 4, 8), Road("bd", "B", "D", 4, 2),
        Road("oc", "O", "C", 5, 8), Road("cd", "C", "D", 5, 3),
        Road("ab", "A", "B", 2, 2),
    ])
    rng = random.Random(seed)
    demands = []
    for i in range(120):
        departure = rng.choice((0, 1, 2, 3, 12, 13, 14, 25, 26, 27))
        demands.append(Demand(f"controlled-{i:04}", "O", "D", departure))
    for i in range(30):
        demands.append(Demand(f"background-{i:04}", rng.choice(("A", "B")), "D",
                              rng.randrange(3, 33), False))
    return network, sorted(demands, key=lambda d: (d.departure, d.id))


def run_experiment(network: Network, demands: list[Demand], policy: Policy | str,
                   config: ExperimentConfig | None = None,
                   forecast: list[Demand] | None = None, *, capture_trace: bool = False,
                   rl_agent: SwarmPolicy | None = None) -> dict:
    config = config or ExperimentConfig()
    policy = Policy(policy)
    if policy == Policy.RL_SWARM and rl_agent is None:
        raise ValueError("rl_swarm requires an explicit trained model (rl_agent / --rl-model)")
    if rl_agent is not None:
        rl_agent.episode_gradients.clear()
    ids = [d.id for d in demands]
    if len(set(ids)) != len(ids):
        raise ValueError("Demand ids must be unique")
    if demands and max(d.departure for d in demands) >= config.measurement_window:
        raise ValueError("All departures must occur before measurement_window")
    if config.max_ticks < config.measurement_window:
        raise ValueError("max_ticks must be at least measurement_window")
    # Default forecast is perfect knowledge of external departures within the
    # rolling horizon, not knowledge of future controlled routing requests.
    forecast = [d for d in demands if not d.controlled] if forecast is None else list(forecast)
    if any(d.controlled for d in forecast):
        raise ValueError("Forecast cannot include controlled requests")
    if len({d.id for d in forecast}) != len(forecast):
        raise ValueError("Forecast ids must be unique")
    actual_by_id = {d.id: d for d in demands}
    for d in forecast:
        # Forecast-only vehicles are allowed (false positives). Reusing an
        # actual id for a different event risks counting it twice in rollouts.
        if d.id in actual_by_id and d != actual_by_id[d.id]:
            raise ValueError("A forecast id matching actual demand must describe the same event")
        network.paths(d.origin, d.destination, 1)
    schedule: dict[int, list[Demand]] = {}
    for demand in demands:
        network.paths(demand.origin, demand.destination, config.candidates)
        schedule.setdefault(demand.departure, []).append(demand)
    planner = Planner(network, k=config.candidates, forecast_horizon=config.forecast_horizon,
                      rollout_limit=config.max_ticks, alpha=config.alpha,
                      beta=config.beta, gamma=config.gamma)
    world = World(network, capture=capture_trace)
    decisions = []
    replans = []
    series = []
    last_departure = max(schedule, default=0)
    for t in range(config.max_ticks + 1):
        arriving = sorted(k for k, v in world.active.items()
                          if v.exit_at == t and v.demand.controlled)
        world.release(t)
        batch = sorted(schedule.get(t, []), key=lambda d: d.id)
        for demand in batch:
            if not demand.controlled:
                world.add(demand, network.paths(demand.origin, demand.destination, 1)[0], t)
        if config.replan:
            for key in arriving:
                if key not in world.active:
                    continue
                vehicle = world.active[key]
                prefix = vehicle.path[:vehicle.index]
                old = vehicle.path[vehicle.index:]
                node = network.roads[prefix[-1]].target
                # Geometry-only segment boundaries offer no immediate choice.
                # Reconsider at actual branching junctions, after finishing a road.
                if len(network.outgoing[node]) < 2:
                    continue
                visited = {vehicle.demand.origin} | {network.roads[e].source for e in prefix}
                candidates = tuple(p for p in network.paths(node, vehicle.demand.destination, config.candidates)
                                   if not any(network.roads[e].target in visited for e in p))
                # Retaining the existing suffix always provides a legal fallback.
                candidates = tuple(sorted(set(candidates + (old,)), key=lambda p: (network.travel_time(p), p)))
                if len(candidates) == 1:
                    continue
                snapshot = world.clone()
                del snapshot.active[key]
                snapshot.queues[old[0]].remove(key)
                request = Demand(key, node, vehicle.demand.destination, t)
                decision = (rl_agent.choose(request, snapshot, config.candidates, candidates=candidates)
                            if policy == Policy.RL_SWARM else
                            planner.choose(request, policy, snapshot, snapshot, t, forecast, candidates=candidates))
                changed = decision.path != old
                if changed:
                    network.validate_path(vehicle.demand, prefix + decision.path)
                    vehicle.path = prefix + decision.path
                    if old[0] != decision.path[0]:
                        world.queues[old[0]].remove(key)
                        if capture_trace:
                            world.history[key].pop()  # Provisional junction enqueue; no road entered yet.
                        world._enqueue(vehicle, t)
                replans.append({"id": key, "tick": t, "node": node, "old_path": list(old),
                                "path": list(decision.path), "changed": changed})
        observed = world.clone()
        for demand in batch:
            if not demand.controlled:
                continue
            decision = (rl_agent.choose(demand, world, config.candidates) if policy == Policy.RL_SWARM
                        else planner.choose(demand, policy, observed, world, t, forecast))
            world.add(demand, decision.path, t)
            decisions.append({"id": demand.id, "departure": t, "path": list(decision.path),
                              "predicted_travel_time": decision.predicted_travel_time,
                              "score": decision.score})
        world.serve(t)
        world.assert_consistent()
        occupancy = world.occupancy()
        utilization = {e: count / (network.roads[e].capacity * network.roads[e].free_flow)
                       for e, count in occupancy.items()}
        series.append({"tick": t, "active": len(world.active), "completed": len(world.finished),
                       "queued": sum(len(q) for q in world.queues.values()),
                       "occupancy": occupancy, "utilization": utilization,
                       "overload": world.overload(),
                       "severe_edges": sum(u >= config.severe_utilization for u in utilization.values())})
        if t >= max(last_departure, config.measurement_window) and not world.active:
            break
    if world.active:
        raise RuntimeError(f"{len(world.active)} vehicles unfinished at max_ticks={config.max_ticks}")
    if len(world.finished) != len(demands):
        raise AssertionError("Vehicle conservation failed")
    trips = sorted(world.finished.values(), key=lambda trip: trip.id)
    times = sorted(trip.travel_time for trip in trips)
    controlled = [trip.travel_time for trip in trips if trip.controlled]
    window_completed = sum(trip.arrival <= config.measurement_window for trip in trips)
    overload = sum(row["overload"] for row in series)
    total_time = sum(times)
    total_delay = sum(trip.delay for trip in trips)
    metrics = {
        "vehicles": len(trips), "completed": len(trips),
        "replanning_decisions": len(replans),
        "route_changes": sum(r["changed"] for r in replans),
        "average_travel_time": statistics.mean(times) if times else 0.0,
        "controlled_average_travel_time": statistics.mean(controlled) if controlled else 0.0,
        "p95_travel_time": times[math.ceil(0.95 * len(times)) - 1] if times else 0,
        "total_travel_time": total_time, "total_delay": total_delay,
        "peak_utilization": max((u for row in series for u in row["utilization"].values()), default=0.0),
        "peak_severely_congested_edges": max((row["severe_edges"] for row in series), default=0),
        "severe_edge_ticks": sum(row["severe_edges"] for row in series),
        "overload_vehicle_ticks": overload,
        "window_completed": window_completed,
        "window_throughput": window_completed / config.measurement_window,
        "last_arrival": max((trip.arrival for trip in trips), default=0),
        "objective": config.alpha * total_time + config.beta * total_delay + config.gamma * overload,
    }
    result = {"policy": policy.value, "config": asdict(config), "metrics": metrics,
            "route_counts": dict(sorted(Counter(" → ".join(d["path"]) or "Already at destination"
                                                 for d in decisions).items())),
            "decisions": decisions,
            "replans": replans,
            "trips": [dict(asdict(trip), travel_time=trip.travel_time, delay=trip.delay) for trip in trips],
            "series": series}
    if capture_trace:
        result["trajectories"] = world.history
    return result
