from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

from .model import Demand, Network, positive_integer
from .simulation import World


class Policy(StrEnum):
    STATIC = "static"
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    ANTICIPATORY = "anticipatory"
    COOPERATIVE = "cooperative"
    RL_SWARM = "rl_swarm"


@dataclass(frozen=True)
class Decision:
    path: tuple[str, ...]
    predicted_travel_time: float
    score: float


class Planner:
    def __init__(self, network: Network, *, k: int = 8, forecast_horizon: int = 30,
                 rollout_limit: int = 10_000, alpha: float = 1.0, beta: float = 0.0,
                 gamma: float = 0.0):
        positive_integer(k, "k")
        positive_integer(forecast_horizon, "forecast_horizon", allow_zero=True)
        positive_integer(rollout_limit, "rollout_limit")
        for name, weight in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if alpha + beta + gamma == 0:
            raise ValueError("At least one objective weight must be positive")
        self.network = network
        self.k = k
        self.forecast_horizon = forecast_horizon
        self.rollout_limit = rollout_limit
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def _rollout(self, world: World, t: int, future: list[Demand]) -> tuple[World, int]:
        schedule: dict[int, list[Demand]] = {}
        for demand in future:
            schedule.setdefault(demand.departure, []).append(demand)
        last = max(schedule, default=t)
        overload = 0
        # The input is already released and populated at t, but not served.
        for now in range(t, t + self.rollout_limit + 1):
            if now > t:
                world.release(now)
                for demand in sorted(schedule.get(now, []), key=lambda d: d.id):
                    world.add(demand, self.network.paths(demand.origin, demand.destination, 1)[0], now)
            world.serve(now)
            overload += world.overload()
            if not world.active and now >= last:
                return world, overload
        raise RuntimeError("Forecast rollout did not drain; increase rollout_limit")

    def choose(self, demand: Demand, policy: Policy | str, observed: World,
               committed: World, t: int, forecast: list[Demand], *, candidates=None) -> Decision:
        policy = Policy(policy)
        if policy == Policy.RL_SWARM:
            raise ValueError("RL swarm requires a SwarmPolicy; pass rl_agent to run_experiment")
        candidates = candidates or self.network.paths(demand.origin, demand.destination, self.k)
        if policy == Policy.STATIC:
            path = candidates[0]
            cost = self.network.travel_time(path)
            return Decision(path, cost, cost)
        if policy == Policy.REACTIVE:
            # Hold current queue measurements fixed across the whole path.
            # floor(q/c) is the entry delay for a new last-in-queue vehicle.
            def current_cost(path: tuple[str, ...]) -> int:
                return sum(self.network.roads[e].free_flow
                           + len(observed.queues[e]) // self.network.roads[e].capacity for e in path)
            path = min(candidates, key=lambda p: (current_cost(p), p))
            cost = current_cost(path)
            return Decision(path, cost, cost)

        future = [d for d in forecast if t < d.departure <= t + self.forecast_horizon]
        if any(d.controlled for d in future):
            raise ValueError("Exogenous forecasts must contain background (uncontrolled) demand only")
        # Predictive knows physical positions, but not downstream route intentions.
        # Commitment-aware policies retain every assigned vehicle's full suffix.
        source = observed if policy == Policy.PREDICTIVE else committed
        best: Decision | None = None
        for path in candidates:
            simulation = source.clone(commitments=policy != Policy.PREDICTIVE)
            simulation.add(demand, path, t)
            simulation, overload = self._rollout(simulation, t, future)
            own_time = simulation.finished[demand.id].travel_time
            if policy == Policy.COOPERATIVE:
                # The baseline-without-candidate objective is constant for all
                # candidate paths. Minimizing this total is equivalent to
                # minimizing the marginal objective, without an extra rollout.
                total_time = sum(trip.travel_time for trip in simulation.finished.values())
                total_delay = sum(trip.delay for trip in simulation.finished.values())
                score = self.alpha * total_time + self.beta * total_delay + self.gamma * overload
            else:
                score = float(own_time)
            decision = Decision(path, float(own_time), float(score))
            if best is None or (decision.score, decision.path) < (best.score, best.path):
                best = decision
        assert best is not None
        return best
