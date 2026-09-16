"""Shared-parameter multi-agent REINFORCE policy with a commitment board.

Each controlled vehicle is an agent choosing one route. All agents use the same
learned softmax policy and announce their route through the live World. Their
shared terminal reward is negative network mean trip time. This is swarm-style
coordination through shared road signals, not a decentralized communications model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random

from .model import Demand, Network
from .routing import Decision
from .simulation import World

FEATURES = ("free_flow_excess", "observed_queue_delay", "peak_committed_work",
            "mean_committed_work", "peak_committed_work_squared")


def softmax(scores: list[float]) -> list[float]:
    if not scores or not all(math.isfinite(s) for s in scores):
        raise ValueError("Policy scores must be nonempty and finite")
    maximum = max(scores)
    exponentials = [math.exp(s - maximum) for s in scores]
    total = sum(exponentials)
    return [v / total for v in exponentials]


def route_features(network: Network, world: World, candidates: tuple[tuple[str, ...], ...],
                   *, coordination: bool = True) -> list[list[float]]:
    """Current road queues plus future route announcements; no future requests.

    The board counts vehicles still committed to each edge, including physically
    present vehicles. Values are coarse workloads, not an exact ETA forecast.
    Free-flow route time normalizes signals across differently sized networks.
    """
    base = max(1, min(network.travel_time(p) for p in candidates))
    board = {e: 0 for e in network.roads}
    for vehicle in world.active.values():
        edges = vehicle.path[vehicle.index:] if coordination else vehicle.path[vehicle.index:vehicle.index+1]
        for edge in edges:
            board[edge] += 1
    result = []
    for path in candidates:
        free_flow = network.travel_time(path)
        observed = sum(len(world.queues[e]) / network.roads[e].capacity for e in path) / base
        loads = [board[e] / network.roads[e].capacity / base for e in path]
        peak = max(loads, default=0.0)
        mean = sum(load * network.roads[e].free_flow for load, e in zip(loads, path)) / max(1, free_flow)
        result.append([free_flow / base - 1, observed, peak, mean, peak * peak])
    return result


@dataclass
class SwarmPolicy:
    weights: list[float] = field(default_factory=lambda: [0.0] * len(FEATURES))
    stochastic: bool = False
    seed: int = 0
    coordination: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.weights = list(self.weights)
        if len(self.weights) != len(FEATURES) or not all(type(w) in (int, float) and math.isfinite(w) for w in self.weights):
            raise ValueError(f"Model requires {len(FEATURES)} finite numeric weights")
        self.rng = random.Random(self.seed)
        self.episode_gradients: list[list[float]] = []
        self.last_probabilities: list[float] = []

    def choose(self, demand: Demand, world: World, k: int, *, candidates=None) -> Decision:
        network = world.network
        candidates = candidates or network.paths(demand.origin, demand.destination, k)
        features = route_features(network, world, candidates, coordination=self.coordination)
        logits = [sum(w*x for w, x in zip(self.weights, row)) for row in features]
        probabilities = softmax(logits)
        self.last_probabilities = probabilities
        if self.stochastic:
            action = self.rng.choices(range(len(candidates)), weights=probabilities, k=1)[0]
            expectation = [sum(p * row[j] for p, row in zip(probabilities, features)) for j in range(len(FEATURES))]
            self.episode_gradients.append([features[action][j] - expectation[j] for j in range(len(FEATURES))])
        else:
            # Stable ties follow candidate order (shortest free-flow path first).
            action = max(range(len(candidates)), key=lambda i: logits[i])
        path = candidates[action]
        eta = network.travel_time(path) + sum(len(world.queues[e]) / network.roads[e].capacity for e in path)
        return Decision(path, eta, logits[action])

    def log_policy_gradient(self) -> list[float]:
        return [sum(row[j] for row in self.episode_gradients) for j in range(len(FEATURES))]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "algorithm": "parameter-sharing REINFORCE", "features": list(FEATURES),
                   "weights": self.weights, "metadata": self.metadata}
        path.write_text(json.dumps(payload, indent=2, allow_nan=False))

    @classmethod
    def load(cls, path: Path) -> SwarmPolicy:
        payload = json.loads(path.read_text())
        if payload.get("version") != 1 or payload.get("features") != list(FEATURES):
            raise ValueError("Unsupported RL checkpoint version or feature schema")
        if not isinstance(payload.get("metadata", {}), dict):
            raise ValueError("Checkpoint metadata must be an object")
        return cls(payload["weights"], metadata=payload.get("metadata", {}))


class AdamAscent:
    """Adam on an ascent gradient; the learner supplies actual episode returns."""
    def __init__(self, size: int, learning_rate: float = .04):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.learning_rate = learning_rate
        self.m = [0.0] * size
        self.v = [0.0] * size
        self.steps = 0

    def update(self, weights: list[float], gradient: list[float]) -> None:
        if len(gradient) != len(weights) or len(weights) != len(self.m) or not all(math.isfinite(g) for g in gradient):
            raise ValueError("Invalid gradient")
        self.steps += 1
        # Norm clipping limits variance from unusually congested episodes.
        norm = math.sqrt(sum(g*g for g in gradient))
        factor = min(1.0, 10.0 / max(norm, 1e-12))
        for j, grad in enumerate(gradient):
            grad *= factor
            self.m[j] = .9*self.m[j] + .1*grad
            self.v[j] = .999*self.v[j] + .001*grad*grad
            m_hat = self.m[j] / (1-.9**self.steps)
            v_hat = self.v[j] / (1-.999**self.steps)
            weights[j] += self.learning_rate*m_hat/(math.sqrt(v_hat)+1e-8)
