from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

from .model import Demand, Network


@dataclass
class Vehicle:
    demand: Demand
    path: tuple[str, ...]
    index: int = 0
    exit_at: int | None = None


@dataclass(frozen=True)
class Trip:
    id: str
    controlled: bool
    departure: int
    arrival: int
    path: tuple[str, ...]
    free_flow: int

    @property
    def travel_time(self) -> int:
        return self.arrival - self.departure

    @property
    def delay(self) -> int:
        return self.travel_time - self.free_flow


class World:
    """FIFO point queues, fixed transit time, integer entry capacity per tick.

    Tick order: release transit, enqueue departures, serve queues. A vehicle
    admitted at t exits at t + free_flow. No overtaking within an edge queue.
    Queue storage is unbounded; there is no physical spillback model.
    """

    def __init__(self, network: Network, *, capture: bool = False):
        self.network = network
        self.active: dict[str, Vehicle] = {}
        self.finished: dict[str, Trip] = {}
        self.queues: dict[str, deque[str]] = {e: deque() for e in network.roads}
        self.capture = capture
        self.history: dict[str, list[list]] = {}

    def _enqueue(self, vehicle: Vehicle, t: int) -> None:
        edge = vehicle.path[vehicle.index]
        self.queues[edge].append(vehicle.demand.id)
        if self.capture:
            self.history.setdefault(vehicle.demand.id, []).append([edge, t, None, None])

    def clone(self, *, commitments: bool = True) -> World:
        other = World(self.network)
        for key, vehicle in self.active.items():
            path = vehicle.path if commitments else vehicle.path[:vehicle.index + 1]
            other.active[key] = replace(vehicle, path=path)
        other.queues = {e: deque(queue) for e, queue in self.queues.items()}
        # Completed vehicles cannot affect future traffic or marginal costs.
        return other

    def add(self, demand: Demand, path: tuple[str, ...], t: int) -> None:
        if demand.id in self.active or demand.id in self.finished:
            raise ValueError(f"Duplicate vehicle id: {demand.id}")
        if demand.departure != t:
            raise ValueError("Vehicles must be added at their departure tick")
        self.network.validate_path(demand, path)
        vehicle = Vehicle(demand, path)
        if not path:
            self._finish(vehicle, t)
        else:
            self.active[demand.id] = vehicle
            self._enqueue(vehicle, t)

    def _finish(self, vehicle: Vehicle, t: int) -> None:
        demand = vehicle.demand
        self.finished[demand.id] = Trip(demand.id, demand.controlled, demand.departure,
                                       t, vehicle.path, self.network.travel_time(vehicle.path))

    def release(self, t: int) -> None:
        # Stable tie breaking for vehicles reaching a junction simultaneously.
        arriving = sorted(k for k, v in self.active.items() if v.exit_at == t)
        for key in arriving:
            vehicle = self.active[key]
            vehicle.index += 1
            vehicle.exit_at = None
            if vehicle.index == len(vehicle.path):
                self._finish(vehicle, t)
                del self.active[key]
            else:
                self._enqueue(vehicle, t)

    def serve(self, t: int) -> None:
        for edge, road in self.network.roads.items():
            queue = self.queues[edge]
            for _ in range(min(road.capacity, len(queue))):
                key = queue.popleft()
                self.active[key].exit_at = t + road.free_flow
                if self.capture:
                    self.history[key][-1][2:] = [t, t + road.free_flow]

    def occupancy(self) -> dict[str, int]:
        counts = {e: 0 for e in self.network.roads}
        for vehicle in self.active.values():
            counts[vehicle.path[vehicle.index]] += 1
        return counts

    def overload(self) -> int:
        return sum(max(0, count - self.network.roads[e].capacity * self.network.roads[e].free_flow)
                   for e, count in self.occupancy().items())

    def assert_consistent(self) -> None:
        queued = [key for queue in self.queues.values() for key in queue]
        assert len(queued) == len(set(queued)), "Vehicle queued more than once"
        assert set(queued) == {k for k, v in self.active.items() if v.exit_at is None}
        assert not (self.active.keys() & self.finished.keys())
        for edge, queue in self.queues.items():
            assert all(self.active[k].path[self.active[k].index] == edge for k in queue)
