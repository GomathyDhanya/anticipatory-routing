from __future__ import annotations

from dataclasses import dataclass
import heapq


def positive_integer(value: int, name: str, *, allow_zero: bool = False) -> None:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} integer")


@dataclass(frozen=True)
class Road:
    id: str
    source: str
    target: str
    free_flow: int
    capacity: int

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) and v for v in (self.id, self.source, self.target)):
            raise ValueError("Road identifiers and endpoints must be nonempty strings")
        if self.source == self.target:
            raise ValueError("Self-loop roads are not supported")
        positive_integer(self.free_flow, "free_flow")
        positive_integer(self.capacity, "capacity")


@dataclass(frozen=True)
class Demand:
    id: str
    origin: str
    destination: str
    departure: int
    controlled: bool = True

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) and v for v in (self.id, self.origin, self.destination)):
            raise ValueError("Demand identifiers and endpoints must be nonempty strings")
        positive_integer(self.departure, "departure", allow_zero=True)
        if type(self.controlled) is not bool:
            raise ValueError("controlled must be a boolean")


class Network:
    def __init__(self, roads: list[Road]):
        if not roads:
            raise ValueError("Network must contain at least one road")
        self.roads = {r.id: r for r in roads}
        if len(self.roads) != len(roads):
            raise ValueError("Duplicate road id")
        self.nodes = {n for r in roads for n in (r.source, r.target)}
        self.outgoing: dict[str, list[Road]] = {n: [] for n in self.nodes}
        for road in sorted(roads, key=lambda r: r.id):
            self.outgoing[road.source].append(road)
        self._paths: dict[tuple[str, str, int], tuple[tuple[str, ...], ...]] = {}

    def paths(self, origin: str, destination: str, k: int = 8) -> tuple[tuple[str, ...], ...]:
        """Enumerate the k cheapest simple paths by free-flow time, deterministically.

        Parallel roads are supported. An explicit expansion budget prevents silent
        combinatorial hangs; this prototype is intended for small networks.
        """
        positive_integer(k, "k")
        if origin not in self.nodes or destination not in self.nodes:
            raise ValueError(f"Unknown endpoint: {origin} -> {destination}")
        key = (origin, destination, k)
        if key in self._paths:
            return self._paths[key]
        # Yen's algorithm avoids enumerating every partial simple path in a city
        # grid. Each spur search uses Dijkstra with explicitly removed nodes/edges.
        expanded = 0
        def shortest(start: str, banned_nodes: set[str], banned_edges: set[str]):
            nonlocal expanded
            heap = [(0, (), start)]
            settled = set()
            while heap:
                cost, path, node = heapq.heappop(heap)
                if node in settled:
                    continue
                settled.add(node)
                if node == destination:
                    return path
                expanded += 1
                if expanded > 100_000:
                    raise ValueError("Path search exceeded 100,000 expansions; reduce the network or k")
                for road in self.outgoing[node]:
                    if road.id not in banned_edges and road.target not in banned_nodes and road.target not in settled:
                        heapq.heappush(heap, (cost + road.free_flow, path + (road.id,), road.target))
            return None

        first = shortest(origin, set(), set())
        if first is None:
            raise ValueError(f"No route: {origin} -> {destination}")
        found = [first]
        options = []
        queued = {first}
        while len(found) < k:
            previous = found[-1]
            nodes = [origin] + [self.roads[e].target for e in previous]
            for index in range(len(previous)):
                root = previous[:index]
                banned_edges = {p[index] for p in found if len(p) > index and p[:index] == root}
                spur = shortest(nodes[index], set(nodes[:index]), banned_edges)
                if spur is not None:
                    candidate = root + spur
                    if candidate not in queued:
                        queued.add(candidate)
                        heapq.heappush(options, (self.travel_time(candidate), candidate))
            if not options:
                break
            found.append(heapq.heappop(options)[1])
        self._paths[key] = tuple(found)
        return self._paths[key]

    def travel_time(self, path: tuple[str, ...]) -> int:
        return sum(self.roads[e].free_flow for e in path)

    def validate_path(self, demand: Demand, path: tuple[str, ...]) -> None:
        if demand.origin not in self.nodes or demand.destination not in self.nodes:
            raise ValueError("Unknown demand endpoint")
        node = demand.origin
        visited = {node}
        for edge in path:
            if edge not in self.roads or self.roads[edge].source != node:
                raise ValueError(f"Disconnected or unknown road in route: {edge}")
            node = self.roads[edge].target
            if node in visited:
                raise ValueError("Routes must be simple (no repeated nodes)")
            visited.add(node)
        if node != demand.destination:
            raise ValueError("Route does not reach destination")
