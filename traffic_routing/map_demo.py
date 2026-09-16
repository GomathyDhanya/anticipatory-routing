"""Build an offline map replay using cached OSM / OSRM geometry.

Run: python3 -m traffic_routing.map_demo
No network calls occur during replay or building from the supplied cache.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import random

from .experiment import ExperimentConfig, run_experiment
from .model import Demand, Network, Road
from .routing import Policy
from .swarm import FEATURES, SwarmPolicy

ROOT = Path(__file__).resolve().parent.parent / "map_demo"
TICK_SECONDS = 10


def distance(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6_371_000 * 2 * math.asin(min(1, math.sqrt(h)))


def coordinate_id(point):
    return f"{point[0]:.6f},{point[1]:.6f}"


def load_corridors(data_dir: Path = ROOT / "data"):
    """Union three sourced, directed driving routes and merge geometry-only nodes.

    Shared directed coordinate pairs become one physical segment, so overlapping
    route options cannot manufacture additional road capacity.
    """
    atoms = {}
    coords = {}
    endpoints = []
    source_routes = []
    for filename, label in (("routes.json", "CA 237 / Mathilda"),
                            ("routes-tasman.json", "Via Tasman Drive"),
                            ("routes-montague.json", "Via Montague Expressway")):
        response = json.loads((data_dir / filename).read_text())
        if response.get("code") != "Ok" or not response.get("routes"):
            raise ValueError(f"No valid OSRM route in {filename}")
        route = response["routes"][0]
        points = route["geometry"]["coordinates"]
        durations = [v for leg in route["legs"] for v in leg["annotation"]["duration"]]
        if len(durations) != len(points)-1:
            raise ValueError("OSRM duration annotations do not align with full geometry")
        names = {}
        for leg in route["legs"]:
            for step in leg["steps"]:
                step_points = step["geometry"]["coordinates"]
                for a, b in zip(step_points, step_points[1:]):
                    names[(coordinate_id(a), coordinate_id(b))] = (step.get("name") or step.get("ref") or "Connector", step.get("ref", ""))
        endpoints.append((coordinate_id(points[0]), coordinate_id(points[-1])))
        source_routes.append({"label": label, "distance_m": route["distance"],
                              "duration_s": route["duration"], "file": filename})
        for i, (a, b) in enumerate(zip(points, points[1:])):
            u, v = coordinate_id(a), coordinate_id(b)
            if u == v:
                continue
            coords[u], coords[v] = a, b
            name, ref = names.get((u, v), ("Connector", ""))
            atoms.setdefault((u, v), {"seconds": durations[i], "name": name, "ref": ref,
                                      "meters": distance(a, b)})
    if len(set(endpoints)) != 1:
        raise ValueError("Route options must share snapped endpoints")
    origin, destination = endpoints[0]
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for u, v in atoms:
        outgoing[u].append(v)
        incoming[v].append(u)
    cuts = {origin, destination}
    for node in coords:
        if len(incoming[node]) != 1 or len(outgoing[node]) != 1:
            cuts.add(node)
        elif atoms[(incoming[node][0], node)]["name"] != atoms[(node, outgoing[node][0])]["name"]:
            cuts.add(node)
    # Add shared split points on long stretches. Every route traversing that
    # physical segment sees the same split, duration, and capacity.
    for start in sorted(cuts.copy()):
        for next_node in outgoing[start]:
            u, v, meters, visited = start, next_node, 0.0, {start}
            while True:
                meters += atoms[(u, v)]["meters"]
                if v in cuts or v in visited:
                    break
                visited.add(v)
                if meters >= 900:
                    cuts.add(v)
                    meters = 0.0
                u, v = v, outgoing[v][0]
    roads, features = [], {}
    for start in sorted(cuts):
        for next_node in sorted(outgoing[start]):
            u, v = start, next_node
            geometry, seconds, meters = [coords[u]], 0.0, 0.0
            first = atoms[(u, v)]
            visited = {u}
            while True:
                atom = atoms[(u, v)]
                seconds += atom["seconds"]
                meters += atom["meters"]
                geometry.append(coords[v])
                if v in cuts or v in visited:
                    break
                visited.add(v)
                u, v = v, outgoing[v][0]
            if start == v:
                continue
            edge_id = f"road-{len(roads):03}"
            # Demonstration admission limits, not measured lane capacities.
            # Source/destination access roads are deliberately unconstrained.
            is_highway = "237" in first["ref"] or first["name"] == "CA 237"
            capacity = 3 if is_highway else 4
            if distance(coords[start], coords[origin]) < 1800 or distance(coords[v], coords[destination]) < 900:
                capacity = 24
            roads.append(Road(edge_id, start, v, max(1, round(seconds / TICK_SECONDS)), capacity))
            features[edge_id] = {"id": edge_id, "name": first["name"], "ref": first["ref"],
                                 "geometry": geometry, "meters": round(meters), "capacity": capacity,
                                 "free_flow": roads[-1].free_flow, "is_highway": is_highway}
    network = Network(roads)
    # Drop disused branches from routing rollouts; keep full sourced map context.
    paths = network.paths(origin, destination, 6)
    used = {e for path in paths for e in path}
    network = Network([r for r in roads if r.id in used])
    features = {e: f for e, f in features.items() if e in used}
    return network, features, origin, destination, coords, source_routes


def build():
    network, features, origin, destination, coords, source_routes = load_corridors()
    model_path = ROOT / "rl/swarm-model.json"
    if not model_path.exists():
        raise ValueError("Train the swarm model first: python3 -m traffic_routing.train_swarm")
    swarm = SwarmPolicy.load(model_path)
    evaluation = json.loads((ROOT / "rl/evaluation.json").read_text())
    if evaluation.get("checkpoint_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest():
        raise ValueError("RL evaluation does not match checkpoint; run train_swarm --evaluate-only")
    raw = json.loads((ROOT / "data/osm-roads.json").read_text())
    basemap = []
    # Keep sourced context, with a small margin around the commute.
    west, south, east, north = -122.051, 37.355, -121.886, 37.442
    for way in raw["elements"]:
        geometry = way.get("geometry", [])
        tags = way.get("tags", {})
        if not any(west <= p["lon"] <= east and south <= p["lat"] <= north for p in geometry):
            continue
        basemap.append({"name": tags.get("name", ""), "ref": tags.get("ref", ""),
                        "kind": tags.get("highway", ""),
                        "points": [[round(p["lon"], 6), round(p["lat"], 6)] for p in geometry]})
    # A single labeled segment per important roadway avoids repeated labels.
    labels = []
    for name in ("Tasman Drive", "Montague Expressway", "Lawrence Expressway", "North Mathilda Avenue",
                 "Great America Parkway", "North First Street"):
        matches = [r for r in basemap if r["name"] == name]
        if matches:
            item = max(matches, key=lambda r: sum(distance(a, b) for a, b in zip(r["points"], r["points"][1:])))
            labels.append({"name": name, "point": item["points"][len(item["points"])//2]})
    payload = {"title": "Milpitas → Sunnyvale", "tickSeconds": TICK_SECONDS,
               "origin": coords[origin], "destination": coords[destination],
               "bounds": [west, south, east, north], "basemap": basemap, "labels": labels,
               "roads": features, "sourceRoutes": source_routes,
               "sources": {"geometry": "OpenStreetMap contributors / OSRM", "license": "ODbL",
                           "osmTimestamp": raw.get("osm3s", {}).get("timestamp_osm_base"),
                           "builtAt": datetime.now(timezone.utc).isoformat(),
                           "traffic": "Simulated; no live traffic feed"}, "scenes": [],
               "rl": {"metadata": swarm.metadata, "summary": evaluation["summary"],
                      "features": list(FEATURES),
                      "weights": swarm.weights}}
    bottlenecks = sorted((e for e, f in features.items() if f["is_highway"] and f["capacity"] < 24),
                         key=lambda e: features[e]["meters"], reverse=True)
    if not bottlenecks:
        raise ValueError("No CA 237 segment found for the capacity-stress scene")
    bottleneck = bottlenecks[0]
    for scene_id, label, total, spread, restrict in [
            ("light", "Light commute", 72, 48, False),
            ("surge", "Departure surge", 210, 18, False),
            ("restriction", "CA 237 capacity stress", 210, 18, True)]:
        rng = random.Random(19)
        demands = [Demand(f"car-{i:04}", origin, destination, rng.randrange(spread), i % 6 != 0)
                   for i in range(total)]
        roads = [Road(r.id, r.source, r.target, r.free_flow, 1 if restrict and r.id == bottleneck else r.capacity)
                 for r in network.roads.values()]
        scenario_network = Network(roads)
        scene = {"id": scene_id, "label": label, "vehicles": total, "departureWindow": spread,
                 "bottleneck": bottleneck if restrict else None, "policies": {}, "replanning": {},
                 "capacities": {r.id: r.capacity for r in roads}}
        for policy, replan in ((p, mode) for mode in (False, True) for p in Policy):
            result = run_experiment(scenario_network, demands, policy,
                                    ExperimentConfig(seed=19, candidates=6, forecast_horizon=30,
                                                     measurement_window=300, max_ticks=2000, replan=replan), capture_trace=True,
                                    rl_agent=swarm if policy == Policy.RL_SWARM else None)
            series = result["series"][:result["metrics"]["last_arrival"] + 1]
            scene["replanning" if replan else "policies"][policy.value] = {
                "replans": result["replans"],
                "metrics": result["metrics"],
                "series": [[r["completed"], r["queued"], r["active"]] for r in series],
                "vehicles": [{"id": trip["id"], "departure": trip["departure"], "arrival": trip["arrival"],
                              "controlled": trip["controlled"], "delay": trip["delay"],
                              "initialPath": next((d["path"] for d in result["decisions"] if d["id"] == trip["id"]), list(trip["path"])),
                              "visits": result["trajectories"].get(trip["id"], [])} for trip in result["trips"]]}
            print(scene_id, policy.value, "replan" if replan else "departure", round(result["metrics"]["average_travel_time"] * TICK_SECONDS / 60, 2), flush=True)
        payload["scenes"].append(scene)
    (ROOT / "data/demo.json").write_text(json.dumps(payload, separators=(",", ":")))
    # Embed cached data so the demo works without a server or network requests.
    serialized = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    (ROOT / "data.js").write_text("window.TRAFFIC_DEMO=" + serialized + ";\n")
    from .demo_reports import build as build_reports
    build_reports(ROOT)
    print(f"Built {len(features)} modeled segments and {len(basemap)} context roads")


if __name__ == "__main__":
    build()
