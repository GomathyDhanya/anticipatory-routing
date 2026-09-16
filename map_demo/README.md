# Milpitas → Sunnyvale: side-by-side map demo

Open `index.html` in a browser. All road geometry, simulation events, scripts,
and styles are bundled locally; replay needs no API keys or internet access.

For a local web preview, from the project root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory map_demo
```

Visit `http://127.0.0.1:8765`. This is local to your computer, not a hosted site.

## See the difference

Each panel has an independent Planning selector: **At departure** or
**At every junction**. The latter reconsiders the remaining route during travel,
with purple rings highlighting changed routes. The description reports the full
run's number of route changes. Road commitment readouts use the route announced
at the displayed time, including subsequent revisions. The RL junction demo
reuses the departure-trained weights; the evaluation table is departure-only.

The first view pauses 15 simulated minutes into the **CA 237 capacity stress**
scenario. Click **Play** to watch both policies continue, or **Restart** to begin
at departure. Playback is user-started. Reduced-motion users start at time zero.

- Choose light demand, a departure surge, or a hypothetical CA 237 capacity limit.
- Compare static/reactive/predictive routing on the left against
  RL swarm/anticipatory/cooperative routing on the right.
- Play, pause, scrub the timeline, or choose 10×–80× playback speed.
- Pan or zoom either map; both views move together.
- Click a modeled road or use the road selector to compare current queues and
  vehicles already assigned to reach it later.
- Read full-run mean travel times separately from current moving/queued/arrived
  counts. The replay is precomputed by the Python simulator, not random motion.

At narrow phone widths the maps stack vertically. At widths over 620px, the maps
remain side by side. The dropdown provides a keyboard-accessible alternative to
clicking map segments.

## Sources and assumptions

**Real:** City-center coordinates from Nominatim; route geometry and base
traversal estimates from OSRM; surrounding road geometry from OpenStreetMap via
Overpass. The base route follows CA 237/Mathilda. Two additional requests use
actual OSM points on Tasman Drive and Montague Expressway as waypoints.

**Simulated:** Demand, vehicles, admission capacities, entry queues, delays, and
the CA 237 capacity restriction. No actual/live traffic was requested or connected.
These trip times must not be used for real-world journey planning.

Six candidate paths are computed on a 59-segment union of those three driving
corridors. Shared directed geometry is represented once, avoiding duplicated
capacity. Additional context roads are displayed but do not participate in the
simulation. Turn restrictions at recombined junctions are not modeled. Geometry
alone does not make this a calibrated or complete road-network model.

The capacity assumptions are deliberately explicit: most CA 237 segments admit
3 vehicles per 10-second tick, other modeled roads 4, and common access roads 24.
The capacity-stress case restricts one CA 237 segment to 1 vehicle per tick.
These values are demonstration parameters, not measured roadway throughput.
Five-sixths of vehicles are controlled; background cars use static routes.
All policies receive the same actual demand. Predictive policies see a rolling
five-minute perfect forecast of background departures only.

Queues are point queues with unlimited storage. Waiting markers cluster at their
entry point so every vehicle remains visible; the clusters do not depict actual
spillback or lane positions. Moving markers interpolate each recorded edge visit
uniformly in geometric distance. Road colors encode modeled entry-queue delay,
not measured vehicle speed. Mouse pan/zoom and road selection affect presentation
only, not the precomputed experiment.

### Rebuild and test

```sh
python3 -m traffic_routing.map_demo
python3 -m unittest discover -s tests -v
```

The build reads the supplied raw cache in `data/` and writes `data/demo.json` and
`data.js`. Rebuilding makes no network requests. The animation's per-vehicle
events are recorded by the existing simulator; tests check all exported runs
against their capacities, queue timings, completion counts, and aggregate metrics.

### Attribution

- Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
  available under the [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/).
- Driving routes: [OSRM](https://project-osrm.org/docs/v26.4.0/http), based on OSM.
- Geocoding: [Nominatim](https://nominatim.openstreetmap.org/).
- Raw map source: `https://overpass-api.de/api/interpreter`.

Raw geometry, source route responses, and the derived geometry are included to
keep the source data and reconstruction available alongside the visualization.

## RL swarm option

The right panel includes **RL swarm coordination**, trained for 480 episodes on
this corridor, with frozen inference in the demo. The left panel also offers
anticipatory and cooperative policies for direct comparisons. The expandable
training section shows held-out evaluation and an inference-only coordination
ablation. See [the RL guide](rl/README.md) for the method, seeds, checkpoint, and
retraining commands. The bundled checkpoint is required by the replay builder;
to replace it, run `python3 -m traffic_routing.train_swarm` before rebuilding.
