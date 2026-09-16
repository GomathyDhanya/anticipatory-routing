# Anticipatory traffic routing

A working Python research prototype for routing around the congestion created by
already assigned routes. Includes a deterministic simulator, six policies,
custom scenarios, a standalone HTML comparison report, CSV metrics, full JSON
traces, and automated tests. Python 3.11+; no runtime dependencies.

## Real-road visual demo: Milpitas → Sunnyvale

Open [map_demo/index.html](map_demo/index.html) for synchronized, side-by-side
vehicle animation on actual OpenStreetMap road geometry. Choose three demand /
capacity scenarios and compare all six policies. Vehicles, queues, and trip
times are simulated; there is no live traffic feed. Play, pause, scrub, pan,
zoom, and inspect future route commitments. See [the demo guide](map_demo/README.md)
for sources, modeling assumptions, and rebuilding instructions.

The replay uses recorded events from this Python simulator. It works offline
from the included files. To preview through a local web server:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory map_demo
```

Open `http://127.0.0.1:8765`. The bundled visual demo is a concrete counterfactual
experiment, not real-time navigation or an estimate of today's commute.

## Run

Mid-trip planning is available for every policy. In the map demo, choose
**At every junction** independently on each side; purple rings mark route changes.
For a CLI experiment, run `python3 -m traffic_routing --replan --output results-replanning`.
The default CLI mode retains departure-only routing for reproducible older results.

At each branching junction, controlled vehicles reconsider the remaining route using current
queues and updated commitments. Candidates are regenerated from that junction;
previously visited nodes are excluded, and the old suffix remains a legal option.
Vehicles finish their current road before reconsidering and do not abandon an
entry queue. Simultaneous arrivals plan in vehicle-ID order. Background vehicles
retain their routes. Static routing still minimizes free-flow time, so it normally
keeps its route. Predictive rollouts assume current commitments remain fixed
within each forecast; the next branching junction triggers a fresh plan.
Geometry-only segment boundaries with a single outgoing road do not trigger planning.

JSON results record every nontrivial junction decision in `replans`, with old/new
suffixes and whether the route changed. Trip paths and animation traces contain
the roads actually traversed. The RL checkpoint is reused at junctions; its
published training and held-out scores still describe departure-only routing,
so these new runs are an evaluation of transfer without retraining.

From this directory:

```sh
python3 -m traffic_routing --compare-planning --rl-model map_demo/rl/swarm-model.json --output results
python3 -m unittest discover -s tests -v
```

Open `results/report.html` in a browser. No server, account, API key, or network
connection is needed. The included `results/` contains a completed demo run.

Optional installation: `python3 -m pip install -e .` enables the
`traffic-routing` command. Installation may fetch the setuptools build dependency;
running directly from this directory does not install anything.

## Policies

| Policy | Information and decision rule |
| --- | --- |
| `static` | Lowest free-flow travel time. |
| `reactive` | Lowest sum of free-flow time and current entry-queue delay, holding today's measurements fixed across the route. |
| `predictive` | Forward simulation of physically present vehicles on their current road, plus forecast external demand; minimize the new vehicle's predicted trip time. Downstream intentions are hidden. |
| `anticipatory` | Same forecast, retaining all assigned downstream routes and earlier commitments in the same request batch; minimize the new vehicle's predicted trip time. |
| `cooperative` | Same commitment-aware simulation, minimizing predicted total weighted travel time, delay, and overload across affected vehicles. |
| `rl_swarm` | Shared learned route policy: each vehicle observes road commitments and selects a route using frozen weights trained on network-wide returns. |

The cooperative baseline is a model-based controller. The sixth policy, `rl_swarm`, is an actual trained, shared-parameter reinforcement-learning policy. See [RL method and evaluation](map_demo/rl/README.md).
It trades off an individual's travel time against costs imposed on other vehicles.
It is greedy over the currently known vehicles and the candidate paths, not a
global optimizer over all future requests. Improvements are not guaranteed.

### How commitments work

The live world stores each assigned vehicle's full route, current edge, queue
position, and scheduled edge exit. Planning clones this state. The anticipatory
rollout moves those vehicles along their assigned downstream routes, so their
future arrivals consume bottleneck capacity before evaluating the new route.
There is no independent reservation ledger to become stale or double-count the
same vehicle. A new assignment is added immediately to the live planning state.

Requests departing at the same tick form a batch. Static, reactive, and predictive
policies share the same pre-batch observation. Anticipatory and cooperative
policies additionally see earlier route commitments in that batch. This is an
explicit synchronous-observation assumption: reactive routing cannot treat an
unexecuted assignment as newly observed traffic. Requests are processed in ID
order for reproducibility; experiments should also vary arrival order.

Each candidate is rolled forward until every included vehicle completes its
trip. The forecast horizon limits **which future external departures are known**;
it does not stop the simulation prematurely or discard committed vehicles.
Candidate evaluation never modifies the live state; the selected plan updates
the live route. The engine runs sequentially in one process.

## Traffic and forecast assumptions

- Directed roads have integer free-flow duration and integer vehicle admissions
  per tick. Choose the tick duration to match the desired units before supplying
  data. The demo uses abstract ticks, not calibrated real-world minutes.
- Every road has a FIFO entry queue with unlimited storage. Admitted vehicles
  spend exactly the free-flow duration on the road. There is no spillback,
  intersection signal model, lane changing, stochastic incidents, or overtaking
  within an entry queue. Capacity is an admission rate, not maximum occupancy.
- Each tick releases completed road traversals, enqueues background departures,
  replans arriving controlled vehicles when enabled, assigns controlled departures,
  then admits queued vehicles to roads.
  Simultaneous road exits use vehicle ID order; existing queue members keep priority.
- Background vehicles use fixed free-flow shortest paths under every policy.
- By default, forecast background departures are exact within a rolling horizon.
  This is an idealized forecast, explicitly separate from knowing future
  controlled requests, which are never exposed to the planner.
- A custom `forecast` list can omit actual background vehicles or include
  forecast-only vehicles. `forecast: []` disables external demand prediction.
  All forecast entries must have `controlled: false`. An ID shared with actual
  demand must describe the same event; use distinct IDs for alternative estimates.
- Candidate routes are the `k` shortest simple paths by free-flow time, including
  parallel roads. The search stops with an explicit error at 100,000 expansions.
  A congested network's best route may be outside the candidate set.
- Assigned routes are fixed in departure mode; `--replan` revises remaining
  routes at junctions. No traffic feed, GPS navigation, or production service is included.

## Custom scenario

```json
{
  "roads": [
    {"id": "oa", "source": "O", "target": "A", "free_flow": 2, "capacity": 8},
    {"id": "ad", "source": "A", "target": "D", "free_flow": 2, "capacity": 1},
    {"id": "od", "source": "O", "target": "D", "free_flow": 6, "capacity": 3}
  ],
  "demands": [
    {"id": "vehicle-1", "origin": "O", "destination": "D", "departure": 0},
    {"id": "background-1", "origin": "A", "destination": "D", "departure": 2, "controlled": false}
  ]
}
```

Save as `scenario.json`, then:

```sh
python3 -m traffic_routing --scenario scenario.json --output custom-results
python3 -m traffic_routing --seed 42 --forecast-horizon 0 --output no-forecast
python3 -m traffic_routing --candidates 4 --alpha 1 --beta 0.5 --gamma 2 --output weighted
python3 -m traffic_routing --help
```

`controlled` defaults to true. IDs must be unique; endpoints must exist and be
reachable. Durations, capacities, and departure ticks are validated; self-loop
roads and repeated-node routes are rejected. An origin equal to its destination
completes immediately. All actual departures must be before
`--measurement-window` (default 120). The simulator drains remaining trips after
that window. Exceeding `--max-ticks` is an explicit failure, never a silently
truncated experiment.

## Objective and metrics

Cooperative route selection minimizes:

```text
alpha × sum(travel_time)
+ beta × sum(travel_time − route_free_flow_time)
+ gamma × sum_over_edges_and_ticks(max(0, occupancy − capacity × free_flow_time))
```

The baseline objective without the new vehicle is constant across candidate
routes. Minimizing the total therefore also minimizes the marginal objective.
Weights default to `(1, 0, 0)`; all must be finite and nonnegative, with at least
one positive. Weights trade off quantities with different meanings and must be
chosen deliberately. A longer uncongested route can reduce measured delay while
increasing trip time; inspect both metrics.

Reports include all actual vehicles, including background traffic:

- Mean travel time and nearest-rank 95th percentile, through full completion.
- Total travel time, queueing delay, and weighted objective.
- Peak edge occupancy divided by `capacity × free_flow_time`. This **load ratio
  is a congestion proxy, not measured physical density or admission utilization**.
- Peak count of severe edges and cumulative severe-edge ticks. Severe means load
  ratio at least 1.5, configurable through the Python API.
- Throughput: arrivals at or before the common measurement-window boundary,
  divided by that window. Final completion count is also reported separately.
- Per-vehicle predicted and actual times, chosen paths, route counts, and per-tick
  occupancy and queue totals in `results.json`. Predictions can change in accuracy
  when subsequently arriving controlled requests create additional traffic.

The demo is intentionally a bottleneck stress test, not evidence of real-world
effectiveness. For research, vary seeds, request ordering, capacities, demand,
forecast error, candidate count, and routing penetration; calibrate against a
microscopic simulator or measured data before making deployment claims.

## Code map

```text
traffic_routing/model.py        Validated graph and candidate path enumeration
traffic_routing/simulation.py   FIFO queues and vehicle state transitions
traffic_routing/routing.py      Five routing policies and forward rollouts
traffic_routing/experiment.py   Shared demand, batch execution, metrics
traffic_routing/report.py       Portable HTML/SVG, CSV, JSON reporting
traffic_routing/__main__.py     Command-line entry point and scenario loading
traffic_routing/map_demo.py     Cached real-road import and six-policy replay build
traffic_routing/swarm.py        Shared RL policy and coordination features
traffic_routing/train_swarm.py  REINFORCE training and held-out evaluation
map_demo/                      Offline interactive map comparison and source geometry
tests/test_routing.py           Timing, capacity, commitment, forecast, externality,
                               conservation, validation, reproducibility, CLI tests
```

Planning cost grows with requests × candidate paths × rollout length × included
vehicles. This is a correctness-focused small-network reference implementation;
large-scale deployment would require faster path search, incremental forecasting,
and a calibrated traffic model.

## Complete comparison reports

`results/report.html` compares all six policies in both planning modes (12 runs).
`results-replanning/report.html` includes all six junction-planning runs.
Real-road reports matching the map replay are in `map_demo/reports/light/`,
`map_demo/reports/surge/`, and `map_demo/reports/restriction/`. Each contains
`report.html`, `metrics.csv`, and `results.json`. The map links to its selected
scenario’s report. CSV rows identify both policy and planning mode; chart colors
identify policies, and dashed lines identify junction planning. Rebuild these
reports from cached replay data with `python3 -m traffic_routing.demo_reports`.
