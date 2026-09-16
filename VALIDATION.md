# Validation record

Environment: Python 3.14.2. Runtime requires Python 3.11+; other versions were not executed.

## Automated checks

`python3 -m unittest discover -s tests -v` - 38 tests passed.

Mid-trip planning checks cover a queue appearing after departure, loop-free
completed routes and complete traces for all six policies, unchanged static
free-flow choices, and deterministic cooperative replay. The CLI experiment
with `--replan` also completes all vehicles under the five analytical policies.
The RL junction demo reuses the departure-trained checkpoint; older held-out
RL results do not establish its performance under repeated decisions.

Coverage includes hand-calculated FIFO timing, an independent single-edge
capacity oracle, junction timing, vehicle conservation, clone isolation, future
route commitments, forecast horizons, cooperative externalities, input
validation, deterministic ordering, CLI execution, report serialization,
OpenStreetMap replay conservation, route geometry checks, and the RL swarm
policy checkpoint/schema/gradient tests.

`node --check map_demo/app.js` - passed.

## Executed synthetic benchmark

`python3 -m traffic_routing --output results`

| Policy | Average trip (ticks) | Total delay (vehicle-ticks) |
| --- | ---: | ---: |
| static | 53.67 | 7224 |
| reactive | 18.30 | 1675 |
| predictive | 18.30 | 1675 |
| anticipatory | 10.68 | 473 |
| cooperative | 9.89 | 298 |

All 150 vehicles completed under every policy. These are synthetic benchmark
results, not real-world performance claims.

## RL swarm training and evaluation

The bundled `rl_swarm` checkpoint was trained for 480 episodes on the cached
Milpitas-Sunnyvale road graph and selected by six validation seeds. Twelve
held-out demand seeds, excluded from training and checkpoint selection, produced:

| Policy | Held-out mean trip (minutes) |
| --- | ---: |
| reactive | 22.18 |
| anticipatory | 19.61 |
| cooperative | 19.36 |
| RL swarm | 19.53 |
| RL without future commitments | 22.47 |
| untrained uniform RL | 21.48 |

The RL policy improves over reactive routing and is competitive with the
model-based coordination policies in this small test. Cooperative routing remains
slightly better on the 12-seed average. This is a prototype evaluation on the
same topology and scenario families, not a claim of real-world transfer.

## Report verification

JSON round-tripping, CSV creation, and HTML escaping passed automated tests.
HTML/SVG report generation completed.

## Milpitas to Sunnyvale map demo

The offline interactive replay uses real OpenStreetMap and OSRM road geometry.
All three scenarios and all six policies were executed, and their vehicle
trajectories were exported directly from the Python simulator.

- JavaScript syntax checks passed for the replay application.
- Map replays were checked for entry capacities, queue timing, conservation,
  metrics consistency, connected sourced corridors, and shared-road
  deduplication.
- Browser verification covered play/pause, scenario changes, policy changes,
  timeline completion, road selection, linked zoom controls, and the RL learning
  panel.
- Desktop, normal app width, and narrow phone layouts were inspected. The app
  reported no JavaScript errors.
- Live traffic is not connected; all traffic and congestion are explicitly
  simulated.

Hypothetical CA 237 capacity stress: mean trip time is 31.80 minutes with
reactive routing and 22.65 minutes with RL swarm coordination in the bundled demo
seed, a 28.8% reduction. This is a selected synthetic capacity scenario, not a
measurement or forecast of real traffic.

Report coverage checks all 12 policy/mode combinations in HTML, charts, CSV,
and JSON, including RL in both modes. Real-road reports use the cached map
replays; they do not mix synthetic-network results with real-road scenarios.
