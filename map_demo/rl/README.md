# RL swarm coordination

The map demo includes a genuine trained reinforcement-learning policy. Each
controlled vehicle is a route-selecting agent. Agents share learned parameters
and coordinate through a board of current and future road commitments. This is a
small, interpretable **swarm-style coordination experiment**, not a claim to have
solved decentralized autonomous traffic control.

## Method

- **Policy:** a linear softmax over candidate routes, with five learned weights.
  This is not deep RL. Every vehicle uses the same policy parameters.
- **Observations:** excess free-flow route time, observed queue delay, peak
  remaining committed workload, time-weighted mean committed workload, and its
  peak squared. Workload is assigned vehicles divided by admission capacity;
  features are normalized by the shortest free-flow trip time.
- **Coordination:** agents announce their chosen route by adding it to the shared
  simulator state. Later agents see that commitment before those vehicles reach
  downstream roads. Road announcements disappear as vehicles pass each edge.
  This board is centrally available; communication delays and message passing are
  not modeled. Current queues and capacities are also available.
- **Actions:** choose one of up to six simple paths on the cached corridor graph.
  Each agent acts once at departure. There is no learned mid-trip rerouting.
- **Reward:** terminal negative network-wide mean travel time, including background
  vehicles, normalized by shortest free-flow time. All vehicles finish before the
  return is computed. For a fixed demand episode, minimizing mean travel time is
  equivalent to minimizing total travel time.
- **Learning:** REINFORCE with a shared terminal team return. For each sampled
  episode, sum the gradients of all agents' log action probabilities. Multiply by
  the episode advantage, average over an eight-episode batch, and apply Adam
  ascent (learning rate 0.18, gradient norm clipped to 10).
- **Baseline:** an action-independent deterministic rollout using the current
  frozen policy and the same demand. This self-critical baseline reduces variance;
  it supplies no route labels or oracle actions.
- **Initialization:** all five weights start at zero, yielding a uniform
  exploration policy. Training samples actions from the softmax distribution.
- **Execution:** frozen deterministic argmax, with stable shortest-path tie
  breaking. No online learning, exploration, or candidate traffic rollouts occur
  during RL evaluation. Logged decision scores are policy logits. The generic
  `predicted_travel_time` field is a current-queue estimate, not a learned ETA head.

The RL agent does not receive future controlled requests or the background demand
forecast. It receives current vehicle state and assigned route commitments.
Forecast-based baselines retain their original forecast access. All policies use
the same realized demand, capacities, candidate count, and simulation physics.

## Training and evaluation split

The included run uses 480 training episodes with demand seeds 10000–10479. Six
separate seeds 20000–20005 select the best checkpoint, checking every 40 episodes.
Twelve untouched test seeds 30000–30011 compare the selected frozen policy with
reactive, anticipatory, cooperative, and untrained-uniform baselines. The demo's
seed 19 is also excluded from training and checkpoint selection.

Training varies departure timing and fleet size within light, surge, and CA 237
capacity-stress families. Tests use new demand seeds on **the same road network
and scenario families**. They do not establish transfer to a new city, unforeseen
incidents, other traffic models, or real traffic. One training seed/run and twelve
test episodes are a prototype evaluation, not statistical proof of superiority.

The **no-commitments ablation** uses the same trained weights while hiding only
downstream route intentions. It retains physical vehicle positions and current
queues. It is an inference-only intervention, not a separately retrained
non-coordinating agent, so it also introduces an observation-distribution shift.

The displayed held-out mean gives each test episode equal weight; standard
deviation describes variation across the twelve episodes, not a confidence
interval. Scenario-level outcomes and all exact metrics are in `evaluation.json`.
Some scenarios may favor an established baseline. No policy is forced to win.

## Reproduce

From the project root:

```sh
python3 -m traffic_routing.train_swarm --episodes 480
python3 -m traffic_routing.map_demo
python3 -m unittest discover -s tests -v
```

Training uses Python's standard library and the cached real-road data; no API key,
GPU, learning framework, or internet connection is required. It takes several
minutes on this machine. To evaluate the frozen checkpoint without retraining:

```sh
python3 -m traffic_routing.train_swarm --evaluate-only
```

To use a checkpoint in the generic CLI:

```sh
python3 -m traffic_routing --policies reactive rl_swarm \
  --rl-model map_demo/rl/swarm-model.json --output rl-comparison
```

That CLI command uses the small synthetic network unless `--scenario` is supplied.
The checkpoint was trained on the Milpitas–Sunnyvale graph; use on another network
is an unvalidated transfer experiment. The generic CLI still defaults to the five
non-learning baselines, so it does not silently load or train a model.

Artifacts: `swarm-model.json` contains weights and training metadata;
`training-history.json` records batch returns, weights, and validation scores;
`evaluation.json` contains held-out results and the frozen-checkpoint digest.
JSON checkpoints contain data only and are schema-validated on load.

## References

- [Williams (1992): REINFORCE](https://doi.org/10.1007/BF00992696).
- [Terry et al.: Revisiting Parameter Sharing in Multi-Agent Deep Reinforcement Learning](https://arxiv.org/abs/2005.13625).

The implementation uses the policy-gradient and parameter-sharing ideas; it does
not reproduce the neural architectures or empirical results of those papers.
