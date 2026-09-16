# RL swarm: held-out evaluation

Frozen checkpoint selected using validation only. Twelve test seeds on the same corridor and scenario families.

| Policy | Mean trip across test seeds (min) | Standard deviation (min) |
| --- | ---: | ---: |
| reactive | 22.18 | 6.59 |
| anticipatory | 19.61 | 3.07 |
| cooperative | 19.36 | 2.79 |
| rl_swarm | 19.53 | 2.55 |
| rl_no_commitments | 22.47 | 6.30 |
| untrained_uniform | 21.48 | 2.26 |

This is one training run with twelve held-out demand episodes, not evidence of real-world generalization. The no-commitments result is an inference-only ablation of the same trained weights.
