# Flow Experiments: MuJoCo vs Robomimic

Audit date: 2026-07-28

## Current Transport evidence

| Item | Checkpoint/job | Result | Interpretation |
|---|---|---:|---|
| Flow BC unified eval | Transport state150 | 0.5% | Too weak for meaningful online Flow comparison |
| ReinFlow native eval | Transport state150 | 0% through itr 12 | Already attempted; no usable baseline result |
| LFGPO-Flow diagnostic | Transport state150 | 0.5% initial, then mostly 0% | Sparse replay contains almost no successes |
| ReinFlow old run | Transport state50 | 0% | Base policy was even weaker |
| Diffusion BC | state8000 | 17.0% | Much stronger starting point |
| DPPO Transport | job 7955270 | 77.95% at itr50; 96.83% at itr120 | Partial run; stopped at 24-hour limit |

Conclusion: continue Transport Flow BC first. Re-run ReinFlow and LFGPO-Flow
from the same selected checkpoint only after unified evaluation reaches a useful
range (preferably 20–30% or higher).

## Experiment-setting differences

| Dimension | MuJoCo LFGPO-Flow-GRPO archive | Robomimic Flow | Consequence |
|---|---|---|---|
| Task | HalfCheetah dense locomotion | Can/Square/Transport sparse manipulation | Much harder critic credit assignment in Robomimic |
| Initial policy | Randomly initialized | Pretrained ReFlow BC | Robomimic must preserve existing skills |
| Replay prefill | 30,000 random transitions before updates | Online pretrained-policy trajectories; optional critic warmup | Different initial state/action/reward distribution |
| Observation | Flat state (HalfCheetah dimensions) | 23/23/59-dimensional normalized task state | Different conditioning complexity |
| Action | One primitive action | 4-step Can/Square or 8-step Transport chunk | Q and ratio operate over much higher-dimensional actions |
| Horizon/act steps | 1/1 | 4/4 or 8/8 | Robomimic is a chunk-level semi-MDP |
| Reward | Dense signed return | Sparse binary success | All-zero critic is an easy absorbing failure mode |
| Gamma | .99 | .999 in original Robomimic configs | Longer bootstrapping and greater Q error propagation |
| Batch | 256 | 1000 | Different optimization noise/generalization |
| Replay capacity | 1,000,000 transitions | 5,000 vector time slots, up to ~250k flattened transitions | Less historical successful data retention |
| UTD | 1 | Faithful sweep now 1; original port 16 | Original port over-updated critic |
| Flow integration | 20 Euler steps | 4 steps at fine-tuning/eval | Coarser policy sampling; required by current BC setup |
| Actor LR | 3e-4 linear decay in archive | 0.5e-6–20e-6 cosine in tuning | Direct numeric transfer is not scale-equivalent |
| Critic LR | 3e-4 | usually 3e-4 | Nominally aligned in faithful runs |
| Ratio LR | 3e-4 | swept 1e-5–3e-4 | Nominal default aligned |
| Reward scale | .2 | 1–3 | Not aligned; sparse reward requires separate tuning |
| Eval | 20 MuJoCo episodes | hundreds of vectorized Robomimic episodes | Different estimator variance and metric semantics |

## Network/code differences

| Component | MuJoCo/JAX | Robomimic/PyTorch | Alignment status |
|---|---|---|---|
| Framework | JAX/Haiku/Optax | PyTorch/Hydra | Mathematical port, not identical numerics |
| Flow policy input | flat obs + one action + time embedding | normalized obs + flattened action chunk + time embedding | Structurally different |
| Flow policy hidden layers | 3x256 | 3x1024 residual | Not the same network |
| Policy activation | Mish | ReLU in Robomimic ReFlow configs | Not aligned |
| Time embedding | scaled sinusoidal, dim 16 | sinusoidal MLP, dim 32 | Not aligned |
| Policy output | one-action velocity | 4/8-action trajectory velocity | Not aligned |
| Flow loss | Optax squared error, ratio weighted | PyTorch MSE, ratio weighted | Same objective; reduction/constant and numerics differ |
| Twin critic | 3x256 Mish, flat state+single action | 3x256 Mish, state+full action chunk | Architecture width aligned; input problem differs |
| Ratio network | 3x256 Mish, logit clip 7.5 | 3x256 ReLU, logit clip 10 | Important port mismatch still present |
| Ratio head | zero initialized, output exp(logit) | same | Aligned |
| Target actions | current flow actor + adaptive Gaussian noise | faithful path now same | Aligned after fixes |
| Noise | `.1 * exp(log_alpha)`, alpha init 5 | faithful path now same | Aligned after fixes |
| GRPO group | replay action + 16 sampled actions | faithful path now same | Aligned after fixes |
| Policy/target delay | every 2 global updates | every 2 global updates | Aligned after fixes |
| Target Polyak tau | .005 | .005 | Aligned |
| Optimizer | Adam | AdamW with zero configured weight decay | Nearly aligned, implementation details differ |
| LR schedule | per-gradient-step linear decay | per-iteration cosine warmup/restart | Not aligned |

## ReinFlow differences

| Component | MuJoCo | Robomimic |
|---|---|---|
| Policy output | one action | action chunk |
| Policy/network family | JAX FlowPPO variants, task-resolved settings | PyTorch 3x1024 residual ReFlow policy |
| Fine-tuning | on-policy Flow PPO/SDE likelihood | on-policy PPOFlow with learned exploration-noise network |
| Critic | state value on flat observation | 3x256 Mish state value on normalized task observation |
| Starting point | task-dependent policy/checkpoint | required ReFlow BC checkpoint |

## Highest-priority code questions

1. Ratio activation and logit range are not faithfully ported (Mish/7.5 vs ReLU/10).
2. Actor loss has the same conceptual objective but JAX/PyTorch loss constants,
   optimizer scheduling, and trajectory dimensionality differ.
3. A Q over an entire 4/8-action chunk is not equivalent to MuJoCo's one-action Q.
4. Sparse online replay loses successful examples, creating a critic/actor collapse
   feedback loop absent from dense-reward MuJoCo.
5. Transport Flow BC is currently the dominant blocker independent of online RL.
