# LFGPO-Flow-GRPO: MuJoCo to Robomimic Alignment Audit

Audit date: 2026-07-28

## Evidence boundary

The archived HalfCheetah run
`lfgpo_flow_grpo_2026-04-13_04-21-59_s100_test_use_atp1` used the intended
off-policy GRPO code and configuration, but stopped at 5,000 of 50,000 steps.
Its `log.csv` contains only the header and no evaluation result. Therefore it
verifies the executed configuration/update metrics, not successful final policy
performance.

## Alignment table

| Component | Archived MuJoCo run | Current Robomimic faithful path | Status / implication |
|---|---|---|---|
| Policy regime | Off-policy replay | Off-policy replay | Aligned |
| UTD | 1 update per transition | `replay_ratio=1` gives one expected replay use per collected transition | Aligned in expectation |
| Target/policy delay | 2 global updates | 2 global updates | Aligned after fix |
| GRPO group | replay Q + 16 sampled Q values | replay Q + 16 sampled Q values | Aligned after fix |
| Sampling actor | current actor | current actor | Aligned after fix |
| Exploration | `0.1 * exp(log_alpha)`, alpha init 5 | same | Aligned after fix |
| Alpha update | LR .007, delay 250, entropy scale .9 | same | Aligned after fix |
| Ratio objective | PPO clipped + double-sample regularizer | same | Formula aligned |
| Policy objective | ratio-weighted flow velocity matching | same | Formula aligned |
| Optimizers | Optax Adam | PyTorch AdamW with zero weight decay (actor/critic/ratio) | Nearly aligned, numerical details differ |
| Gamma | .99 | historical Robomimic sweep .999 | Not aligned; ablation added |
| Batch size | 256 | historical Robomimic sweep 1000 | Not aligned; ablation added |
| Reward scale | .2 | swept 1–3 | Not aligned; task reward scales differ; `.2` ablation added |
| Actor LR | 3e-4 with linear decay | 2e-6–2e-5 with cosine schedule | Not aligned; tasks/network sizes differ; exact-numeric ablation added |
| Replay capacity | 1,000,000 transitions | 5,000 vector time slots (up to 250,000 flattened transitions) | Not aligned; 1e6 ablation added |
| Initial replay | 30,000 random transitions before any update | pretrained-policy transitions; optional critic-only warmup | Structurally different |
| Initial policy | random flow policy | pretrained BC state75/state275 | Structurally different; requires conservative fine-tuning |
| Action representation | one primitive action | four-action trajectory chunk | Structurally different |
| Flow integration | 20 Euler steps | 4 Euler steps, matching the BC checkpoint | Structurally different; cannot switch without retraining/evaluating BC |
| Policy network | 3x256 MLP | 3x1024 residual MLP | Structurally different, inherited from task checkpoint |
| Critic input | state + one action | state + four-action chunk | Structurally different and harder sparse-reward credit assignment |
| Environment reward | dense MuJoCo return | sparse binary Robomimic success | Fundamental distribution difference |

## Current causal hypothesis

The failure is not a numerical crash. During critic-only warmup, the state75
policy remains strong. Once Q-derived advantages drive ratio/actor updates, the
policy degrades; successful transitions disappear from online replay; the critic
then converges to the easy all-zero solution and the policy cannot recover.
Small actor LR delays this feedback loop, which is consistent with all observed
curves.

## Causal ablations

The launcher `slurm/submit_lfgpo_flow_causal_ablations.sh` separates:

1. critic-only (actor and ratio frozen),
2. ratio-only (actor frozen),
3. uniform-weight actor (ratio frozen at one),
4. zero target/group action noise,
5. fixed action noise 0.1,
6. portable archived MuJoCo numeric settings,
7. strong pretrained-policy velocity anchor,
8. actor LR 5e-7.

All runs log Q(data), Q(next-policy), TD target, advantage mean/std, group Q
standard deviation, ratio mean/max, and effective noise standard deviation.
