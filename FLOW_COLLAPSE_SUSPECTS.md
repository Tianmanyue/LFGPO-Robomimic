# LFGPO-Flow Collapse: Ranked Suspects and Tests

Last updated: 2026-07-28

## Observed signature

- A pretrained state75 Can Flow policy starts at 61.93% success.
- Critic-only warmup retains/improves it to 65.89%.
- Success declines immediately after ratio/actor updates start.
- Smaller actor LR delays collapse, but all completed full-update runs reach 0%.
- Losses remain finite; late critic loss becomes very small as online reward and
  success approach zero.
- LFGPO-Diffusion with the shared PyTorch RatioNet is stable and reaches 99%+, so
  a generic RatioNet execution failure is unlikely.

## Ranked suspects

| Priority | Suspect | Evidence / mechanism | Decisive test |
|---:|---|---|---|
| 1 | Flow actor weighted-velocity update | Collapse starts only once actor updates; smaller actor LR slows it. PyTorch MSE constants/scheduler and high-dimensional chunk loss differ from JAX. | `lfa_uniform_actor`: ratio frozen at 1. If it collapses, actor/replay flow-matching path is sufficient. |
| 2 | Critic/GRPO advantage is directionally wrong | Sparse-reward Q can fit an all-zero solution; Q loss can shrink while policy quality worsens. Noisy sampled chunk actions may be OOD. | Compare `critic-only`, `ratio-only`, and logged `q_data/q_next/q_target/adv/group_std`. |
| 3 | Action-chunk mismatch | MuJoCo critic/ratio score one primitive action; Robomimic scores 4/8-action trajectories. Chunk Q credit assignment is a different problem. | Uniform-actor test plus future act_steps/horizon ablation using a compatible BC checkpoint. |
| 4 | Exploration noise creates OOD TD/GRPO actions | MuJoCo initial std .5 is large for normalized robot action chunks; Q targets for perturbed trajectories may be meaningless. | `lfa_noise_zero` vs `lfa_noise_fixed01` vs adaptive-noise run. |
| 5 | RatioNet port mismatch amplifies bad advantages | MuJoCo uses Mish and logit clip 7.5; shared Robomimic RatioNet uses ReLU and clip 10. Diffusion success shows this is unlikely to be the sole cause. | `lf_mjratio_w20`: restore Mish/7.5 with original hyperparameters and warmup20. |
| 6 | No explicit pretrained-policy constraint | MuJoCo trains from scratch; Robomimic must avoid destroying a good BC policy. | `lfa_bc_anchor1` and checkpoint-drift diagnostics. |
| 7 | Replay loses successful transitions | Once policy degrades, sparse successful data disappear and Q/policy enter a self-reinforcing zero-reward loop. | Larger replay / success-stratified replay / offline BC-data mixing. |
| 8 | Optimizer and LR schedule mismatch | JAX uses Adam and per-gradient-step linear decay; Robomimic uses AdamW and per-iteration cosine schedule. | Match optimizer/schedule only after actor-vs-Q causality is known. |
| 9 | Policy architecture mismatch | MuJoCo: 3x256 Mish non-residual; Robomimic: 3x1024 residual ReLU with different time embedding. | Requires a separately trained architecture-matched Flow BC policy. |

## Important non-conclusions

- A small critic TD loss does not establish critic accuracy under sparse reward.
- The archived HalfCheetah GRPO run stopped at 5k/50k steps and has no eval rows;
  it is not proof of stable completed GRPO training.
- Successful LFGPO-Diffusion strongly supports the overall LFGPO idea, but does
  not validate Flow actor gradient scaling or action-chunk Q estimation.

## Current decision order

1. Read the eight causal ablations at itr 5/10/15.
2. If uniform actor collapses, fix Flow actor loss/gradient scaling before tuning Q.
3. If uniform actor is stable but full update collapses, isolate Q advantage vs ratio.
4. Use the restored MuJoCo RatioNet + warmup20 run as a targeted confirmation.
5. Do not run Transport ReinFlow/LFGPO-Flow until Flow BC has a useful success rate.
