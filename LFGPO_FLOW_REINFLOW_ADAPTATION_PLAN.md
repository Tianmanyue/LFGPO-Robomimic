# LFGPO-Flow: MuJoCo to Robomimic adaptation diagnostics

## Scope

- Tasks: Can and Square only. Transport is intentionally excluded.
- Preserve the MuJoCo LFGPO algorithmic spine: twin-Q, GRPO/PPO advantage,
  clipped ratio objective, double-sample regularizer, batch-normalized ratio
  weights, and weighted velocity matching.
- Change only the Robomimic adapter: action sampling noise, chunk entropy/time
  scale, replay distribution, trust-region hyperparameters, and update timing.
- Seed 42, 61 iterations, critic warmup 20, evaluation every 3 iterations.

## Common checkpoints

| Task | Checkpoint | Approximate initial success |
|---|---|---:|
| Can | `pretrained/flow_bc/can_reflow_state75.pt` | 61.9% |
| Square | `pretrained/flow_bc/square_reflow_state275.pt` | 33.7% |

## Matrix (run once for each task)

| ID | Adapter change | Causal question |
|---|---|---|
| D0 | Instrumented reference | Reproduce collapse with new diagnostics |
| D1 | Fixed action noise 0.1 | Does ReinFlow-scale noise stabilize Q/actor? |
| D2 | Adaptive noise, alpha init 1 (sigma 0.1) | Can the original adaptive mechanism work from a safe scale? |
| D3 | No noise in TD target actions | Is noisy Bellman extrapolation the cause? |
| D4 | No noise in GRPO samples | Is group-relative advantage built from OOD chunks? |
| D5 | No noise in TD or GRPO | Combined sampling-path isolation |
| D6 | Entropy dimension = horizon x action_dim | Does macro-action entropy accounting stabilize alpha? |
| D7 | Original LFGPO Q(s,a)-Q(s,a') advantage | Is GRPO normalization the unstable component? |
| D8 | 25% positive replay minibatches | Is sparse-reward replay starving the ratio update? |
| D9 | Tight clip, low ratio LR, recent replay | ReinFlow-like conservative Robomimic adaptation |
| D10 | Bootstrap gamma = 0.999^4 | Does macro-action discount mismatch matter? |
| D11 | Policy update every 8 critic steps | Is actor/critic timescale the cause? |

`D10` changes the bootstrap discount only. The current MultiStep wrapper returns a
chunk reward sum rather than each primitive reward, so this is a diagnostic
semi-MDP approximation, not the final exact discounted-reward implementation.

## Added diagnostics

- Q on replay action and sampled group: `q_replay`, `q_group`,
  `q_replay_group_gap`.
- Ratio-weight distribution: `weight_mean`, `weight_std`, `weight_max`, and
  `weight_ess_frac`.
- Existing metrics retained: group standard deviation, advantage mean/std,
  ratio mean/max, alpha/noise, Q target, actor/critic/ratio losses.

## Decision rules

1. If D2/D3/D4/D5 remains stable while D0 collapses, keep the LFGPO objective
   unchanged and adopt the identified sampling adapter.
2. If D7 alone is stable, GRPO action-group construction is the primary issue;
   retain the original MuJoCo PPO-advantage branch for the rebuttal setting.
3. If D8/D9 is stable, collapse is a replay feedback problem rather than a flow
   objective problem.
4. If only D11 is stable, critic learning lags actor feedback; tune update ratio
   before learning rates.
5. Extend only configurations whose last five evaluations do not show a
   sustained decline. Stop runs after a drop greater than 20 percentage points
   across three evaluations.

## Submission

```bash
bash slurm/submit_lfgpo_flow_reinflow_adaptation.sh p32948 0
bash slurm/submit_lfgpo_flow_reinflow_adaptation.sh p32827 1
```

Each shard contains 12 jobs, balanced across Can and Square.
