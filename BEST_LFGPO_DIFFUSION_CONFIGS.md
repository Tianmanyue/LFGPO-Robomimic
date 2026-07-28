# Best LFGPO-Diffusion Configurations

Last updated: 2026-07-28. Seed 42 unless noted.

## Current winners

| Environment | Source | Actor LR | Ratio LR | Reward scale | PPO eps | Ratio cap | Ratio regularizer | Best | Final | Last-10 mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Can | Phase 8 `rs3_reg035` | 1.25e-5 | 7.5e-5 | 3 | .2 | 2 | .035 | 99.1% | 97.6% | 96.2% |
| Square | Phase 8 `rs25` | 2e-5 | 1e-4 | 2.5 | .2 | 2 | .05 | 98.5% | 98.5% | 94.4% |

## Stability alternatives

| Environment | Configuration | Best | Final | Last-10 mean | Reason to retain |
|---|---|---:|---:|---:|---|
| Can | Phase 8 `rs3_r65` | 98.2% | 97.9% | 96.6% | Better late mean than the peak winner |
| Square | Phase 8 `reg04` | 98.0% | 97.8% | 94.7% | Better late mean than the peak winner |

## Transport reference

Transport DPPO job 7955270 uses the state8000 base policy, starts at 17.0%,
and reaches 96.83% at itr 120. The job is terminated by the 24-hour wall-time
before completing 201 iterations, so 96.83% is a partial-run best rather than a
completed final score.

For the Phase 9 matched short-horizon comparison, DPPO is 77.95% at itr 50.
With eval every 10 itr, both DPPO and LFGPO-Diffusion have 45 training
iterations by that point: `45 * 400 * 50 * 8 = 7.2M` environment steps.
