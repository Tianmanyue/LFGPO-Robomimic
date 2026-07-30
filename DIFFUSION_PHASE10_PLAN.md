# LFGPO-Diffusion Phase 10

## Winner replication

- Can: exact P8 `rs3_reg035` at seeds 100 and 200. This distinguishes the exact
  99.1%-peak setting from the related `rs3` setting already summarized over seeds.
- Square: P9 `rs25_reg03` at seeds 100 and 200 to test whether the 99.0% record
  is reproducible.

## Transport causal block

All eight Transport runs use the same Phase9 Can-winner transfer parameters,
51 iterations, warmup 10, and evaluation every 5 iterations.

| Run | Question |
|---|---|
| critic-only | Does the base policy remain intact while critic/replay train? |
| ratio-only | Can RatioNet train without policy degradation when actor is frozen? |
| uniform actor | Is actor denoising/self-training sufficient to trigger collapse? |
| full reference | Reproduce the Phase9 failure at finer evaluation cadence |
| actor LR 1e-6 | Is collapse proportional to actor update magnitude? |
| policy frequency 8 | Is accumulated policy-update count the trigger? |
| BC anchor 1 | Does a fixed pretrained-denoiser trust penalty prevent drift? |
| BC anchor 5 | Does a stronger trust penalty preserve the base policy? |

Interpretation mirrors the Flow ablation. Stable critic/ratio controls and a
collapsing uniform actor identify actor drift independently of learned ratio.
Stable uniform actor but collapsing full LFGPO implicates advantage/ratio
weighting. If all frozen-policy controls are already 0%, first validate the
Transport base checkpoint/evaluator instead of interpreting online training.
