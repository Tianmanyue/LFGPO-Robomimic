# LFGPO-Flow actor localization plan

## Evidence entering this plan

- Critic-only and ratio-only preserve Can success, while actor updates with a
  frozen near-uniform RatioNet reduce success. Actor optimization is therefore
  a necessary observed trigger of the current four-action-chunk degradation.
- Horizon 1 is stable for 31 iterations, but its epoch-150 BC, evaluation sample
  count, and environment-step budget differ from the horizon-4 state75 tests.
  It is a strong lead, not yet an isolated causal result.
- Robomimic ReinFlow uses `horizon_steps=4` and `act_steps=4`; action chunking is
  viable in general. The suspect is its interaction with LFGPO's off-policy,
  chunk-level Q/ratio-weighted flow-matching update.

## Stage A: matched base policies

Train default 3x1024 residual ReLU Flow BC policies at horizons 1, 2, and 4,
save/evaluate epochs 25, 50, 75, 100, 125, and 150, and select checkpoints with
the closest initial success. Selection by success rather than epoch prevents BC
quality from being mistaken for horizon stability.

## Stage B: 3x3 causal matrix

For each matched horizon run 31 iterations with identical optimizer-step counts:

| Variant | Purpose |
|---|---|
| critic-only | Control for rollout/replay and critic learning |
| uniform actor, ratio frozen | Test actor self-distillation without GRPO weighting |
| full LFGPO | Measure the additional effect of learned ratio weighting |

Compare both iterations and primitive environment steps. Record success, actor
gradient norm, actor-to-base output distance per horizon position, per-position
flow MSE, Q group standard deviation, advantage, and ratio statistics.

## Stage C: one-factor repairs

Only after Stage B identifies the failing edge, test independently:

1. First-action-only actor loss: update the executed near-term action while
   leaving later chunk positions BC-anchored.
2. Chunk-aware TD: use discounted within-chunk reward when available and
   bootstrap with `gamma ** act_steps`.
3. Strong trust region: adaptive base-actor velocity penalty selected by measured
   actor-to-base deviation rather than a fixed weak coefficient.
4. Offline anchoring: reserve part of every actor minibatch for expert BC data,
   preventing replay from becoming pure self-training on a drifting policy.
5. Per-step ratio/advantage: only if first-action-only succeeds, replace one
   scalar chunk weight with temporal weights; this requires primitive-step
   transition/reward bookkeeping.

## Stage D: confirmation and transfer

- Extend the best Can setting to 151 iterations before declaring stability.
- Repeat on Square with a matched base policy.
- Apply the diagnosed safeguards to Transport LFGPO-Diffusion: monitor critic/Q
  group spread, actor-to-base distance, and test stronger trust/offline anchoring.
  Do not assume Flow and Diffusion share the same root cause merely because both
  can collapse.

## Literature-driven chunk hypotheses

ReinFlow explicitly supports action chunking. Its Appendix A.2 treats a chunk's
log likelihood as a **sum of per-action log likelihoods**, under conditional
independence given the initial observation. Its reported Robomimic horizons are
4 for Can/Square and 8 for Transport. Therefore action chunks are not inherently
invalid for Flow policies.

The more relevant differences from current LFGPO-Flow are:

1. ReinFlow has a factorized likelihood construction across chunk positions;
   LFGPO learns one unrestricted scalar ratio from the flattened whole chunk.
2. ReinFlow reports PPO clipping epsilon 0.001 for Robomimic visual tasks and a
   target-KL constraint. Current LFGPO uses epsilon 0.2 and no direct new/old
   policy KL constraint. A joint chunk ratio generally needs stricter control as
   chunk length grows.
3. ReinFlow's critic is observation-only (a value function). Current LFGPO must
   rank high-dimensional full chunks with Q(s, action_chunk) under sparse reward;
   measured group-Q spread is very small, so normalized advantages can amplify
   ranking noise.
4. ReinFlow collects reward at each executed action and accumulates it for the
   chunk update. Current LFGPO stores the wrapper-aggregated reward and applies a
   single-step TD bootstrap; verify discounted within-chunk reward semantics.

Add the following one-factor tests before abandoning horizon 4:

- joint-ratio clip epsilon in {0.001, 0.01, 0.05};
- factorized per-position log-ratio with summed or horizon-normalized aggregation;
- explicit ratio/KL early stopping or trust-region penalty;
- observation-value advantage versus full-chunk Q advantage;
- chunk-aware reward and `gamma ** act_steps` bootstrap.

Recent work on action-chunked Flow/VLA policies primarily reports chunk-boundary
discontinuity, multimodal switching, and asynchronous-execution mismatch, and
uses continuation conditioning or trust-region guidance. This supports tracking
boundary action jumps, but does not by itself explain the present training-time
success collapse.
