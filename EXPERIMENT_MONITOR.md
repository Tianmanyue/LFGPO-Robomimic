# LFGPO Robomimic Experiment Monitor

Last updated: 2026-07-26 07:19 CDT

## Plan

- Gate 1: environment and native-library validation.
- Gate 2: one `can` LFGPO-Flow smoke job (`n_envs=10`, `n_train_itr=4`, `val_freq=2`).
- Gate 3: submit the 12 seed-42 full runs only after the smoke completes at least one training iteration and exits successfully.
- Monitor Slurm state, stderr/stdout, local result files, evaluation success rate, losses, and iteration wall-clock.

## Current status

- The original 12-job attempts were stopped before meaningful training.
- No flow pretraining is needed. Existing checkpoints are used from `pretrained/flow_bc/`.
- Environment import/build gate passes.
- Single smoke job `7952639` completed successfully in 3m09s.
- Flow checkpoint loading, EGL environment creation, evaluation, checkpoint saving, and actor/critic/ratio updates all passed.

## Validated environment

- Conda env: `lfgpo_robomimic` (Python 3.8)
- PyTorch: 2.4.1+cu121
- MuJoCo: 3.1.6
- Robosuite: 1.4.1
- Robomimic: 0.3.0
- Cython: 0.29.37
- Native dependencies: GLEW, Mesa, patchelf
- `mujoco_py` imports successfully.
- `pip check`: no broken requirements.

## Checkpoints

| Task | Flow checkpoint | Validation |
|---|---|---|
| can | `pretrained/flow_bc/can_reflow_state50.pt` | PyTorch load OK; `model` state present |
| square | `pretrained/flow_bc/square_reflow_state50.pt` | PyTorch load OK; `model` state present |
| transport | `pretrained/flow_bc/transport_reflow_state50.pt` | PyTorch load OK; `model` state present |

All three checkpoints expose a 32-dimensional time embedding. Finetune configs match with `time_dim=32`, `mlp_dims=[1024,1024,1024]`, ReLU, and residual connections.

## Incident log

### 2026-07-26 — incorrect assumption about flow pretraining

- Symptom: initially prepared three unnecessary flow BC pretrain jobs.
- Cause: followed the internal guide's older Step A without first checking the repository's included `pretrained/flow_bc/*.pt` files.
- Resolution: stopped submission before any pretrain job entered Slurm; validated all three existing checkpoints.

### 2026-07-26 — unconditional D4RL import

- Symptom: early jobs exited with `ModuleNotFoundError: No module named 'd4rl'` from `script/run.py`.
- Cause: the general launcher imported D4RL unconditionally although Robomimic does not use it.
- Resolution: made D4RL registration opt-in through `REINFLOW_IMPORT_D4RL=1`.

### 2026-07-26 — D4RL dependency changed MuJoCo

- Symptom: installing D4RL pulled MuJoCo 3.2.3, which the experiment guide explicitly forbids.
- Resolution: restored `mujoco==3.1.6`, `dm_control==1.0.16`, and `Cython<3`; verified versions.

### 2026-07-26 — missing native build dependencies

- Symptom: `mujoco_py` failed first on `GL/glew.h`, then on missing `patchelf`.
- Resolution: installed Conda GLEW/Mesa/patchelf, exported Conda include/library paths in `set_env.sh`, and successfully imported `mujoco_py`.

### 2026-07-26 — concurrent first-run download risk

- Risk: the two diffusion methods for a task can start together and write the same normalization/checkpoint path.
- Resolution: added per-target `fcntl` locks and a second existence check around first-run downloads in `script/run.py`.

## Job tracking

| Stage | Job ID | Name | State | Notes |
|---|---:|---|---|---|
| Smoke | 7952639 | `smoke_lf_can_s42` | COMPLETED | Exit 0; elapsed 00:03:09 |
| Full | 7952657 | `ld_can_s42` | COMPLETED | Exit 0; elapsed 04:08:08 |
| Full | 7952658 | `dp_can_s42` | COMPLETED | Exit 0; elapsed 06:14:25 |
| Full | 7952659 | `ld_square_s42` | RUNNING | BC loaded; 50 envs created |
| Full | 7952660 | `dp_square_s42` | RUNNING | BC loaded; rollout started |
| Full | 7952661 | `ld_transport_s42` | RUNNING | BC loaded; environment initialization |
| Full | 7952662 | `dp_transport_s42` | RUNNING | No Slurm error |
| Full | 7952663 | `lf_can_s42` | COMPLETED | Exit 0; elapsed 03:38:08 |
| Full | 7952664 | `rf_can_s42` | COMPLETED | Exit 0; elapsed 03:46:55 |
| Full | 7952665 | `lf_square_s42` | RUNNING | No Slurm error |
| Full | 7952666 | `rf_square_s42` | RUNNING | No Slurm error |
| Full | 7952667 | `lf_transport_s42` | RUNNING | No Slurm error |
| Full | 7952668 | `rf_transport_s42` | RUNNING | No Slurm error |

## Result tracking

| Job | Iteration | Env steps | Success rate | Actor loss | Critic loss | Ratio loss | Wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke eval 0 | 0 | 0 | 0.0000 | — | — | — | — |
| smoke train 1 | 1 | 12000 | — | 0.1385 | 0.0004 | -0.1541 | 27.7041 s |
| smoke eval 2 | 2 | — | 0.0000 | — | — | — | — |
| smoke train 3 | 3 | 24000 | — | 0.1467 | 0.0037 | -0.0844 | 27.4210 s |
| 7952657 LFGPO-Diffusion Can eval 0 | 0 | 0 | 0.0000 | — | — | — | — |
| 7952658 DPPO Can eval 0 | 0 | 0 | 0.0000 | — | — | — | — |
| 7952659 LFGPO-Diffusion Square eval 0 | 0 | 0 | 0.0000 | — | — | — | — |
| 7952660 DPPO Square eval 0 | 0 | 0 | 0.0000 | — | — | — | — |
| 7952657 LFGPO-Diffusion Can | 1 | 60000 | — | 0.1704 | 0.0146 | -0.0465 | 100.6292 s |
| 7952657 LFGPO-Diffusion Can | 2 | 120000 | — | 0.2757 | 0.0150 | -0.0353 | 99.3091 s |
| 7952659 LFGPO-Diffusion Square | 1 | 80000 | — | 3.5867 | 0.0075 | -0.0391 | 139.9216 s |

## Monitoring notes

### 2026-07-26 07:15 CDT

- Active: jobs 7952657–7952661 except 7952662; five running, seven pending.
- Can and Square diffusion jobs loaded their DPPO `state_8000.pt` checkpoints and created the configured 50 environments.
- Transport LFGPO-Diffusion loaded its `state_8000.pt` checkpoint.
- DPPO Can/Square rollout counters are advancing.
- No traceback, CUDA OOM, checkpoint mismatch, missing file, or Hydra execution error in active-job stderr.
- No full-run evaluation point has been emitted yet; initialization/first rollout remains in progress.

### 2026-07-26 07:16 CDT

- Can LFGPO-Diffusion and DPPO both completed eval 0 with identical initialization metrics: success rate 0.0000, average episode reward 0.9139, average best reward 0.2285.
- The identical eval-0 values are expected because both methods share the same pretrained diffusion checkpoint and seed.
- Five jobs remain running and seven pending; no active-job errors detected.
- Square LFGPO-Diffusion and DPPO also completed identical eval 0: success rate 0.0000, average episode reward 0.4120, average best reward 0.1030.

### 2026-07-26 07:19 CDT

- Five jobs running, seven pending; no Slurm failures or stderr exceptions.
- LFGPO-Diffusion Can completed iterations 1–2 at 60k/120k steps. Actor loss 0.1704→0.2757, critic loss 0.0146→0.0150, ratio loss -0.0465→-0.0353; values are finite.
- LFGPO-Diffusion Square completed iteration 1 at 80k steps. Actor/critic/ratio losses are 3.5867/0.0075/-0.0391; values are finite, with actor loss higher than Can but not sufficient evidence of divergence.
- DPPO jobs have not emitted their first training summary yet; processes remain active.

### 2026-07-26 — success-rate metric defect

- Status at inspection: 4/12 full jobs completed successfully and 8/12 remain running.
- Every emitted `success rate` is 0, but this value is invalid: Robomimic terminates an episode immediately on success and `MultiStep` returns the sum of rewards in the action chunk, while the agent computes `episode_best_reward = max(chunk_reward) / act_steps` and compares it with threshold 1.
- Consequently, a successful Can/Square episode is capped at a reported best reward of 0.25 (`act_steps=4`), and Transport at 0.125 (`act_steps=8`), so the configured threshold can never be reached.
- Sparse Robomimic episode reward is currently the useful proxy for success probability. Completed Can final evaluations: LFGPO-Diffusion 0.95, DPPO 1.00, LFGPO-Flow 0.00, ReinFlow 1.00.
- This is a reporting/statistics defect, but LFGPO-Flow Can also has a substantive zero-reward result that needs separate investigation.
- Fixed the active code paths for LFGPO, DPPO, and ReinFlow by removing the erroneous second division by `act_steps`. Already-running jobs retain the old in-memory code and therefore still print zero.
- Added `script/export_robomimic_csv.py` and generated `experiment_metrics.csv` (1,190 rows at creation). `effective_success_rate` reconstructs historical sparse-task success from mean episode reward; `success_rate` preserves the original logged value.

### 2026-07-26 — LFGPO-Flow policy collapse

- Can LFGPO-Flow starts from the expected BC performance (eval-0 episode reward 0.2338), briefly reaches 0.2816, and then declines to zero after actor updates begin.
- With 50 environments, the configuration performs about 240 replay minibatch updates per iteration. Flow used actor LR `1e-4`, ten times the successful LFGPO-Diffusion LR (`1e-5`).
- Changed all three LFGPO-Flow configs to actor LR `1e-5` and scheduler minimum `3e-6`.
- Submitted Can validation job 7955223 for 18 iterations (evaluations every 4 iterations); pending because of `QOSMaxGRESPerUser` while the eight formal jobs occupy the allocation.

### 2026-07-26 — corrected 12-job rerun

- At user direction, cancelled the eight still-running old-code jobs and pending smoke 7955223.
- Increased LFGPO-Flow `num_grpo_samples` from 16 to 32 for lower-variance GRPO group advantages; retained actor LR `1e-5` to address the observed collapse.
- Submitted all 12 experiments with corrected success-rate reporting:
  - Can: LFGPO-Flow 7955260, LFGPO-Diffusion 7955261, ReinFlow 7955262, DPPO 7955263.
  - Square: LFGPO-Diffusion 7955264, LFGPO-Flow 7955265, DPPO 7955266, ReinFlow 7955267.
  - Transport: LFGPO-Diffusion 7955268, LFGPO-Flow 7955269, DPPO 7955270, ReinFlow 7955271.
- All 12 entered Slurm in PENDING/Priority state. CSV will be regenerated during monitoring, with special attention to LFGPO-Flow eval-0, post-warmup, peak, and final success rates.

### 2026-07-26 — weaker Can diffusion base policy

- Corrected eval-0 showed both Can diffusion methods used `state_8000.pt` and started at 138/151 = 91.39% success, leaving little improvement headroom.
- Cancelled jobs 7955261 (LFGPO-Diffusion) and 7955263 (DPPO).
- Changed both Can configs to the repository's algorithm-comparison checkpoint `state_5000.pt`.
- Resubmitted LFGPO-Diffusion as 7955329 and DPPO as 7955328. Their eval-0 will be checked before allowing the full runs to continue.

### 2026-07-26 — LFGPO-Flow Can ratio-LR tuning

- Cancelled formal LFGPO-Flow jobs 7955260, 7955265, and 7955269 after Can again collapsed from 23.38% to 2.50%; retained all partial logs/checkpoints.
- Added `slurm/submit_lfgpo_flow_can_tune.sh`, parameterized by ratio LR, actor LR, GRPO group size, and run tag.
- First isolated trial: ratio LR `1e-4` (down from `3e-4`), actor LR `1e-5`, GRPO samples 32, 30 iterations, evaluation/checkpoint every 3 iterations.
- Submitted as job 7956005 (`lf_can_ratio1e4`); currently pending on priority.
- Added an `advantage_mode` switch to `LFGPOFlow`. `grpo` retains the 32-action group-relative estimate; `ppo` matches LFGPO-Diffusion by estimating `V(s)` from one independent target-policy action and minibatch-normalizing `Q(s,a)-V(s)`.
- Submitted the matched PPO trial as 7956158 (`lf_can_ratio1e4_ppo`), also 30 iterations/eval every 3 with ratio LR `1e-4` and actor LR `1e-5`. Both trials are pending; all settings except advantage estimator are matched.

### 2026-07-26 — stronger Square/Transport Flow BC

- Cancelled ReinFlow Square 7955267 and Transport 7955271 after their state-50 bases evaluated at 0.5% and 0%; corresponding LFGPO-Flow jobs were already cancelled.
- Verified both state-50 files contain model, EMA, optimizer, and scheduler state and can be resumed safely.
- Added `slurm/run_flow_pretrain_long.sbatch`: resume from state 50 for 150 additional epochs, save every 25 epochs, eight-hour limit, and do not replace production symlinks automatically.
- Submitted Square as 7956238 and Transport as 7956239. Both target state 200 and are pending on priority. Intermediate state 75/100/125/150/175/200 checkpoints will be evaluated before selecting a replacement.
- Extended the same continuation plan to Can because its state-50 LFGPO evaluation is only 23.38%, versus 69.64% for the selected diffusion base. Submitted Can Flow BC continuation as 7956252; selection target is a stronger but non-saturated base, not automatically state 200.

### 2026-07-26 — Flow diagnostics and Can base-policy selection

- Square continued Flow BC evaluations: state-125 = 10.50%, state-150 = 11.94%; state-150 is the better current candidate.
- Transport continued Flow BC state-150 evaluates at 0.50%, so both ReinFlow and LFGPO short diagnostics are required to distinguish weak BC from algorithm tuning issues.
- Corrected Square/Transport LFGPO-Flow diagnostic submissions after Hydra rejected a redundant `model.advantage_mode` override absent from those structured configs. Replacement jobs: Square 7959827, Transport 7959828. ReinFlow diagnostic jobs: Square 7959830, Transport 7959831.
- Can continued Flow BC state-150 started at 88.67% and reached 90.55% at itr 3 in job 7959761. This base is too strong for the intended improvement study, so the job was cancelled with logs preserved.
- Submitted pure evals for intermediate Can checkpoints to select a roughly 50% base: state-75 job 7959918, state-100 job 7959920, state-125 job 7959922. The selected checkpoint will be the one closest to 50%, not automatically the strongest.

### 2026-07-26 20:15 CDT — current formal and diagnostic status

- Can LFGPO-Diffusion 7955329 completed successfully: peak eval 95.94%, final eval 94.23%. Can DPPO 7955328 remains running and most recently reached 99.84%.
- Square LFGPO-Diffusion 7955264 showed a strong late improvement: 74.91% at eval 130 and 81.16% at eval 140. Square DPPO 7955266 most recently reached 96.55%.
- Transport LFGPO-Diffusion 7955268 remains collapsed at 0% through the latest evaluations/training itr 28. Transport DPPO 7955270 improved from 17.00% to 32.00% and then 48.86%.
- Square LFGPO-Flow diagnostic 7959827 is healthy but initially declined from 11.94% to 8.50% (itr 3) and 9.50% (itr 6); continue through 30 itrs before deciding.
- Transport LFGPO-Flow diagnostic 7959828 loaded correctly and eval-0 is 0.50%; it is still in its long first training iteration.
- Square ReinFlow diagnostic 7959830 initialized successfully and loaded the same state-150 BC; no evaluation result yet. Transport ReinFlow 7959831 is pending.
- Can BC candidate evals state-75 7959918, state-100 7959920, state-125 7959922 are pending. No new traceback, NaN, CUDA OOM, or checkpoint mismatch was found in active jobs.
- Regenerated plotting CSV after this inspection.

### 2026-07-26 22:00 CDT — late-run results

- Can diffusion curves are effectively complete: LFGPO 7955329 starts at 69.64%, peaks at 96.75%, and finishes at 94.23%; DPPO 7955328 peaks at 99.85% and most recently reports 99.71%.
- Square LFGPO-Diffusion 7955264 continues its late surge: 41.20% base, 88.96% peak, 88.38% latest near itr 180. Square DPPO 7955266 most recently/at peak reports 99.23%.
- Transport DPPO 7955270 improved 17.00% -> 32.00% -> 48.86% -> 67.50%. Transport LFGPO 7955268 remains at 0% after its 17.00% base and is the clear diffusion tuning failure.
- Square LFGPO-Flow diagnostic 7959827 completed 30 itrs: base 11.94%, peak 15.27%, latest eval 12.94%. It can improve the weak base slightly but is noisy and needs conservative tuning.
- Square ReinFlow diagnostic 7959830 starts at 3.50% under its evaluation path, briefly revisits 3.50%, and collapses to 0% from itr 15 onward.
- Transport LFGPO-Flow 7959828 stays in {0%, 0.5%} through eval itr 9; Transport ReinFlow 7959831 is also 0% through eval itr 3. The 0.5% BC is too weak for either algorithm under current settings.
- Can intermediate BC eval jobs 7959918/7959920/7959922 still have no stdout result; do not select the approximately-50% base until these results exist.
- Refreshed plotting CSV after extracting full curves.

### 2026-07-27 — Can state-75 formal Flow runs and further BC training

- Intermediate Can Flow BC evals completed: state-75 = 61.93%, state-100 = 69.44%, state-125 = 93.45%. Selected state-75 as the non-saturated formal starting point.
- Submitted matched Can state-75 formal runs on p32948: LFGPO-GRPO-Flow 151 itrs job 7966887 (ratio LR 1e-4, actor LR 1e-5, G=32, eval/save every 5), ReinFlow 151 itrs job 7966890 (official training parameters, eval/save every 5).
- Square state-150 BC achieved 11.94% while Transport state-150 remained at 0.50%. Because BC continuation is inexpensive, submitted another 150 epochs from each state-150 checkpoint: Square job 7966913, Transport job 7966916. Evaluate intermediate/final checkpoints before replacing any production base path.

### 2026-07-27 — rebuttal scope reduced to Can and Square

- At user direction, stopped all remaining Transport work except the DPPO baseline 7955270: LFGPO-Diffusion 7955268, LFGPO-Flow diagnostic 7959828, ReinFlow diagnostic 7959831, and BC continuation 7966916. Running jobs entered COMPLETING; partial logs/checkpoints are preserved.
- Current Flow target is exactly four formal runs: Can/Square x LFGPO-Flow/ReinFlow. Can jobs 7966887/7966890 are pending. Square BC continuation 7966913 is pending; evaluate its checkpoints before submitting the two Square formal Flow jobs.
- Removed Transport candidates from both external-account Diffusion tuning launchers so future permission activation cannot accidentally consume resources on Transport.

### 2026-07-27 14:03 CDT — LFGPO-Flow collapse diagnosis

- Current queue: Transport DPPO 7955270 and Square Flow jobs 7973360/7973364/7973365 are running. No traceback, NaN, Inf, CUDA OOM, or Slurm failure explains the Flow degradation.
- Can Flow Phase 3 job 7973316 starts at 61.93%, peaks at 67.13%, falls to 16.50% by the sixth post-start evaluation, and reaches 0% by the ninth. Square Phase 3 job 7973317 similarly starts at 33.65%, peaks at 37.14%, and decays to approximately 0%.
- All three current Square LFGPO-Flow variants repeat the same pattern. Jobs 7973360/7973364/7973365 share a 33.65% start and 37.14% early peak, then decline to latest success rates of 1.0%, 1.5%, and 0.5%, respectively. This rules out a Can-only checkpoint defect and makes hyperparameter-only ratio-LR failure unlikely.
- Root-cause finding: the shared off-policy loop computes `300 * 50 / 1000 * replay_ratio(16) = 240` actor/ratio optimizer opportunities per training iteration. ReinFlow uses approximately `max(1, 15000 / 10000) * 5 = 5` actor optimizer steps per iteration. LFGPO-Flow therefore receives roughly 48x as many actor updates, or about 36,000 over 151 iterations, without a fixed-policy anchor.
- Added a configurable `train.policy_update_freq` (default 1, so prior experiments remain reproducible). Phase 5 uses 48: critic updates remain at 240/iteration while actor and ratio update approximately 5 times/iteration. Phase 5 also retains the frozen-BC anchor and conservative actor/ratio settings.
- Diffusion is healthy under the existing loop: Phase 3 Can reached 97.35%; Square reached about 91.9%. The update-frequency correction is enabled only by the Flow Phase 5 launcher.

### 2026-07-27 — corrected Flow Phases 1–5 submitted across two accounts

- Added and pushed `slurm/submit_lfgpo_flow_phases1_5.sh` at commit `93ec3b0`. It contains 12 non-duplicate Can/Square Flow configurations spanning Phases 1–5. Every run uses `policy_update_freq=48`, the selected state75/state275 bases, eval/save every 5 iterations, and a frozen-BC anchor.
- Submitted Can-heavy shard 0 to p32948: 8006533, 8006534, 8006536, 8006541, 8006543, 8006544.
- Submitted Square-heavy shard 1 to newly enabled p32827: 8006559, 8006562, 8006564, 8006565, 8006566, 8006568. All 12 were accepted by Slurm and initially entered PENDING.
- The three old unthrottled Square LFGPO-Flow jobs 7973360/7973364/7973365 remain running but are already collapsed to approximately 0–1.5%. They were not cancelled without explicit user direction.
- Added `slurm/submit_lfgpo_diffusion_phase6.sh` for account 2 only. It refines the Phase-2 winners: Can c3_cap2 (97.3%) and Square s3_a2e5_cap2 (98.0%); no local Diffusion Phase-6 jobs were submitted.
- Environment-step convention: the training loop increments by `n_envs * act_steps` for every environment vector step and does not increment during eval. A Can training iteration collects `300*50*4 = 60,000` primitive environment steps; Square collects `400*50*4 = 80,000`. Because every fifth iteration is eval, plots should use the logged `step` field rather than naively multiplying itr by a constant.

### 2026-07-27 — ReinFlow final targets and eight-candidate Diffusion Phase 6

- Full Can ReinFlow state75 job 7966867: evaluation starts at 79.04% under ReinFlow's stochastic evaluation path, reaches 95.83% at itr 30, 99.80% at itr 80, peaks at 100.00%, and finishes at 99.82%. Earlier references to approximately 95% were intermediate results, not the final baseline.
- Full Square ReinFlow state275 job 7973359: starts at 32.21%, rises late, and peaks/finishes at 96.55% at itr 200.
- Consequently, the Flow target is to exceed 96.55% on Square and approach/match 99–100% on Can; exceeding Can's measured 100% peak is impossible.
- Expanded the account-2-only Diffusion Phase 6 launcher from four to eight jobs: four local refinements around Can c3_cap2 (97.3%) and four around Square s3_a2e5_cap2 (98.0%). Added actor-LR, ratio-LR, ratio-cap, regularization, and clip local variations without repeating the unstable Square Phase-4 combination.
- Pushed the eight-job launcher to `quest-two-account-launchers` at commit `964c165`; no Phase-6 Diffusion jobs were submitted locally.

### 2026-07-27 — Can corrected Flow sweep changed from state75 to state50

- At user direction, changed the corrected Can LFGPO-Flow sweep base from state75 (61.93% under LFGPO evaluation) back to state50 (23.38%). Square remains on state275 (33.65%).
- Verified the official Can state50 ReinFlow job 7955262 already completed all 151 iterations: it starts at 49.26% under ReinFlow's native inference, reaches 100%, and finishes at 100%. It does not need to be rerun.
- Cancelled the six still-pending state75 Can sweep jobs: 8006533, 8006534, 8006536, 8006541, 8006543, 8006566. No running work or results were lost.
- Resubmitted the same six corrected/limited-frequency configurations using state50. p32948: 8012029, 8012031, 8012032, 8012035, 8012041. p32827: 8012077. All were accepted and initially entered PENDING.
- The six Square state275 jobs remain unchanged: 8006544, 8006559, 8006562, 8006564, 8006565, 8006568.
- Updated the launcher with an optional `all|can|square` target filter and made state50 the default Can checkpoint; pushed at commit `ee02abd`.

### 2026-07-27 — corrected Flow queue check after state50 replacement

- No corrected Flow job has started yet: 0 RUNNING, 12 PENDING, all with reason Priority. Therefore no state50/state275 Phase 1–5 eval or loss is available yet and there is no new runtime error.
- Slurm's current estimated starts for p32948 are 2026-07-27 22:10–22:52 CDT (8006544, 8012029, 8012031, 8012032, 8012035, 8012041).
- Slurm's current estimated starts for p32827 are 2026-07-28 22:40 through 2026-07-29 00:10 CDT (8006559, 8006562, 8006564, 8006565, 8006568, 8012077). Estimates are scheduler projections and may move earlier.
- Old unthrottled Square LFGPO-Flow jobs 7973360/7973364/7973365 completed successfully at the Slurm level but their policies remained collapsed; these are negative diagnostic results, not usable final models.
- Transport DPPO 7955270 ended in FAILED state after 23:57:47 with exit code `0:15`, consistent with the 24-hour wall-time signal rather than an application traceback. Its partial curve/results remain available; Transport is outside the current Can/Square tuning focus.

### 2026-07-27 — first corrected Flow jobs running

- Two corrected limited-frequency jobs started on p32948; the remaining ten are PENDING/Priority.
- Square Phase-5 safe job 8006544 correctly loaded state275. Eval success progresses 33.65% (itr 0) -> 34.62% (itr 5) -> 35.92% (itr 10). Training reward remains approximately 0.26–0.40 through itr 11; critic/actor/ratio losses are finite. This is the first long Square LFGPO-Flow run that has not immediately decayed after policy updates.
- Can state50 Phase-1 job 8012029 correctly loaded state50. Eval success progresses 23.38% (itr 0) -> 26.73% (itr 5); training reward is approximately 0.23–0.32 through itr 4. Losses are finite.
- No traceback, NaN, CUDA OOM, or checkpoint mismatch. The stderr Gym/Gymnasium text is a non-fatal deprecation warning.
- These early curves support the update-frequency diagnosis, but both jobs remain too early to conclude final stability; monitor especially after itr 30–50, where old runs began sustained decline.

### 2026-07-27 — project storage exhaustion, cleanup, and Flow resubmission

- All six p32948 corrected Flow jobs that started (8006544, 8012029, 8012031, 8012032, 8012035, 8012041) exited with code 1 because PyTorch checkpoint writes failed (`PytorchStreamWriter failed writing file`, truncated archive position mismatch). This was a storage failure, not a numerical/policy failure.
- Before the write failure, useful early evidence was preserved: Square Phase-5 safe progressed 33.65% -> 34.62% -> 35.92% -> 34.45% through itr 15; Can Phase-1 progressed 23.38% -> 26.73% -> 19.50% -> 31.50% through itr 15. Neither had entered the old sustained-zero collapse regime.
- `/gpfs/projects/p32948` reported 100% filesystem use with only 8 MiB visible free. Repository runtime logs held about 15.33 GiB of checkpoint files.
- With user authorization, deleted checkpoint files only from confirmed collapsed Flow runs and failed partial runs; deleted ReinFlow intermediate `state_*.pt` while retaining `best.pt`/`last.pt`; deleted BC continuation checkpoints whose selected state75/state275 models already exist under `pretrained/flow_bc/`. All logs, configs, CSVs, monitor records, and formal pretrained copies were retained. Deleted files are not directly recoverable.
- `runtime/log` decreased from about 16 GiB to 4.9 GiB, and a direct 64 MiB write/delete probe succeeded. Shared-GPFS `df` still shows 100% because it reflects the entire mounted filesystem, but project writes work again.
- Reduced corrected Flow checkpoint frequency from every 5 to every 10 iterations while retaining eval every 5; pushed at commit `f0d7825` to prevent another ~13 GiB accumulation.
- Cancelled the six held p32827 old-save-frequency jobs and resubmitted all 12 corrected jobs. p32948: 8030939–8030944. p32827: 8030945–8030950. All were accepted and initially entered PENDING.
- Added an eight-job account-2 Can Diffusion Phase-7 refinement around the 98.5% winner (actor 1.25e-5, ratio 7.5e-5): ratio/actor brackets plus cap, reg, clip, and reward-scale one-factor variants. Pushed at commit `764e4fb`; not submitted locally.

### 2026-07-27 — MagicSim archival audit and storage recovery

- Before deleting the approximately 213.6 GiB MagicSim directory, audited Git branch `yu_training`. HEAD initially matched its upstream but six tracked pi0.5 training/deployment files had uncommitted changes, and untracked code/config included a training launcher, README/environment files, and three normalization-stat assets.
- Committed all non-model code/config/assets, fetched and rebased over concurrent remote updates without conflict, and pushed commit `8ce8b523` (`Save pi0.5 training and deployment updates`) to `git@github.com:luhr2003/MagicSim.git`, branch `yu_training`.
- Verified local HEAD exactly matched `origin/yu_training` (`0 0` divergence), tracked worktree was clean, and untracked source/config count was zero.
- Permanently deleted `/gpfs/projects/p32948/MagicSim` with user authorization. The directory is not locally recoverable; code is recoverable from GitHub, while local models/data were intentionally not archived.
- GPFS changed from 100% used/8 MiB available to 79% used/216 GiB available. Four corrected Can Flow jobs (8030939–8030942) are now running; the other eight remain Pending/Priority.

### 2026-07-27 — corrected-frequency Can state50 Flow results

- Three corrected Can state50 jobs completed cleanly at the Slurm/application level, and Phase4 remains running. No NaN, OOM, traceback, checkpoint error, or storage error occurred after cleanup.
- Phase1 8030939 (`actor=1e-5`, `ratio=1e-4`) starts at 23.38%, peaks at 31.50% (itr 15), then declines steadily and finishes at 0%.
- Phase2 8030940 (`actor=1e-5`, `ratio=3e-4`) starts at 23.38%, peaks at 31.00% (itr 15), and finishes at 0% (one late 0.5% fluctuation).
- Phase3 8030941 (`actor=5e-6`, `ratio=5e-5`, scale 2) starts at 23.38%, peaks at 26.73% (itr 5), and finishes at 0%.
- Phase4 8030942 (`actor=1e-5`, `ratio=2e-5`, clip .15, scale 3) starts at 23.38%, peaks at 29.50% (itr 15), reaches 0% by approximately itr 85, and remains near zero through the latest eval.
- `policy_update_freq=48` delayed the collapse and reduced optimizer work, but did not solve the underlying off-policy weighted-flow policy drift. BC anchor 0.1 is insufficient in Phases1–4.
- Can Phase5 safe 8030943 (actor 2e-6, ratio 1e-5, BC anchor 0.3) is still PENDING and is the only remaining Can configuration that tests substantially stronger drift protection. All six Square state275 corrected jobs are also still PENDING; no new Square result exists.

### 2026-07-28 — faithful MuJoCo off-policy port diagnosis and state75 tests

- Confirmed Robomimic LFGPO-Flow was not on-policy: it trains from a replay deque and randomly sampled replay minibatches. The collapse instead correlates with mismatches from the successful MuJoCo off-policy implementation.
- The most consequential mismatch was effective critic UTD: Robomimic used `replay_ratio=16` (240 critic minibatches per iteration), while the MuJoCo LFGPO-Flow-GRPO configuration uses `utd=1`. Earlier throttling affected actor/ratio updates but left this critic over-updating unchanged.
- Restored the MuJoCo data path: UTD 1, policy/target delay 2, current-policy noisy actions for TD/GRPO sampling, and GRPO normalization over the replay action plus the 16 sampled actions. Added population-standard-deviation behavior and finite TD-target protection. A real-checkpoint CPU smoke test completed with finite critic, ratio, actor, and advantage values.
- Pushed the implementation and sharded launcher to `quest-two-account-launchers` through commit `8c38551` (core implementation commit `528f0fd`).
- Replaced pending old state50 Phase5-balanced job 8030949. Phase5-safe 8030943 had already completed, so it was retained as a finished comparison.
- Old state50 Phase5-safe 8030943 starts at 23.38%, peaks at 28.36% (itr 10), and finishes at 7.5%; stronger BC anchoring slowed degradation but still did not solve it.
- Submitted two Can state75 faithful-port diagnostics: 8072662 (`p32948`, action-noise std 0.1) and 8072663 (`p32827`, action-noise std 0.3). Both use 151 iterations and eval every 5 iterations.

### 2026-07-28 — complete faithful Flow sweep and Diffusion Phase 8

- A second line-by-line audit restored two remaining MuJoCo details: adaptive action noise (`noise_scale * exp(log_alpha)`, alpha initialized to 5 and entropy-target optimized every 250 global gradient steps) and global rather than per-iteration delayed-update counters. Actor/target delays are 2, UTD is 1, policy warmup is disabled, current actor supplies sampled actions, and GRPO uses replay plus 16 sampled actions.
- Cancelled 8072662/8072663 because one had already started with the prior fixed-noise implementation. Also cancelled the remaining old non-faithful Square jobs 8030944–8030948 and 8030950; their partial logs remain available.
- Submitted the full 12-configuration faithful Phase 1–5 sweep. p32948 jobs 8073626–8073631; p32827 jobs 8073632–8073637. Can uses state75 and Square uses state275. The first six started cleanly with no traceback/NaN/OOM; initial eval is 61.93% for Can and 33.65% for Square, and early losses are finite.
- User-reported Diffusion results: Can global best is Phase7 `rs3` at 98.8% best/final; Square remains Phase2 `s3_a2e5_cap2` at 98.0%, with Phase6 `r75e5` second at 97.5%.
- Added eight Phase8 Diffusion local-combination runs: four combine Can scale 3 with nearby ratio/actor/regularizer settings, and four interpolate Square near its 98.0% and 97.5% solutions. Added a configurable best-known multiseed launcher for Can and Square. These were pushed but intentionally not submitted on the two Flow accounts.
- Pushed all implementation and launchers to `quest-two-account-launchers`, commit `cc186f1`.

### 2026-07-28 — paired critic-only warmup sweep

- Early no-warmup faithful results showed Can falling from 61.93% by itr 5–10, while the lowest-actor-LR Phase5-safe retained 56.73% and Square Phase5-safe improved 33.65% -> 35.55%. This points to early actor/ratio updates from an immature critic rather than an execution failure.
- Corrected warmup semantics so both ratio and actor are frozen; only critic and targets update until `n_critic_warmup_itr` is reached. Previously the ratio network still updated during critic warmup.
- Submitted a paired 12-job sweep with 5-iteration critic-only warmup and otherwise identical faithful Phase 1–5 settings. p32948: 8074572–8074577; p32827: 8074578–8074583. Job names use prefix `lfw5_`. At confirmation, 8074572 was RUNNING and the remaining 11 were PENDING/Priority.
- Pushed warmup implementation and launcher at commit `20aceae` on `quest-two-account-launchers`.

### 2026-07-28 — faithful and warmup Flow outcomes

- The no-warmup faithful sweep did not solve collapse. Eleven jobs completed and one Square job remained running; every available final/latest success rate was 0%. Can never exceeded its state75 initial 61.93%; Square Phase5-safe briefly reached 35.55% from 33.65%.
- Critic-only warmup improves the policy before actor updates but does not provide long-run stability. All five completed Can warmup runs share 61.93% at itr 0 and 65.89% at itr 5 (critic-only collection), then decline after actor/ratio updates begin and finish at 0%.
- Warmup Can Phase5-safe is the slowest collapse: 65.89% at itr 5, 62.68% at itr 10, 43.84% at itr 25, 21.50% at itr 40, and 0% by itr 110. Other Can variants fall near zero by itr 40–75.
- Warmup Square Phase5-safe briefly improves 33.65% -> 40.57% at itr 10, then reaches 0% by itr 95. Warmup Square Phase1 reaches 36.54% at itr 10 and 0% by itr 50.
- Losses remain finite and Slurm jobs exit cleanly; this is policy/data-distribution collapse, not NaN/OOM/runtime failure. The paired evidence rules out missing critic warmup as the sole cause and points next to the exploration-noise/actor-update path or need for an explicit pretrained-policy constraint.

### 2026-07-28 — end-to-end MuJoCo audit and causal ablations

- The implementations are formula-aligned in replay/off-policy regime, UTD, global delays, current-actor sampling, adaptive alpha noise, G+1 GRPO normalization, ratio objective, and weighted velocity matching. They are not end-to-end identical.
- Archived MuJoCo settings differ materially: gamma .99 vs .999, batch 256 vs 1000, reward scale .2 vs 1–3, replay 1e6 vs up to 250k flattened transitions, 30k random replay prefill vs pretrained-policy collection, one primitive action vs a four-action chunk, 20 vs 4 Euler steps, and much smaller networks. The action/reward/task structure is inherently different.
- The archived HalfCheetah GRPO run itself stopped at 5,000/50,000 steps and has no evaluation rows in `log.csv`; it establishes executed update metrics but cannot substantiate successful final GRPO performance. This evidence limitation is documented in `FLOW_MUJOCO_ALIGNMENT_AUDIT.md`.
- Added per-update diagnostics for Q(data), Q(next-policy), TD target, advantage mean/std, group-Q std, ratio mean/max, and effective noise std. Added independent actor/ratio freeze controls.
- Cancelled the already-zero running Square job 8073634 and five pending redundant warmup jobs 8074579–8074583.
- Submitted eight 31-iteration Can/state75 causal ablations: p32948 8157173/8157174/8157180/8157181; p32827 8157182/8157183/8157188/8157189. They test critic-only, ratio-only, ratio-frozen uniform actor, zero/fixed noise, portable MuJoCo numeric settings, strong BC anchor, and actor LR 5e-7.
- Pushed audit, diagnostics, and launcher at commit `c2a0abf` on `quest-two-account-launchers`.

### 2026-07-28 — Diffusion Phase 8 results and Phase 9 design

- User-reported Phase8 Can winner `rs3_reg035` reached a new 99.1% peak, finished 97.6%, and averaged 96.2% over its last ten evals. `rs3_r65` peaked 98.2%, finished 97.9%, and had the best Can late mean at 96.6%.
- User-reported Phase8 Square winner `rs25` reached and finished at 98.5% (late mean 94.4%). `reg04` peaked 98.0%, finished 97.8%, and had a slightly better late mean of 94.7%.
- Added eight Phase9 combination refinements. Can combines reg .035 with ratio LR 6.5e-5 and brackets reg .03/.04 plus reward scale 3.5. Square combines reward scale 2.5 with reg .03/.04 and brackets reward scales 2.25/2.75.
- Phase9 is intended to search for a 100% peak while selecting by final and late-eval mean as well, so a single rollout fluctuation is not mistaken for a stable improvement.
- Pushed `slurm/submit_lfgpo_diffusion_phase9.sh` to `quest-two-account-launchers`, commit `7faa60d`. It was not submitted on the local Flow accounts.

### 2026-07-28 — Transport added to Diffusion Phase 9

- Recorded the current Can/Square winners and stability alternatives in `BEST_LFGPO_DIFFUSION_CONFIGS.md`.
- Transport DPPO job 7955270 starts at 17.0% and progresses through 32.0%, 48.86%, 67.50%, 79.28%, 88.32%, 91.93%, 94.96%, 96.51%, and a best/latest 96.83% at itr 120. It was terminated at the 24-hour wall-time before completing 201 itr; 96.83% is therefore a partial-run best.
- Added four Transport LFGPO-Diffusion candidates to Phase9: transferred Can winner, transferred Square winner, conservative low-LR/clip/cap settings, and a 20-itr critic warmup configuration.
- Transport jobs run 121 itr and eval every 20 itr, reducing evaluation overhead so they can target completion within the existing 24-hour runner limit.
- Pushed the extended Phase9 and best-configuration record at commit `b8e25d0` on `quest-two-account-launchers`.

### 2026-07-28 — Transport Phase 9 shortened to matched itr 50

- At user direction, changed all four Transport Phase9 runs from 121 to 51 configured iterations (indices 0–50) and restored eval frequency 10 to match the DPPO curve exactly.
- DPPO success at itr 50 is 77.95%. Both methods have 45 actual training iterations by itr 50 after excluding eval iterations, equal to `45 * 400 * 50 * 8 = 7.2M` environment steps.
- Pushed the matched-horizon correction at commit `f410db5`.

### 2026-07-28 — Transport Flow base-policy rebuild and cross-suite audit

- Confirmed ReinFlow Transport has already been attempted. State50 runs report 0%; the state150 diagnostic also remains 0% through itr 12. The unified LFGPO evaluator measures state150 at only 0.5%, so no existing ReinFlow Transport result is meaningful.
- Created `FLOW_MUJOCO_ROBOMIMIC_COMPARISON.md` with separate experiment-setting, network/code, ReinFlow, and current-result tables. Major remaining code mismatches include MuJoCo 3x256 Mish vs Robomimic 3x1024 residual ReLU policy, time embedding 16 vs 32, single-action vs action-chunk Q/ratio inputs, and ratio Mish/logit-clip 7.5 vs ReLU/clip 10.
- The original state150 checkpoint had been removed during authorized storage cleanup; only its logs/evaluation remain. Initial continuation job 8159262 therefore failed immediately without consuming meaningful GPU time.
- Updated the launcher to rebuild from scratch when no checkpoint exists. Submitted job 8159666 to train Transport Flow BC from epoch 0 through 500, save and uniformly evaluate every 25 epochs. It entered PENDING.
- Pushed comparison and initial launcher at commit `36d13ce`; pushed scratch-rebuild fallback at `01cbbdc`.

### 2026-07-28 — Transport BC final range selection

- After briefly selecting epoch 400, user restored the Transport Flow BC target to epoch 500. Cancelled the superseded 0–400 job 8160157 and submitted final job 8160201 (`bc_tr0to500_eval25`).
- Final job rebuilds from scratch, saves and evaluates every 25 epochs, and was confirmed PENDING/Priority on p32948.
- Final launcher version is pushed at commit `3ef0a8f`.

### 2026-07-28 — expanded one-factor Flow collapse ablations

- Expanded the causal matrix so every ranked implementation/optimization suspect has a direct test rather than relying on bundled parameter sweeps.
- Original eight short Can/state75 jobs remain the first layer: critic-only 8157173, ratio-frozen/uniform actor 8157174, fixed noise 8157180, BC anchor 1.0 8157181, ratio-only 8157182, zero noise 8157183, MuJoCo numeric settings 8157188, and actor LR 5e-7 8157189.
- Submitted seven complementary jobs. p32948: 8162399 (original settings plus warmup20 and MuJoCo RatioNet), 8162402 (25% positive-reward replay sampling), 8162403 (Flow actor-loss scale 0.5), 8162407 (compatible 3x256 Mish non-residual BC and online test). p32827: 8162404 (Adam with constant LR), 8162405 (combined MuJoCo numeric/optimizer/ratio settings), 8162409 (compatible horizon-1 BC and online test).
- The combined matrix isolates update ownership, ratio weighting, exploration noise, actor drift, reward imbalance, loss normalization, optimizer/schedule, numeric settings, network architecture, and action-chunk horizon.
- All seven new jobs were accepted and initially PENDING/Priority. Structural jobs first train a compatible Can BC checkpoint, then run a 31-iteration online diagnostic, because changing architecture or horizon cannot load the existing state75 checkpoint.
- RatioNet activation is now configurable while retaining ReLU as the Diffusion default; positive-replay sampling, actor-loss scaling, optimizer selection, and scheduler selection are independently configurable.
- Fixed the structural runner's checkpoint lookup to constrain it to its own named BC run, preventing concurrent architecture and horizon jobs from loading one another's incompatible checkpoints.
- Implementation and launchers were pushed at `ed3f692`; the checkpoint lookup race fix follows in the next commit.

### 2026-07-29 — Flow causal-ablation results

- All 15 causal jobs completed successfully at the Slurm/application level. The strongest controls are critic-only (61.93% initial, 70.45% best, 64.65% final) and ratio-only with actor frozen (61.93%, 70.09%, 66.05%): neither collapses. Therefore critic learning and RatioNet updates alone are not the collapse trigger.
- Uniform actor with ratio frozen still declines to 33.50%, proving actor optimization can damage the four-action-chunk policy without learned ratio weighting. Strong BC anchor 1.0 and actor LR 5e-7 slow the decline (50.00% and 52.71% final), but do not reverse it.
- Fixed/zero noise, positive-replay 25%, actor-loss 0.5, Adam/constant LR, MuJoCo numeric settings, and the combined MuJoCo settings do not solve the decline. MuJoCo numeric alone is especially harmful (0.50% final).
- The 3x256 Mish structural run starts from a weak 6.5% BC and is not a clean stability comparison; it peaks at 12% and finishes 4.5%.
- The horizon-1 run is the only full actor+ratio experiment that remains strong after warmup: 74% initial, 84% best, 76% final over the short 31-iteration test. However it also uses a newly trained epoch-150 BC and 50 eval episodes, so horizon and base-policy quality are confounded and require matched controls before attributing the gain solely to horizon.
- Current diagnosis: the collapse is driven by the actor update under four-action chunking, not by critic-only training or RatioNet-only training. Next clean tests should compare matched epoch-150 BC policies at horizons 1, 2, and 4, then run actor-only/ratio-only controls for each horizon and extend the horizon-1 winner to 151 iterations.
- Transport BC job 8160201 failed after the epoch-25 evaluation (0%); its continuation pipeline did not reach epoch 500 and must be repaired/resubmitted separately.

### 2026-07-30 — matched horizons, h1 long run, and Transport BC

- Matched Can BC sweeps completed. Horizon1 epochs 25/50/75/100/125/150: 2/14/44/28/58/74%. Horizon2: 6/79.05/19/34.31/83.93/90.08%. Horizon4: 3/25.87/61.93/69.44/93.45/88.67%.
- A close matched trio is h1-e150=74%, h2-e50=79.05%, and h4-e100=69.44%; h4-e75=61.93% is an alternative lower match.
- ReinFlow h1 job 8291116 completed and reaches 100% at multiple late evaluations, including itr150. Action horizon1 is therefore not intrinsically limiting for ReinFlow.
- LFGPO h1-safe job 8291118 starts at 74%, peaks at 84%, but declines after roughly itr40 and finishes at 2%. Horizon1 delays but does not remove the LFGPO actor-collapse mechanism.
- Late h1-safe diagnostics show ratio_mean around 0.78-0.80, normalized GRPO advantage mean around -0.36, and rollout reward near 0-0.10. This reinforces that horizon/action chunk length is not the sole root cause.
- h1 reference 8291117 and stronger-anchor 8291119 remain Pending/Priority on p32827.
- Transport BC continuation job 8290609 completed through epoch500. Success rises from 0% at e50 to 17.5% at e300, 22.5% at e375, and 30% at e500. Epoch500 is the current best Transport Flow BC checkpoint.

### 2026-07-30 — ReinFlow-style Robomimic adapter diagnostics

- Scope is now Can and Square only; Transport is excluded to concentrate the remaining rebuttal time.
- Added sampling-path switches that preserve the LFGPO objective: TD-target noise, GRPO-group noise, and chunk-aware adaptive-noise entropy can now be isolated independently. Added Q replay/group-gap and normalized actor-weight ESS diagnostics.
- Submitted 24 jobs, each 61 itr with critic-only warmup through itr19 and eval every 3 itr. Can uses state75 (61.9% initial); Square uses state275 (33.7% initial).
- p32948 shard: 8311802–8311813. It contains D0/D2/D4/D6/D8/D10 for each task and was accepted Pending/Priority.
- p32827 shard: 8311934–8311945. It contains D1/D3/D5/D7/D9/D11 for each task and was accepted Pending with reason None.
- D0–D11 test, respectively: reference, fixed sigma .1, adaptive alpha-init1, no TD noise, no GRPO noise, neither noise, chunk entropy dimension, original PPO advantage, 25% positive replay, ReinFlow-like tight trust/recent replay, gamma^4 bootstrap, and actor update frequency 8.
- Code, launcher, matrix, and decision rules are pushed to `quest-two-account-launchers` at commit `ce82d3e`.

### 2026-07-30 — conservative actor-timescale follow-up

- All 24 adapter diagnostics completed normally. Noise path, chunk entropy,
  PPO advantage, positive replay, and gamma^4 alone still collapse. D11 is the
  only clear improvement: policy frequency 8 finishes at 37.93% on Can and
  34.48% on Square, versus reference finals 15.50% and 15.50%.
- Ratio-weight ESS remains 0.994–0.999, so actor weighting is nearly uniform.
  The weighted velocity loss itself has the same horizon/action-dimension mean
  reduction as Flow BC; no broadcasting or missing-normalization error was
  found. The remaining mismatch is that ReinFlow constrains the policy directly
  with clip/KL, while LFGPO's PPO clip constrains the ratio network rather than
  the actor velocity field.
- Added direct diagnostics for actor gradient norm and MSE drift from the frozen
  initial BC velocity field.
- Submitted 16 follow-ups for Can/Square, 121 itr, warmup20, eval5. They target
  approximately 1/0.5/0.25 actor updates per iteration, actor LR 1e-5/5e-6,
  and BC anchors 0/0.1/1.0.
- p32948 jobs: 8397047, 8397048, 8397082–8397087. p32827 jobs:
  8397184–8397186 and 8397196–8397200. All were accepted; p32948 reports
  Pending/Priority and p32827 initially Pending/None.

### 2026-07-31 — first half of conservative actor sweep completed

- Eight p32948 jobs completed; the eight p32827 complementary jobs remain
  Pending/Priority.
- Can finals/last-five means: u1 11.00/19.68%, u0.25 47.52/52.34%,
  u0.5+LR5e-6 37.81/41.10%, u1+anchor1 50.74/51.80%.
- Square finals/last-five means: u1 20.59/25.15%, u0.25 33.01/35.32%,
  u0.5+LR5e-6 36.54/34.41%, u1+anchor1 33.65/36.51%.
- The late frozen-BC velocity drift strongly tracks degradation. Can u1 drift
  is 0.0608 versus 0.0095 for u0.25 and 0.0031 for anchor1. Square u1 is
  0.0703 versus 0.0044 for u0.25 and 0.0045 for anchor1.
- This is direct evidence that cumulative actor velocity-field drift, not
  numerical loss scaling or ratio-weight concentration, drives collapse.
