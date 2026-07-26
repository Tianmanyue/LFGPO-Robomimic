# Guide: Running LFGPO (diffusion + flow) on the second server

> 🗑️ **INTERNAL — DELETE BEFORE/AFTER SUBMISSION.** This guide is an internal work doc. When it
> is copied into the clean deliverable `LFGPO-Robomimic`, it must be **removed once experiments
> are finished** — it is NOT part of the code released to reviewers.

> **For the agent on the OTHER server.** This server runs OUR method (LFGPO-Diffusion +
> LFGPO-Flow-GRPO) on robomimic — flow is STATE for all 3 tasks (image dropped, see §6). The
> primary server runs the 4 diffusion/flow state baselines.
>
> **Clone the git repo `LFGPO-Robomimic`** (`/home/vcj9002/shuyang/xiao/LFGPO-Robomimic`) — it is a
> ReinFlow fork with all our LFGPO code + configs committed and instantiation-validated. (It was
> developed in the sandbox `xiao/ReinFlow-debug`; LFGPO-Robomimic is the clean committed copy.)
> This doc covers env setup, what is validated, and the remaining steps: pretrain state flow BC
> (§6), then run. Gaps (3) ckpt loading + (4) end-to-end are validated on our side.

---

## 0. Our LFGPO code (already written, in the ReinFlow-debug fork)
- `model/diffusion/diffusion_lfgpo.py` — `RatioNet` + `LFGPODiffusion` (twin-Q critic, PPO-clip
  ratio net, ratio-weighted drift matching). **Validated end-to-end** (runs on `can`).
- `agent/finetune/lfgpo/train_lfgpo_diffusion_agent.py` — off-policy LFGPO loop.
- `agent/finetune/lfgpo/train_lfgpo_flow_agent.py` — subclass of the diffusion agent (shared loop).
- `model/flow/ft_lfgpo/lfgpo_flow.py` — `LFGPOFlow` (extends `ReFlow`; twin-Q + **GRPO** group
  advantage G=32 + ratio net + ratio-weighted **velocity matching**). **Core validated, NOT
  end-to-end** (see §3).
- Configs: `cfg/robomimic/finetune/{can,square,transport}/ft_lfgpo_diffusion_mlp.yaml`,
  `cfg/robomimic/finetune/transport/ft_lfgpo_flow_mlp.yaml`.

Design (confirmed): LFGPO is **off-policy** (replay + twin Q + target policy Polyak). Diffusion
ratio update = **PPO**; flow ratio update = **GRPO** (group-relative advantage). Port of JAX
`xiao/lfgpo/relax/algorithm/{lfgpo.py, lfgpo_flow_grpo.py}`.

---

## 1. Environment setup (same recipe as primary server)
```bash
conda create --name reinflow_robomimic --clone reinflow    # or rebuild: py3.8 + torch2.4.1cu121
conda activate reinflow_robomimic
pip install "robosuite==1.4.1" "robomimic==0.3.0" "mujoco==3.1.6" beautifulsoup4
# ⚠️ mujoco MUST be 3.1.6 (DPPO's pinned combo with robosuite 1.4.1). With 3.2.3 the can
#    physics differ and DPPO/LFGPO can't learn to hold the object -> success stuck at 0%.
python <ENV>/lib/python3.8/site-packages/robosuite/scripts/setup_macros.py
pip install -e /path/to/ReinFlow-debug --no-deps     # re-point editable install to YOUR clone
```
Then **always** `source set_env.sh` after activating (sets mujoco_py `LD_LIBRARY_PATH`,
`REINFLOW_DIR`, `DPPO_LOG_DIR`, `DPPO_DATA_DIR`, wandb entity). Edit `CKPT_ROOT` in set_env.sh
for this machine. Verify robomimic env: build `PickPlaceCan` and step once (see ROBOMIMIC_PLAN §9.1).

Known gotchas (all handled by set_env.sh / our configs, but watch for them):
- `import mujoco_py` needs `LD_LIBRARY_PATH` incl. `~/.mujoco/mujoco210/bin`.
- Use `+sim_device=cuda:N` (Hydra append) to enable EGL rendering.
- DPPO baseline configs have a STALE `_target_` (`agent.finetune.train_ppo_diffusion_agent` →
  override to `agent.finetune.dppo.train_ppo_diffusion_agent...`). **Our LFGPO configs are
  already correct** — no override needed.

---

## 2. LFGPO-Diffusion — VALIDATED, ready to run (state: can/square/transport)
```bash
conda activate reinflow_robomimic && cd ReinFlow-debug && source set_env.sh
python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_lfgpo_diffusion_mlp \
  device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
# same for square / transport (their checkpoints are state_8000, actor is [1024]³/time32 — already set)
```
The base checkpoints auto-download via gdown. Smoke first with `env.n_envs=10 train.n_train_itr=4
train.val_freq=2` to confirm, then full. Run 3 seeds (42, 43, 44) per task.

---

## 3. ⚠️ GAP 3 — Validate LFGPO-Flow BC-checkpoint loading (DO THIS FIRST for flow)
`LFGPOFlow` loads the flow (reflow) BC checkpoint into its `FlowMLP` network (ReinFlow format:
`data["model"]`, strip `network.` prefix). This was **NOT validated** because no reflow checkpoint
was on disk. Two risks:
1. **Checkpoint download**: the transport-state reflow ckpt path in the config is
   `${REINFLOW_LOG_DIR}/robomimic/pretrain/transport_pre_reflow_mlp_ta8_td100/.../state_50.pt`.
   Confirm it is in `script/download_url.py::get_checkpoint_download_url` (ReinFlow released mostly
   IMAGE flow ckpts — verify the STATE transport reflow ckpt downloads; if missing, get it from
   ReinFlow's HF repo `ReinFlow/ReinFlow-data-checkpoints-logs` or pretrain it).
2. **Architecture match** (the diffusion square/transport port hit exactly this): the `FlowMLP` in
   `ft_lfgpo_flow_mlp.yaml` is `time_dim=32, mlp_dims=[1024,1024,1024], ReLU, residual` — it MUST
   match the checkpoint or `load_state_dict` throws a size-mismatch. If it throws, read the shapes
   in the error and fix the config's `model.network` block to match (compare against
   `cfg/robomimic/finetune/transport/ft_ppo_reflow_mlp.yaml`'s actor).

**Validate**: instantiate the model WITH the real checkpoint (network_path set) via hydra and
confirm it loads (no size mismatch). Quick script pattern is in ROBOMIMIC_PLAN §; or just launch §4.

---

## 4. ⚠️ GAP 4 — LFGPO-Flow end-to-end smoke (transport-state)
```bash
python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_lfgpo_flow_mlp \
  device=cuda:0 +sim_device=cuda:0 wandb=null \
  env.n_envs=10 train.n_train_itr=4 train.val_freq=2 seed=42
```
Expect: BC flow policy loads → itr 0 eval (nonzero-ish reward) → itr 1 train logs
`loss actor / loss critic / loss ratio` with no error. If it completes ≥1 train itr, the flow loop
works. Then run full (n_train_itr 201, n_envs 50, 3 seeds).

Perf watch: GRPO samples **G=32** flow rollouts per replay batch — if too slow, lower
`model.num_grpo_samples` (e.g. 16) and note it; also `train.batch_size` / `replay_ratio` control cost.

---

## 5. reward_scale (shared open item — flag to human)
LFGPO configs use `train.scale_reward_factor: 0.5` (from MuJoCo). Robomimic reward is **sparse
0/1**, so this almost certainly needs retuning for off-policy Q-learning. If diffusion/flow success
stays flat, sweep `scale_reward_factor ∈ {0.1, 0.5, 1, 5, 10}` on one task and report. **Do not
change without telling the human** (they want to keep methods comparable).

---

## 6. Flow is STATE for ALL 3 tasks — pretrain can/square state flow BC (decision C)

**Decision (human-confirmed): drop image flow entirely. Flow runs STATE-based on can/square/transport**,
consistent with the diffusion side. Reason: ReinFlow only released IMAGE flow checkpoints, and
off-policy image RL (vision encoders for policy + twin-Q + ratio + image replay buffer) is too
heavy/risky for the timeline. Instead we pretrain our own STATE flow BC.

⚠️ **All 3 tasks need self-pretrained state flow BC.** The transport state reflow ckpt referenced
in `ft_ppo_reflow_mlp.yaml` (`transport_pre_reflow_mlp_ta8_td100/.../state_50.pt`) is **NOT in
`script/download_url.py`** (only image flow ckpts are) — so it won't auto-download. Pretrain it too.
Good news: reflow BC is cheap — the pretrain config's `n_epochs: 50` (that's what `state_50` means),
~30–60 min/task.

### 6a. Pretrain state flow BC for ALL 3 tasks (~30–60 min each)
- **transport**: config already exists — `cfg/robomimic/pretrain/transport/pre_reflow_mlp.yaml`
  (STATE reflow, 50 epochs, `agent.pretrain.train_reflow_agent.TrainReFlowAgent`). Just run it.
- **can / square**: create `cfg/robomimic/pretrain/{can,square}/pre_reflow_mlp.yaml` by adapting the
  transport one — change `env_name`, `obs_dim=23`, `action_dim=7`, `horizon_steps=4`, the
  `low_dim_keys` (single-arm 4 keys, same as their diffusion configs), FlowMLP dims to match, and
  the output ckpt path.

The demo data `train.npz` auto-downloads via gdown (same as normalization; state demos exist — the
diffusion state pretrain uses the same file). Run:
```bash
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_reflow_mlp
python script/run.py --config-dir=cfg/robomimic/pretrain/can       --config-name=pre_reflow_mlp
python script/run.py --config-dir=cfg/robomimic/pretrain/square    --config-name=pre_reflow_mlp
```
The resulting ckpt's FlowMLP arch matches `ft_lfgpo_flow_mlp.yaml` (both time_dim 32 / [1024]³ /
ReLU — verified for transport), so it loads cleanly into `LFGPOFlow` (no size-mismatch).

### 6b. Quality target + SAFEGUARD (do NOT skip)
- Target is a **converged BC fit** (flow velocity-matching loss plateaus low) — NOT high success
  (diffusion BC on can is ~0% too; RL does the lifting).
- **Before running RL on a pretrained ckpt**: (1) confirm BC loss converged; (2) eval the BC ckpt —
  it should show sensible reaching/grasping; (3) sanity-compare its loss magnitude / eval behaviour
  against ReinFlow's released **transport `state_50`** (a known-good state flow BC) as a reference.
- If a state flow BC clearly fails to fit, STOP and tell the human — fallback is image (heavy) or
  dropping that task's flow result.

### 6c. Cascade — configs status
Already exist (transport): `pretrain/transport/pre_reflow_mlp.yaml`, `finetune/transport/
ft_lfgpo_flow_mlp.yaml` (OUR method, gap-6 hparams set + end-to-end smoke-tested),
`finetune/transport/ft_ppo_reflow_mlp.yaml` (ReinFlow flow baseline). **After pretraining, point the
`base_policy_path` in BOTH transport finetune configs at your freshly-pretrained `state_50.pt`** (the
current path won't exist until you pretrain).

**can & square configs are ALREADY BUILT + instantiation-validated** (all 6 files exist):
`cfg/robomimic/pretrain/{can,square}/pre_reflow_mlp.yaml`,
`cfg/robomimic/finetune/{can,square}/ft_lfgpo_flow_mlp.yaml` (OUR method),
`cfg/robomimic/finetune/{can,square}/ft_ppo_reflow_mlp.yaml` (ReinFlow baseline).
You only need to: (1) run the pretrain, (2) set `base_policy_path` in the two finetune configs to
the pretrained ckpt (the placeholder is `.../{env}_pre_reflow_mlp_ta4_td100/PRETRAINED_42/checkpoint/
state_50.pt` — replace `PRETRAINED` with your pretrain run's actual timestamp). FlowMLP arch already
matches the pretrain config, so no size-mismatch on load.

### 6d. ⚠️ Perf — GRPO flow is SLOW
LFGPO-Flow smoke on transport measured **~275 s/itr at n_envs=10** (GRPO samples `num_grpo_samples`
flow ODE rollouts per replay minibatch — the dominant cost). At n_envs=50 × 201 itr this is many
hours. Mitigations, in order: keep `num_grpo_samples: 16` (already set), and if still too slow,
reduce `train.n_steps` and/or `train.replay_ratio`. Report per-itr wall-clock to the human before
committing all seeds.

## 7. gap-6 hyperparameters — ALREADY SET (do not re-tune, no time)
`transport/ft_lfgpo_flow_mlp.yaml` now uses the flow-experiment values (from `xiao/lfgpo`
`scripts/train_mujoco.py` flow defaults): **actor_lr 1e-4, critic_lr 3e-4, ratio_lr 3e-4,
num_grpo_samples 16** (16 = default, chosen over 32 for speed), `delay`/tau standard. Carry these
into the can/square flow configs. Goal is simply to beat the baseline — do not sweep unless a run
clearly diverges.

---

## 9. 🚀 Experiment Queue — sbatch-ready command list

### 🔑 Checkpoints — how each is obtained (READ FIRST)
- **Diffusion methods (LFGPO-Diffusion + DPPO baseline)** → BC checkpoints + normalization
  **auto-download** from Google Drive via `script/run.py` (uses `script/download_url.py` + gdown) on
  the FIRST run of each task. **All 3 tasks use DPPO-original `state_8000`** (can/square/transport,
  consistent). No manual download — just run; needs internet. (`state_8000` = the stronger BC ckpt;
  we switched can from `state_5000` because it plateaued.)
- **Flow methods (LFGPO-Flow + ReinFlow baseline)** → state reflow ckpts are NOT released, so you
  **pretrain them yourself** (Step A below, ~5–60 min each), then point `base_policy_path` at the
  result. Both flow finetune configs of a task share that one pretrained ckpt.
- **Normalization** (`normalization.npz`) auto-downloads with the diffusion path too; flow reuses it.

**End-to-end flow for the other agent:** clone → §1 env (mujoco 3.1.6!) → `source set_env.sh` →
run Step A flow pretrains → set base_policy_path → sbatch Step B (diffusion needs no pretrain,
ckpts auto-download). Everything below is copy-paste.

Full sweep = **3 flow pretrains + 4 methods × 3 tasks × 3 seeds (=36) finetunes**. Every finetune
run is one `python script/run.py ...`; wrap each in one sbatch job (template at the bottom).
Env prep once per node: `conda activate reinflow_robomimic && cd <repo> && source set_env.sh`.

### Step A — flow BC pretrain (RUN FIRST; ~30–60 min each; no seed/GPU-render needed)
```bash
python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_reflow_mlp device=cuda:0 wandb=null
python script/run.py --config-dir=cfg/robomimic/pretrain/can       --config-name=pre_reflow_mlp device=cuda:0 wandb=null
python script/run.py --config-dir=cfg/robomimic/pretrain/square    --config-name=pre_reflow_mlp device=cuda:0 wandb=null
```
Each saves to `${REINFLOW_LOG_DIR}/robomimic/pretrain/<env>_pre_reflow_mlp_ta*_td100/<TIMESTAMP>_42/checkpoint/state_50.pt`.
**Then set `base_policy_path` to that exact path in the flow finetune configs** (replace the
`PRETRAINED_42` placeholder): `finetune/{can,square,transport}/ft_lfgpo_flow_mlp.yaml` AND
`finetune/{can,square,transport}/ft_ppo_reflow_mlp.yaml`. (transport's two configs point at a
fixed old path — repoint them too.) The flow finetune runs (B4, B5) DEPEND on this.

### Step B — finetune sweep (each line × seed ∈ {42,43,44}; add `seed=<S>`)
Common flags: `device=cuda:0 +sim_device=cuda:0 wandb=null` (`+sim_device` enables EGL; drop it → osmesa 3× slower).

**B1 · LFGPO-Diffusion (ours):**
```bash
python script/run.py --config-dir=cfg/robomimic/finetune/can       --config-name=ft_lfgpo_diffusion_mlp   device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
python script/run.py --config-dir=cfg/robomimic/finetune/square    --config-name=ft_lfgpo_diffusion_mlp   device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_lfgpo_diffusion_mlp   device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
```
**B2 · DPPO diffusion baseline** (add `_target_` override — DPPO config's is stale):
```bash
python script/run.py --config-dir=cfg/robomimic/finetune/can       --config-name=ft_ppo_diffusion_mlp _target_=agent.finetune.dppo.train_ppo_diffusion_agent.TrainPPODiffusionAgent device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
python script/run.py --config-dir=cfg/robomimic/finetune/square    --config-name=ft_ppo_diffusion_mlp _target_=agent.finetune.dppo.train_ppo_diffusion_agent.TrainPPODiffusionAgent device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_diffusion_mlp _target_=agent.finetune.dppo.train_ppo_diffusion_agent.TrainPPODiffusionAgent device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
```
(diffusion base ckpts auto-download; no pretrain needed for B1/B2.)

**B4 · LFGPO-Flow (ours; needs Step A ckpt):**
```bash
python script/run.py --config-dir=cfg/robomimic/finetune/can       --config-name=ft_lfgpo_flow_mlp device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
python script/run.py --config-dir=cfg/robomimic/finetune/square    --config-name=ft_lfgpo_flow_mlp device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_lfgpo_flow_mlp device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
```
**B5 · ReinFlow flow baseline (needs Step A ckpt):**
```bash
python script/run.py --config-dir=cfg/robomimic/finetune/can       --config-name=ft_ppo_reflow_mlp device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
python script/run.py --config-dir=cfg/robomimic/finetune/square    --config-name=ft_ppo_reflow_mlp device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_reflow_mlp device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
```

### Priority / ordering
- **Time-boxed?** Do **seed=42 only first** (all 12 runs) to get the main comparison curves; add 43/44 later.
- Flow (B4/B5) blocked until Step A pretrain + base_policy_path update. Diffusion (B1/B2) can start immediately.
- Flagship = **can** (diffusion+flow) — queue those first.
- ⚠️ **GRPO flow is slow** (~275 s/itr @ n_envs=10; transport worst). Watch wall-clock; if needed lower `model.num_grpo_samples` (16→8) or `train.n_steps`.

### sbatch template (one job per run)
```bash
#!/bin/bash
#SBATCH --job-name=lfgpo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16      # 50 envs are CPU-heavy; give enough cores
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%x_%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reinflow_robomimic
cd /path/to/LFGPO-Robomimic
source set_env.sh
srun python script/run.py <the exact args from B1/B2/B4/B5 above, with this run's device=cuda:0 +sim_device=cuda:0 seed=$SEED>
```

### Results → curves
Each run writes eval success-rate to wandb (if entity set) AND a local `.pkl` under its logdir
(recover with `util/pkl2wandb.py`). Plot success-rate vs env-step per task, overlaying the 4 methods
(reuse ReinFlow plotting in `xiao/lfgpo/scripts/plot_*` or ReinFlow's own).

## 8. Report back to human
After §3+§4 pass (and §6 pretraining for can/square): report (a) that flow checkpoint loads +
end-to-end runs, (b) BC pretrain loss/eval for can/square + comparison to transport state_50,
(c) any arch fix you made, (d) per-itr wall-clock, (e) first eval numbers. Then wait for sign-off
before full fan-out.

Reference: `xiao/lfgpo/ROBOMIMIC_PLAN.md` (full plan + setup log + gotchas).
