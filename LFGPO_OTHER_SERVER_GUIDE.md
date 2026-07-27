# Quest handoff: LFGPO-Robomimic experiments

This branch contains the code and Slurm workflow validated on Northwestern Quest for
LFGPO-Diffusion, DPPO, LFGPO-Flow-GRPO, and ReinFlow on the robomimic `can`, `square`, and
`transport` state tasks.

## 1. Clone and select this branch

```bash
git clone git@github.com:Tianmanyue/LFGPO-Robomimic.git
cd LFGPO-Robomimic
git switch quest-sbatch-hparam
```

Do not commit `runtime/`, `slurm/logs/`, generated CSV files, or MuJoCo logs.

## 2. Quest settings to change before submission

The checked-in Slurm headers use the original Quest allocation:

```text
#SBATCH --account=p32948
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
```

Change `--account`, partition, GPU request, memory, and wall time in the three
`slurm/*.sbatch` files if the second account uses a different allocation. Do not change the
Python commands just to adapt paths.

Runtime paths are portable. Inside Slurm, the scripts use `SLURM_SUBMIT_DIR` as the repository
directory (submit from the repository root) and accept:

```bash
export LFGPO_REPO=/absolute/path/to/LFGPO-Robomimic   # optional
export LFGPO_CONDA_ROOT=/absolute/path/to/anaconda3
export LFGPO_CONDA_ENV=lfgpo_robomimic
export CKPT_ROOT=/large/scratch/or/project/path/lfgpo_runtime
```

Set these in the login shell before `sbatch`; export them with `sbatch --export=ALL` if the cluster
does not propagate the submission environment. By default, `CKPT_ROOT` is `<repo>/runtime`.

Create the local output directory once after cloning, before any direct `sbatch` command:

```bash
mkdir -p slurm/logs
```

## 3. Environment setup

The validated environment uses Python 3.8, PyTorch 2.4.1 CUDA 12.1, robosuite 1.4.1,
robomimic 0.3.0, and MuJoCo 3.1.6. MuJoCo 3.2.x changed the task physics and previously caused
zero-success runs.

```bash
conda create -n lfgpo_robomimic python=3.8 -y
conda activate lfgpo_robomimic
# Install the CUDA/PyTorch stack appropriate for the server, then:
pip install "robosuite==1.4.1" "robomimic==0.3.0" "mujoco==3.1.6" beautifulsoup4 gdown
python "$CONDA_PREFIX/lib/python3.8/site-packages/robosuite/scripts/setup_macros.py"
pip install -e . --no-deps
source set_env.sh
```

The code skips the optional D4RL import by default. Set `REINFLOW_IMPORT_D4RL=1` only for D4RL
experiments. `set_env.sh` adds Conda headers/libraries required to compile `mujoco_py`.

Before submitting a sweep:

```bash
python -c 'import torch, mujoco, robosuite, robomimic; print(torch.cuda.is_available(), mujoco.__version__)'
bash -n slurm/run_finetune_seed42.sbatch
bash -n slurm/run_flow_pretrain.sbatch
bash -n slurm/run_flow_pretrain_long.sbatch
```

## 4. Important metric definition

For robomimic, a successful episode is an episode whose maximum environment reward reaches 1:

```text
success_rate = successful_completed_episodes / completed_episodes
```

`MultiStep` already sums primitive rewards inside an action chunk. Do not divide the maximum
chunk reward by `act_steps`. This branch fixes that bug in LFGPO, DPPO, and ReinFlow buffers.
The old bug could print zero success even when rollouts succeeded.

Evaluation uses 50 vector environments over one configured rollout iteration, so the number of
completed episodes is generally not exactly 50 or 10,000. Decimal success rates with four digits
are expected.

Export all run logs to plotting-friendly CSV with:

```bash
python script/export_robomimic_csv.py
```

The generated `experiment_metrics.csv` is intentionally ignored by Git.

## 5. Required checkpoints and first-run downloads

Diffusion configs use DPPO checkpoints and normalization data under `CKPT_ROOT`. Missing released
files are downloaded by `script/run.py`. A file lock now serializes concurrent first-run downloads,
so six jobs cannot corrupt the same checkpoint.

Flow uses locally trained state ReFlow BC checkpoints:

```text
pretrained/flow_bc/can_reflow_state50.pt
pretrained/flow_bc/square_reflow_state50.pt
pretrained/flow_bc/transport_reflow_state50.pt
```

The repository includes the original state-50 checkpoints. To retrain on the second account:

```bash
bash slurm/submit_seed42.sh pretrain
```

Each completed pretrain job updates the corresponding symlink in `pretrained/flow_bc/`. For a
longer continuation from state 50:

```bash
sbatch --job-name=bc_square_long slurm/run_flow_pretrain_long.sbatch square pretrained/flow_bc/square_reflow_state50.pt 150
sbatch --job-name=bc_transport_long slurm/run_flow_pretrain_long.sbatch transport pretrained/flow_bc/transport_reflow_state50.pt 150
sbatch --job-name=bc_can_long slurm/run_flow_pretrain_long.sbatch can pretrained/flow_bc/can_reflow_state50.pt 150
```

The continued run numbers its newly saved files from `state_25.pt` to `state_150.pt`; those are
150 additional epochs initialized from state 50, not training from scratch.

Evaluate a Flow BC checkpoint without updating its actor by running one eval iteration:

```bash
sbatch --job-name=eval_flowbc_square slurm/run_finetune_seed42.sbatch lfgpo_flow square \
  base_policy_path=/absolute/path/to/state_150.pt name=square_flowbc_eval \
  train.n_train_itr=1 train.n_critic_warmup_itr=2 train.val_freq=1 train.save_model_freq=999
```

Repeat for `can` and `transport`. Read `eval: success rate` from `slurm/logs/<job>-<id>.out`.

## 6. Smoke test before a sweep

Always submit one short job first:

```bash
sbatch --job-name=smoke_lfgpo_can slurm/run_finetune_seed42.sbatch lfgpo_diffusion can \
  env.n_envs=10 train.n_train_itr=4 train.val_freq=2 train.save_model_freq=99
```

A valid smoke test must show:

1. checkpoint and normalization load successfully;
2. itr 0 produces `eval: success rate ...`;
3. a training iteration prints actor, critic, and ratio loss;
4. no traceback in either `.out` or `.err`.

Only then submit the full sweep.

## 7. Submit the validated seed-42 experiments

```bash
bash slurm/submit_seed42.sh diffusion 42  # 3 LFGPO-Diffusion + 3 DPPO
bash slurm/submit_seed42.sh flow 42       # 3 LFGPO-Flow + 3 ReinFlow
```

The flow command refuses to submit if any required checkpoint is absent. The DPPO wrapper supplies
the corrected Python `_target_` automatically. The remaining positional arguments are seeds. For a
matched three-seed comparison use `bash slurm/submit_seed42.sh diffusion 42 43 44` (and likewise
for `flow` after its selected configuration is validated). Compare methods within each identical
seed, then report mean, standard deviation, paired differences, and curve AUC; do not cherry-pick
one method's favorable seed against another method's unfavorable seed.

Monitor:

```bash
squeue -u "$USER" -o '%.18i %.28j %.10T %.12M %.24R'
rg -i 'eval: success|loss actor|traceback|error executing' slurm/logs/*.out
python script/export_robomimic_csv.py
```

## 8. Current hyperparameter findings

### LFGPO-Diffusion

- Can with diffusion `state_5000`: LFGPO improved from about 69.6% to 90.9%; DPPO reached about
  99.6% in the observed seed-42 run.
- Square: LFGPO improved from about 41% to above 60% while DPPO reached about 92%.
- Transport: the current LFGPO setting collapsed to zero while DPPO improved. Transport is the
  main diffusion tuning target. Do not assume this is the metric bug; it persisted after the fix.

For Transport, tune one variable at a time with 30-iteration runs and eval every 3 iterations.
Recommended first sweep: actor LR `{3e-6, 1e-5, 3e-5}`, ratio LR `{3e-5, 1e-4, 3e-4}`, then
reward scale if needed. Keep the same checkpoint, seed, and evaluation settings.

### LFGPO-Flow

The validated Can short-run setting is:

```text
advantage_mode=grpo
num_grpo_samples=32
actor_lr=1e-5
ratio_lr=1e-4
n_train_itr=30
val_freq=3
```

It improved Can from roughly 23.4% base success to 37.3% at itr 27. The PPO-style advantage
variant fell to about 2%, so use GRPO. A 150-iteration run is justified, but eval every 3–5
iterations and save at the same frequency because later regression remains possible.

Submit the 30-iteration Can tuning template:

```bash
bash slurm/submit_lfgpo_flow_can_tune.sh 1e-4 1e-5 32 ratio1e4 grpo
```

Submit a 150-iteration continuation-style run from a selected BC checkpoint (this starts RL from
that BC; it does not resume the prior 30-iteration RL optimizer):

```bash
sbatch --job-name=lf_can_grpo150 slurm/run_finetune_seed42.sbatch lfgpo_flow can \
  base_policy_path=/absolute/path/to/selected_flow_bc.pt \
  name=can_lfgpo_flow_grpo150 train.n_train_itr=150 train.val_freq=3 train.save_model_freq=3 \
  train.ratio_lr=1e-4 train.actor_lr=1e-5 model.num_grpo_samples=32 model.advantage_mode=grpo
```

Do not launch formal Square/Transport Flow runs until their newly trained BC policies have been
evaluated. Record the base success, checkpoint path, job ID, and all overrides for every sweep.

## 9. Handoff checklist for the other agent

1. Confirm branch and commit hash.
2. Change Slurm account/partition and export Conda/runtime paths.
3. Verify MuJoCo 3.1.6 and run one smoke job.
4. Evaluate the selected Flow BC checkpoint before RL.
5. Run only one 30-iteration tuning job per candidate; eval every 3 iterations.
6. Stop a candidate if success collapses repeatedly or logs contain NaN/traceback.
7. Preserve `.out`, `.err`, Hydra config/overrides, `result.pkl`, and exported CSV.
8. Report results as `(task, method, checkpoint, seed, hyperparameters, itr, success_rate)`.
