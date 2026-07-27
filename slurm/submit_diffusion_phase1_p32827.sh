#!/bin/bash

# Phase-1 LFGPO-Diffusion tuning sweep.
# GPU allocation: p32827
# All logs/checkpoints/results remain under this repository's runtime/ and slurm/logs/.

set -euo pipefail

ACCOUNT=p32827
SEED=42
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch

cd "${REPO}"
mkdir -p slurm/logs

submit() {
  local job_name=$1
  local env_name=$2
  shift 2
  sbatch \
    --account="${ACCOUNT}" \
    --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=${SEED}" \
    --job-name="${job_name}" \
    "${RUNNER}" lfgpo_diffusion "${env_name}" "$@"
}

# Can: current run is stable, so retain critic warmup=5.
submit ld_can_c1_r1e4 can \
  name=can_lfgpo_diffusion_c1_r1e4 \
  train.n_train_itr=151 train.val_freq=5 train.save_model_freq=5 \
  train.ratio_lr=1e-4

submit ld_can_c2_r1e4_rs2 can \
  name=can_lfgpo_diffusion_c2_r1e4_rs2 \
  train.n_train_itr=151 train.val_freq=5 train.save_model_freq=5 \
  train.ratio_lr=1e-4 train.scale_reward_factor=2

# Square: stable but slow; retain critic warmup=5 and test stronger learning signal.
submit ld_sq_s1_r1e4_rs2 square \
  name=square_lfgpo_diffusion_s1_r1e4_rs2 \
  train.n_train_itr=201 train.val_freq=5 train.save_model_freq=5 \
  train.ratio_lr=1e-4 train.scale_reward_factor=2

submit ld_sq_s2_a2e5_r1e4_rs2 square \
  name=square_lfgpo_diffusion_s2_a2e5_r1e4_rs2 \
  train.n_train_itr=201 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=2e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2

# Transport candidates intentionally omitted: rebuttal resources focus on Can and Square.
