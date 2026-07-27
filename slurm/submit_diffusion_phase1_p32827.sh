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

# Transport: preserve the 17% base with conservative actor/ratio updates and longer warmup.
submit ld_tr_t1_cons transport \
  name=transport_lfgpo_diffusion_t1_cons \
  train.n_train_itr=30 train.val_freq=3 train.save_model_freq=3 \
  train.actor_lr=3e-6 train.ratio_lr=1e-4 \
  train.n_critic_warmup_itr=10 train.replay_ratio=8 train.scale_reward_factor=2

submit ld_tr_t2_cons transport \
  name=transport_lfgpo_diffusion_t2_cons \
  train.n_train_itr=30 train.val_freq=3 train.save_model_freq=3 \
  train.actor_lr=3e-6 train.ratio_lr=3e-5 \
  train.n_critic_warmup_itr=15 train.replay_ratio=4 train.scale_reward_factor=5

submit ld_tr_t3_cons transport \
  name=transport_lfgpo_diffusion_t3_cons \
  train.n_train_itr=30 train.val_freq=3 train.save_model_freq=3 \
  train.actor_lr=1e-5 train.ratio_lr=1e-4 \
  train.n_critic_warmup_itr=15 train.replay_ratio=4 train.scale_reward_factor=2
