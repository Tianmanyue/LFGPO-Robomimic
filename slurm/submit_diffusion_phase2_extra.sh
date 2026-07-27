#!/bin/bash

# Complementary LFGPO-Diffusion tuning sweep focused on ratio stability.
# Run from a login user associated with the requested Slurm account.
# Usage: bash slurm/submit_diffusion_phase2_extra.sh <slurm-account>

set -euo pipefail

ACCOUNT=${1:?Usage: $0 '<slurm-account>'}
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

# Can: retain the successful learning rates while constraining ratio weights/clipping.
submit ld_can_c3_cap2 can \
  name=can_lfgpo_diffusion_c3_cap2 \
  train.n_train_itr=151 train.val_freq=5 train.save_model_freq=5 \
  train.ratio_lr=1e-4 train.scale_reward_factor=2 \
  model.max_ratio_weight=2 model.ratio_reg_lambda=0.05

submit ld_can_c4_clip01 can \
  name=can_lfgpo_diffusion_c4_clip01 \
  train.n_train_itr=151 train.val_freq=5 train.save_model_freq=5 \
  train.ratio_lr=1e-4 train.scale_reward_factor=2 \
  model.ppo_eps=0.1 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05

# Square: preserve stronger actor learning but reduce extreme ratio weighting.
submit ld_sq_s3_a2e5_cap2 square \
  name=square_lfgpo_diffusion_s3_a2e5_cap2 \
  train.n_train_itr=201 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=2e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2 \
  model.max_ratio_weight=2 model.ratio_reg_lambda=0.05

submit ld_sq_s4_rs5_cap2 square \
  name=square_lfgpo_diffusion_s4_rs5_cap2 \
  train.n_train_itr=201 train.val_freq=5 train.save_model_freq=5 \
  train.ratio_lr=1e-4 train.scale_reward_factor=5 \
  model.max_ratio_weight=2 model.ratio_reg_lambda=0.05

# Transport: highly conservative policy drift; diagnostic length only.
submit ld_tr_t4_cap2 transport \
  name=transport_lfgpo_diffusion_t4_cap2 \
  train.n_train_itr=30 train.val_freq=3 train.save_model_freq=3 \
  train.actor_lr=3e-6 train.ratio_lr=1e-4 \
  train.n_critic_warmup_itr=10 train.replay_ratio=4 train.scale_reward_factor=2 \
  model.max_ratio_weight=2 model.ratio_reg_lambda=0.05

submit ld_tr_t5_cap15 transport \
  name=transport_lfgpo_diffusion_t5_cap15 \
  train.n_train_itr=30 train.val_freq=3 train.save_model_freq=3 \
  train.actor_lr=3e-6 train.ratio_lr=3e-5 \
  train.n_critic_warmup_itr=15 train.replay_ratio=2 train.scale_reward_factor=5 \
  model.ppo_eps=0.1 model.max_ratio_weight=1.5 model.ratio_reg_lambda=0.1

