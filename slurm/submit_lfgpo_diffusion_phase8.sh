#!/bin/bash

# Phase 8: local combination refinements after Can P7 rs3=98.8% and
# Square P2 s3_a2e5_cap2=98.0%. Intended for the external Diffusion account.
# Usage: bash slurm/submit_lfgpo_diffusion_phase8.sh <slurm-account>

set -euo pipefail
ACCOUNT=${1:?Usage: $0 '<slurm-account>'}
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch
cd "${REPO}"
mkdir -p slurm/logs

submit() {
  local name=$1 env_name=$2
  shift 2
  sbatch --account="${ACCOUNT}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=42" \
    --job-name="${name}" "${RUNNER}" lfgpo_diffusion "${env_name}" \
    "name=${name}" train.val_freq=5 train.save_model_freq=10 "$@"
}

# Can: combine the winning reward scale 3 with close LR/trust-region neighbors.
submit ld_p8_can_rs3_r65 can train.n_train_itr=151 \
  train.actor_lr=1.25e-5 train.ratio_lr=6.5e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
submit ld_p8_can_rs3_r85 can train.n_train_itr=151 \
  train.actor_lr=1.25e-5 train.ratio_lr=8.5e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
submit ld_p8_can_rs3_a11 can train.n_train_itr=151 \
  train.actor_lr=1.1e-5 train.ratio_lr=7.5e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
submit ld_p8_can_rs3_reg035 can train.n_train_itr=151 \
  train.actor_lr=1.25e-5 train.ratio_lr=7.5e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.035

# Square: interpolate between the 98.0% P2 winner and 97.5% ratio-LR neighbor.
submit ld_p8_sq_r875 square train.n_train_itr=201 \
  train.actor_lr=2e-5 train.ratio_lr=8.75e-5 train.scale_reward_factor=2 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
submit ld_p8_sq_a19 square train.n_train_itr=201 \
  train.actor_lr=1.9e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
submit ld_p8_sq_rs25 square train.n_train_itr=201 \
  train.actor_lr=2e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2.5 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
submit ld_p8_sq_reg04 square train.n_train_itr=201 \
  train.actor_lr=2e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.04
