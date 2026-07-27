#!/bin/bash

# Phase-6 LFGPO-Diffusion refinement based on Phase-2 winners.
# Intended for account 2; this script submits only when explicitly invoked.
# Usage: bash slurm/submit_lfgpo_diffusion_phase6.sh <slurm-account>

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
    "name=${name}" train.val_freq=5 train.save_model_freq=5 "$@"
}

# Can: refine c3_cap2 (97.3%) without the overly restrictive clip=0.1.
submit ld_p6_can_a15e5 can train.n_train_itr=151 \
  train.actor_lr=1.5e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
submit ld_p6_can_reg02 can train.n_train_itr=151 \
  train.actor_lr=1e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.02

# Square: local refinement around s3_a2e5_cap2 (98.0%).
submit ld_p6_sq_a25e5 square train.n_train_itr=201 \
  train.actor_lr=2.5e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
submit ld_p6_sq_clip15 square train.n_train_itr=201 \
  train.actor_lr=2e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2 \
  model.ppo_eps=0.15 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
