#!/bin/bash

# Phase 9: combination refinement after Phase 8.
# Can anchor: rs3_reg035 peaked at 99.1%; rs3_r65 had better late stability.
# Square anchor: rs25 peaked/finished at 98.5%; reg04 improved regularization.
# Usage: bash slurm/submit_lfgpo_diffusion_phase9.sh <slurm-account>

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

# Can: combine the 99.1%-peak regularizer with the more stable lower ratio LR,
# then bracket regularization/reward scale near that solution.
submit ld_p9_can_r65_reg035 can train.n_train_itr=151 \
  train.actor_lr=1.25e-5 train.ratio_lr=6.5e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.035
submit ld_p9_can_reg03 can train.n_train_itr=151 \
  train.actor_lr=1.25e-5 train.ratio_lr=7.5e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.03
submit ld_p9_can_reg04 can train.n_train_itr=151 \
  train.actor_lr=1.25e-5 train.ratio_lr=7.5e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.04
submit ld_p9_can_rs35_reg035 can train.n_train_itr=151 \
  train.actor_lr=1.25e-5 train.ratio_lr=7.5e-5 train.scale_reward_factor=3.5 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.035

# Square: combine the 98.5% reward-scale winner with the reg=.04 neighbor and
# bracket reward/regularization without moving actor LR away from the P2 winner.
submit ld_p9_sq_rs25_reg04 square train.n_train_itr=201 \
  train.actor_lr=2e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2.5 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.04
submit ld_p9_sq_rs25_reg03 square train.n_train_itr=201 \
  train.actor_lr=2e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2.5 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.03
submit ld_p9_sq_rs225_reg04 square train.n_train_itr=201 \
  train.actor_lr=2e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2.25 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.04
submit ld_p9_sq_rs275_reg04 square train.n_train_itr=201 \
  train.actor_lr=2e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2.75 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.04
