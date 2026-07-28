#!/bin/bash

# Can-only Diffusion Phase 7: one-factor refinements around the Phase-6 winner
# r75e5 (actor_lr=1.25e-5, ratio_lr=7.5e-5, scale=2, cap=2, reg=0.05),
# which reached 98.5%. Intended for the external account-2 checkout.
# Usage: bash slurm/submit_lfgpo_diffusion_phase7_can.sh <slurm-account>

set -euo pipefail
ACCOUNT=${1:?Usage: $0 '<slurm-account>'}
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch
cd "${REPO}"
mkdir -p slurm/logs

submit() {
  local name=$1
  shift
  sbatch --account="${ACCOUNT}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=42" \
    --job-name="${name}" "${RUNNER}" lfgpo_diffusion can \
    "name=${name}" train.n_train_itr=151 train.val_freq=5 train.save_model_freq=5 \
    train.actor_lr=1.25e-5 train.ratio_lr=7.5e-5 train.scale_reward_factor=2 \
    model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05 "$@"
}

# Ratio-LR bracket immediately around 7.5e-5.
submit ld_p7_can_r6e5 train.ratio_lr=6e-5
submit ld_p7_can_r9e5 train.ratio_lr=9e-5

# Actor-LR bracket, keeping the winning ratio LR fixed.
submit ld_p7_can_a1e5 train.actor_lr=1e-5
submit ld_p7_can_a15e5 train.actor_lr=1.5e-5

# Independent trust-region/regularization refinements.
submit ld_p7_can_cap175 model.max_ratio_weight=1.75
submit ld_p7_can_reg035 model.ratio_reg_lambda=0.035
submit ld_p7_can_clip15 model.ppo_eps=0.15

# Test a stronger sparse-reward signal without changing either learning rate.
submit ld_p7_can_rs3 train.scale_reward_factor=3
