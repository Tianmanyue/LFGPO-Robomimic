#!/bin/bash

# Best known Diffusion configurations on additional seeds.
# Usage: bash slurm/submit_lfgpo_diffusion_best_multiseed.sh <account> <seed> [all|can|square]

set -euo pipefail
ACCOUNT=${1:?Usage: $0 '<account>' '<seed>' '[all|can|square]'}
SEED=${2:?Usage: $0 '<account>' '<seed>' '[all|can|square]'}
TARGET=${3:-all}
[[ "${TARGET}" == all || "${TARGET}" == can || "${TARGET}" == square ]] || exit 2
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch
cd "${REPO}"
mkdir -p slurm/logs

if [[ "${TARGET}" == all || "${TARGET}" == can ]]; then
  sbatch --account="${ACCOUNT}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=${SEED}" \
    --job-name="ld_best_can_s${SEED}" "${RUNNER}" lfgpo_diffusion can \
    "name=ld_best_can_rs3_s${SEED}" train.n_train_itr=151 train.val_freq=5 train.save_model_freq=10 \
    train.actor_lr=1.25e-5 train.ratio_lr=7.5e-5 train.scale_reward_factor=3 \
    model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
fi
if [[ "${TARGET}" == all || "${TARGET}" == square ]]; then
  sbatch --account="${ACCOUNT}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=${SEED}" \
    --job-name="ld_best_sq_s${SEED}" "${RUNNER}" lfgpo_diffusion square \
    "name=ld_best_sq_p2_s${SEED}" train.n_train_itr=201 train.val_freq=5 train.save_model_freq=10 \
    train.actor_lr=2e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2 \
    model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
fi
