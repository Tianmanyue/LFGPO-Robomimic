#!/bin/bash

# Phase-3 sweep: one non-overlapping candidate for each LFGPO setting
# (Can/Square x Diffusion/Flow). This script prepares jobs but does not run
# unless explicitly invoked.
# Usage: bash slurm/submit_lfgpo_phase3.sh <slurm-account> <square-flow-checkpoint>

set -euo pipefail

ACCOUNT=${1:?Usage: $0 '<slurm-account>' '<square-flow-checkpoint>'}
SQUARE_FLOW_CKPT=${2:?Usage: $0 '<slurm-account>' '<square-flow-checkpoint>'}
CAN_FLOW_CKPT=${CAN_FLOW_CKPT:-pretrained/flow_bc/can_reflow_state75.pt}
SEED=${LFGPO_SEED:-42}
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch

cd "${REPO}"
mkdir -p slurm/logs

for ckpt in "${CAN_FLOW_CKPT}" "${SQUARE_FLOW_CKPT}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing Flow base-policy checkpoint: ${ckpt}" >&2
    exit 1
  fi
done

submit() {
  local job_name=$1
  local method=$2
  local env_name=$3
  shift 3
  sbatch \
    --account="${ACCOUNT}" \
    --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=${SEED}" \
    --job-name="${job_name}" \
    "${RUNNER}" "${method}" "${env_name}" "$@"
}

# Diffusion Phase 3: lower ratio learning rate and moderate ratio constraints.
# These do not duplicate Phase 1 (C1/C2, S1/S2) or Phase 2 (C3/C4, S3/S4).
submit ld_can_p3_r5e5_cap3 lfgpo_diffusion can \
  name=can_lfgpo_diffusion_p3_r5e5_cap3 \
  train.n_train_itr=151 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=5e-6 train.ratio_lr=5e-5 train.scale_reward_factor=2 \
  model.max_ratio_weight=3 model.ratio_reg_lambda=0.02

submit ld_sq_p3_r5e5_rs3 lfgpo_diffusion square \
  name=square_lfgpo_diffusion_p3_r5e5_rs3 \
  train.n_train_itr=201 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=1e-5 train.ratio_lr=5e-5 train.scale_reward_factor=3 \
  model.max_ratio_weight=3 model.ratio_reg_lambda=0.02

# Flow Phase 3: same stability direction, with the selected common BC starts.
submit lf_can_p3_r5e5_cap3 lfgpo_flow can \
  name=can_lfgpo_flow_p3_r5e5_cap3 \
  "base_policy_path=${CAN_FLOW_CKPT}" \
  train.n_train_itr=151 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=5e-6 train.ratio_lr=5e-5 train.scale_reward_factor=2 \
  model.num_grpo_samples=32 model.max_ratio_weight=3 model.ratio_reg_lambda=0.02

submit lf_sq_p3_r5e5_cap3 lfgpo_flow square \
  name=square_lfgpo_flow_p3_r5e5_cap3 \
  "base_policy_path=${SQUARE_FLOW_CKPT}" \
  train.n_train_itr=201 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=1e-5 train.ratio_lr=5e-5 train.scale_reward_factor=2 \
  model.num_grpo_samples=32 model.max_ratio_weight=3 model.ratio_reg_lambda=0.02
