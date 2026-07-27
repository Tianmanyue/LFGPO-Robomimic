#!/bin/bash

# Phase-5 Can LFGPO-Flow collapse-safe sweep for a secondary Quest account.
# Usage: bash slurm/submit_lfgpo_flow_phase5.sh <slurm-account>

set -euo pipefail

ACCOUNT=${1:?Usage: $0 '<slurm-account>'}
CAN_FLOW_CKPT=${CAN_FLOW_CKPT:-pretrained/flow_bc/can_reflow_state75.pt}
SEED=${LFGPO_SEED:-42}
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch

cd "${REPO}"
mkdir -p slurm/logs
if [[ ! -f "${CAN_FLOW_CKPT}" ]]; then
  echo "Missing Can Flow checkpoint: ${CAN_FLOW_CKPT}" >&2
  exit 1
fi

submit() {
  local job_name=$1
  shift
  sbatch --account="${ACCOUNT}" \
    --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=${SEED}" \
    --job-name="${job_name}" \
    "${RUNNER}" lfgpo_flow can "$@"
}

# Very small policy steps plus a strong frozen-base anchor.
submit lf_can_p5_safe \
  name=can_lfgpo_flow_p5_safe \
  "base_policy_path=${CAN_FLOW_CKPT}" \
  train.n_train_itr=151 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=2e-6 train.actor_lr_scheduler.min_lr=1e-6 \
  train.ratio_lr=1e-5 train.scale_reward_factor=1 \
  +train.policy_update_freq=48 \
  model.num_grpo_samples=32 model.ppo_eps=0.1 \
  model.max_ratio_weight=1.5 model.ratio_reg_lambda=0.1 \
  +model.bc_anchor_coef=0.3

# More policy movement, with a lighter anchor.
submit lf_can_p5_balanced \
  name=can_lfgpo_flow_p5_balanced \
  "base_policy_path=${CAN_FLOW_CKPT}" \
  train.n_train_itr=151 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=5e-6 train.actor_lr_scheduler.min_lr=2e-6 \
  train.ratio_lr=2e-5 train.scale_reward_factor=1 \
  +train.policy_update_freq=48 \
  model.num_grpo_samples=32 model.ppo_eps=0.1 \
  model.max_ratio_weight=2 model.ratio_reg_lambda=0.05 \
  +model.bc_anchor_coef=0.1
