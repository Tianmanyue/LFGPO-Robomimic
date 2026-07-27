#!/bin/bash

# Phase-4 sweep: stronger ratio stabilization after Phases 1-3.
# Contains Can/Square x Diffusion/Flow and is intended for account 2.
# This script submits only when explicitly invoked.
# Usage: bash slurm/submit_lfgpo_phase4.sh <slurm-account> <square-flow-checkpoint>

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

# Diffusion: slower ratio network, tighter clipping, and stronger regularization.
submit ld_can_p4_r2e5_clip15 lfgpo_diffusion can \
  name=can_lfgpo_diffusion_p4_r2e5_clip15 \
  train.n_train_itr=151 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=1e-5 train.ratio_lr=2e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.15 model.max_ratio_weight=1.5 model.ratio_reg_lambda=0.1

submit ld_sq_p4_a3e5_r2e5 lfgpo_diffusion square \
  name=square_lfgpo_diffusion_p4_a3e5_r2e5 \
  train.n_train_itr=201 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=3e-5 train.ratio_lr=2e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.15 model.max_ratio_weight=1.5 model.ratio_reg_lambda=0.1

# Flow: apply the same conservative ratio update with a moderate actor step.
submit lf_can_p4_r2e5_clip15 lfgpo_flow can \
  name=can_lfgpo_flow_p4_r2e5_clip15 \
  "base_policy_path=${CAN_FLOW_CKPT}" \
  train.n_train_itr=151 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=2e-5 train.ratio_lr=2e-5 train.scale_reward_factor=3 \
  model.num_grpo_samples=32 model.ppo_eps=0.15 \
  model.max_ratio_weight=2 model.ratio_reg_lambda=0.1

submit lf_sq_p4_r2e5_clip15 lfgpo_flow square \
  name=square_lfgpo_flow_p4_r2e5_clip15 \
  "base_policy_path=${SQUARE_FLOW_CKPT}" \
  train.n_train_itr=201 train.val_freq=5 train.save_model_freq=5 \
  train.actor_lr=2e-5 train.ratio_lr=2e-5 train.scale_reward_factor=3 \
  model.num_grpo_samples=32 model.ppo_eps=0.15 \
  model.max_ratio_weight=2 model.ratio_reg_lambda=0.1
