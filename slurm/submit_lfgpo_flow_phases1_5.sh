#!/bin/bash

# Corrected LFGPO-Flow Phases 1-5 sweep (Can/Square only).
# All runs use policy_update_freq=48 to match ReinFlow's ~5 actor updates/itr
# while retaining dense critic updates.  Shard 0/1 splits the 12 jobs evenly
# across two Slurm accounts without duplicate configurations.
# Usage: bash slurm/submit_lfgpo_flow_phases1_5.sh <account> <0|1>

set -euo pipefail

ACCOUNT=${1:?Usage: $0 '<account>' '<0|1>'}
SHARD=${2:?Usage: $0 '<account>' '<0|1>'}
[[ "${SHARD}" == 0 || "${SHARD}" == 1 ]] || { echo "Shard must be 0 or 1" >&2; exit 2; }

CAN_CKPT=${CAN_FLOW_CKPT:-pretrained/flow_bc/can_reflow_state75.pt}
SQUARE_CKPT=${SQUARE_FLOW_CKPT:-pretrained/flow_bc/square_reflow_state275.pt}
SEED=${LFGPO_SEED:-42}
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch
INDEX=0

cd "${REPO}"
mkdir -p slurm/logs
for ckpt in "${CAN_CKPT}" "${SQUARE_CKPT}"; do
  [[ -f "${ckpt}" ]] || { echo "Missing Flow checkpoint: ${ckpt}" >&2; exit 1; }
done

submit() {
  local job_name=$1 env_name=$2 ckpt=$3
  shift 3
  local this_index=${INDEX}
  INDEX=$((INDEX + 1))
  if (( this_index % 2 != SHARD )); then return; fi
  sbatch --account="${ACCOUNT}" \
    --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=${SEED}" \
    --job-name="${job_name}" "${RUNNER}" lfgpo_flow "${env_name}" \
    "name=${job_name}" "base_policy_path=${ckpt}" \
    train.val_freq=5 train.save_model_freq=5 \
    +train.policy_update_freq=48 "$@"
}

# Phase 1: original ratio sweep, now with controlled policy update frequency.
submit lf5_p1_can_r1e4 can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=1e-5 train.ratio_lr=1e-4 \
  model.max_ratio_weight=5 model.ratio_reg_lambda=0.01 +model.bc_anchor_coef=0.1
submit lf5_p1_sq_r1e4 square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=1e-5 train.ratio_lr=1e-4 \
  model.max_ratio_weight=5 model.ratio_reg_lambda=0.01 +model.bc_anchor_coef=0.1

# Phase 2: actor/ratio sensitivity around the original configurations.
submit lf5_p2_can_r3e4 can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=1e-5 train.ratio_lr=3e-4 \
  model.max_ratio_weight=5 model.ratio_reg_lambda=0.01 +model.bc_anchor_coef=0.1
submit lf5_p2_sq_a2e5 square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=2e-5 train.ratio_lr=1e-4 \
  model.max_ratio_weight=5 model.ratio_reg_lambda=0.01 +model.bc_anchor_coef=0.1

# Phase 3: lower learning rates and moderate ratio constraints.
submit lf5_p3_can_r5e5 can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=5e-6 train.ratio_lr=5e-5 train.scale_reward_factor=2 \
  model.max_ratio_weight=3 model.ratio_reg_lambda=0.02 +model.bc_anchor_coef=0.1
submit lf5_p3_sq_r5e5 square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=1e-5 train.ratio_lr=5e-5 train.scale_reward_factor=2 \
  model.max_ratio_weight=3 model.ratio_reg_lambda=0.02 +model.bc_anchor_coef=0.1

# Phase 4: tight clipping and stronger ratio regularization.
submit lf5_p4_can_clip15 can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=1e-5 train.ratio_lr=2e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.15 model.max_ratio_weight=2 model.ratio_reg_lambda=0.1 +model.bc_anchor_coef=0.1
submit lf5_p4_sq_clip15 square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=1e-5 train.ratio_lr=2e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.15 model.max_ratio_weight=2 model.ratio_reg_lambda=0.1 +model.bc_anchor_coef=0.1

# Phase 5: strongest collapse protection, two variants per environment.
submit lf5_p5_can_safe can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=2e-6 train.actor_lr_scheduler.min_lr=1e-6 train.ratio_lr=1e-5 \
  model.ppo_eps=0.1 model.max_ratio_weight=1.5 model.ratio_reg_lambda=0.1 +model.bc_anchor_coef=0.3
submit lf5_p5_can_bal can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=5e-6 train.actor_lr_scheduler.min_lr=2e-6 train.ratio_lr=2e-5 \
  model.ppo_eps=0.1 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05 +model.bc_anchor_coef=0.1
submit lf5_p5_sq_safe square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=2e-6 train.actor_lr_scheduler.min_lr=1e-6 train.ratio_lr=1e-5 \
  model.ppo_eps=0.1 model.max_ratio_weight=1.5 model.ratio_reg_lambda=0.1 +model.bc_anchor_coef=0.3
submit lf5_p5_sq_bal square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=5e-6 train.actor_lr_scheduler.min_lr=2e-6 train.ratio_lr=2e-5 \
  model.ppo_eps=0.1 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05 +model.bc_anchor_coef=0.1

