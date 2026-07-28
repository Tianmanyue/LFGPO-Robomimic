#!/bin/bash

# MuJoCo-faithful off-policy LFGPO-Flow Phases 1-5 sweep (Can/Square only).
# Every run uses UTD=1, globally delayed policy/target updates, current-policy
# adaptive-noise sampling, and replay-action-inclusive GRPO normalization.
# Shard 0/1 splits the 12 jobs evenly
# across two Slurm accounts without duplicate configurations.
# Usage: bash slurm/submit_lfgpo_flow_phases1_5.sh <account> <0|1> [all|can|square] [warmup_itr]

set -euo pipefail

ACCOUNT=${1:?Usage: $0 '<account>' '<0|1>'}
SHARD=${2:?Usage: $0 '<account>' '<0|1>'}
TARGET=${3:-all}
WARMUP_ITR=${4:-0}
[[ "${SHARD}" == 0 || "${SHARD}" == 1 ]] || { echo "Shard must be 0 or 1" >&2; exit 2; }
[[ "${TARGET}" == all || "${TARGET}" == can || "${TARGET}" == square ]] || {
  echo "Target must be all, can, or square" >&2; exit 2;
}
[[ "${WARMUP_ITR}" =~ ^[0-9]+$ ]] || { echo "warmup_itr must be a non-negative integer" >&2; exit 2; }

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
  if [[ "${TARGET}" != all && "${env_name}" != "${TARGET}" ]]; then return; fi
  if (( this_index % 2 != SHARD )); then return; fi
  if (( WARMUP_ITR > 0 )); then
    job_name=${job_name/lfm_/lfw${WARMUP_ITR}_}
  fi
  sbatch --account="${ACCOUNT}" \
    --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=${SEED}" \
    --job-name="${job_name}" "${RUNNER}" lfgpo_flow "${env_name}" \
    "name=${job_name}" "base_policy_path=${ckpt}" \
    train.n_critic_warmup_itr="${WARMUP_ITR}" train.val_freq=5 train.save_model_freq=10 \
    train.replay_ratio=1 +train.policy_update_freq=2 +train.target_update_freq=2 \
    +train.alpha_lr=7e-3 +train.alpha_update_freq=250 \
    model.num_grpo_samples=16 +model.adaptive_sampling_noise=true \
    +model.noise_scale=0.1 +model.alpha_init=5.0 +model.target_entropy_scale=0.9 \
    +model.use_target_actor_for_sampling=false \
    +model.grpo_include_replay_action=true "$@"
}

# Phase 1: closest robotics-scale analogue of the MuJoCo defaults.
submit lfm_p1_can_r1e4 can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=1e-5 train.ratio_lr=1e-4 \
  model.max_ratio_weight=5 model.ratio_reg_lambda=0.01
submit lfm_p1_sq_r1e4 square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=1e-5 train.ratio_lr=1e-4 \
  model.max_ratio_weight=5 model.ratio_reg_lambda=0.01

# Phase 2: actor/ratio sensitivity around the original configurations.
submit lfm_p2_can_r3e4 can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=1e-5 train.ratio_lr=3e-4 \
  model.max_ratio_weight=5 model.ratio_reg_lambda=0.01
submit lfm_p2_sq_a2e5 square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=2e-5 train.ratio_lr=1e-4 \
  model.max_ratio_weight=5 model.ratio_reg_lambda=0.01

# Phase 3: lower learning rates and moderate ratio constraints.
submit lfm_p3_can_r5e5 can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=5e-6 train.ratio_lr=5e-5 train.scale_reward_factor=2 \
  model.max_ratio_weight=3 model.ratio_reg_lambda=0.02
submit lfm_p3_sq_r5e5 square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=1e-5 train.ratio_lr=5e-5 train.scale_reward_factor=2 \
  model.max_ratio_weight=3 model.ratio_reg_lambda=0.02

# Phase 4: tight clipping and stronger ratio regularization.
submit lfm_p4_can_clip15 can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=1e-5 train.ratio_lr=2e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.15 model.max_ratio_weight=2 model.ratio_reg_lambda=0.1
submit lfm_p4_sq_clip15 square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=1e-5 train.ratio_lr=2e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.15 model.max_ratio_weight=2 model.ratio_reg_lambda=0.1

# Phase 5: lower-LR stability variants; no extra BC anchor so the data path
# remains directly comparable to the MuJoCo algorithm.
submit lfm_p5_can_safe can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=2e-6 train.actor_lr_scheduler.min_lr=1e-6 train.ratio_lr=1e-5 \
  model.ppo_eps=0.1 model.max_ratio_weight=1.5 model.ratio_reg_lambda=0.1
submit lfm_p5_can_bal can "${CAN_CKPT}" \
  train.n_train_itr=151 train.actor_lr=5e-6 train.actor_lr_scheduler.min_lr=2e-6 train.ratio_lr=2e-5 \
  model.ppo_eps=0.1 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
submit lfm_p5_sq_safe square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=2e-6 train.actor_lr_scheduler.min_lr=1e-6 train.ratio_lr=1e-5 \
  model.ppo_eps=0.1 model.max_ratio_weight=1.5 model.ratio_reg_lambda=0.1
submit lfm_p5_sq_bal square "${SQUARE_CKPT}" \
  train.n_train_itr=201 train.actor_lr=5e-6 train.actor_lr_scheduler.min_lr=2e-6 train.ratio_lr=2e-5 \
  model.ppo_eps=0.1 model.max_ratio_weight=2 model.ratio_reg_lambda=0.05
