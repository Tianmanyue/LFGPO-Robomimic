#!/bin/bash

# Time-boxed rebuttal track: one matched ReinFlow baseline and three long
# LFGPO-Flow horizon-1 runs from the same epoch-150 BC checkpoint.
# Usage: bash slurm/submit_flow_h1_rebuttal.sh <account-a> <account-b>

set -euo pipefail
ACCOUNT_A=${1:?Usage: $0 '<account-a> <account-b>'}
ACCOUNT_B=${2:?Usage: $0 '<account-a> <account-b>'}
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch
CKPT=runtime/log/robomimic/pretrain/can_pre_reflow_mlp_ta1_td100/2026-07-29_03-57-37_42/checkpoint/state_150.pt
cd "${REPO}"
[[ -f "${CKPT}" ]] || { echo "Missing horizon-1 BC: ${CKPT}" >&2; exit 1; }

submit() {
  local account=$1 name=$2 method=$3
  shift 3
  sbatch --account="${account}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=42" \
    --job-name="${name}" "${RUNNER}" "${method}" can \
    "name=${name}" "base_policy_path=${CKPT}" horizon_steps=1 act_steps=1 \
    train.n_train_itr=151 train.val_freq=5 train.save_model_freq=25 "$@"
}

# Same-policy-class baseline at the same horizon and BC checkpoint.
submit "${ACCOUNT_A}" rf_can_h1_e150 reinflow

COMMON=(train.n_critic_warmup_itr=20 train.replay_ratio=1 \
  +train.policy_update_freq=2 +train.target_update_freq=2 \
  train.critic_lr=3e-4 train.scale_reward_factor=1 \
  model.num_grpo_samples=16 model.ppo_eps=0.2 model.max_ratio_weight=5 \
  model.ratio_reg_lambda=0.01 +model.adaptive_sampling_noise=true \
  +model.noise_scale=0.1 +model.alpha_init=5.0 \
  +model.target_entropy_scale=0.9 +model.use_target_actor_for_sampling=false \
  +model.grpo_include_replay_action=true \
  +model.ratio_net.activation_type=Mish +model.ratio_net.logit_clip=7.5)

# Exact extension of the stable 31-iteration structural diagnostic.
submit "${ACCOUNT_B}" lf_can_h1_ref lfgpo_flow "${COMMON[@]}" \
  train.actor_lr=1e-5 train.ratio_lr=3e-4

# Moderate and strongly anchored alternatives hedge against late actor drift.
submit "${ACCOUNT_A}" lf_can_h1_safe lfgpo_flow "${COMMON[@]}" \
  train.actor_lr=5e-6 train.ratio_lr=1e-4 +model.bc_anchor_coef=0.3
submit "${ACCOUNT_B}" lf_can_h1_anchor lfgpo_flow "${COMMON[@]}" \
  train.actor_lr=2e-6 train.ratio_lr=5e-5 +model.bc_anchor_coef=1.0

echo "Submitted horizon-1 rebuttal track: ReinFlow + 3 LFGPO runs"
