#!/bin/bash

# Follow-up around the stable low-update/low-drift region found in the first
# conservative sweep. Can and Square only; 12 jobs total.
# Usage: bash slurm/submit_lfgpo_flow_actor_refine.sh ACCOUNT

set -euo pipefail
ACCOUNT=${1:?Usage: $0 ACCOUNT}
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch
cd "${REPO}"
mkdir -p slurm/logs

submit() {
  local env_name=$1 name=$2 ckpt=$3 pfreq=$4 actor_lr=$5 anchor=$6
  sbatch --account="${ACCOUNT}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=42" \
    --job-name="${name}" "${RUNNER}" lfgpo_flow "${env_name}" \
    "name=${name}" "base_policy_path=${ckpt}" \
    train.n_train_itr=121 train.n_critic_warmup_itr=20 \
    train.val_freq=5 train.save_model_freq=30 train.replay_ratio=1 \
    train.batch_size=1000 train.buffer_size=5000 \
    "train.actor_lr=${actor_lr}" train.critic_lr=3e-4 train.ratio_lr=1e-4 \
    train.scale_reward_factor=1 ++train.use_lr_scheduler=false \
    "++train.policy_update_freq=${pfreq}" ++train.target_update_freq=2 \
    ++train.alpha_lr=7e-3 ++train.alpha_update_freq=250 \
    model.num_grpo_samples=16 model.ppo_eps=0.1 \
    model.max_ratio_weight=2 model.ratio_reg_lambda=0.05 \
    ++model.adaptive_sampling_noise=true ++model.noise_scale=0.1 \
    ++model.alpha_init=1.0 ++model.target_entropy_scale=0.9 \
    ++model.use_target_actor_for_sampling=false \
    ++model.grpo_include_replay_action=true \
    ++model.track_base_actor_drift=true "++model.bc_anchor_coef=${anchor}"
}

for ENV_NAME in can square; do
  if [[ "${ENV_NAME}" == can ]]; then
    CKPT=pretrained/flow_bc/can_reflow_state75.pt
    PREFIX=lfr2_can
    U1=15
  else
    CKPT=pretrained/flow_bc/square_reflow_state275.pt
    PREFIX=lfr2_sq
    U1=20
  fi
  U05=$((U1 * 2))
  U025=$((U1 * 4))
  U0125=$((U1 * 8))

  # Test whether still fewer updates preserve room for improvement.
  submit "${ENV_NAME}" "${PREFIX}_u0125" "${CKPT}" "${U0125}" 1e-5 0

  # Reduce the magnitude of the successful u0.25 schedule.
  submit "${ENV_NAME}" "${PREFIX}_u025_lr5" "${CKPT}" "${U025}" 5e-6 0

  # Weak anchors around u0.25: preserve BC without dominating RL weighting.
  submit "${ENV_NAME}" "${PREFIX}_u025_a001" "${CKPT}" "${U025}" 1e-5 0.01
  submit "${ENV_NAME}" "${PREFIX}_u025_a003" "${CKPT}" "${U025}" 1e-5 0.03
  submit "${ENV_NAME}" "${PREFIX}_u025_a01" "${CKPT}" "${U025}" 1e-5 0.1

  # Slightly faster update with a weak anchor: candidate for actual gains.
  submit "${ENV_NAME}" "${PREFIX}_u05_a003" "${CKPT}" "${U05}" 1e-5 0.03
done
