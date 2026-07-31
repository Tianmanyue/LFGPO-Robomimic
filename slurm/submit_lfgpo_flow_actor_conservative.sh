#!/bin/bash

# Conservative actor-timescale sweep for Can/Square.  Keeps the LFGPO
# objective unchanged and targets approximately 1, 0.5, or 0.25 actor updates
# per training iteration, matching ReinFlow's iteration-level update control.
# Usage: bash slurm/submit_lfgpo_flow_actor_conservative.sh ACCOUNT SHARD

set -euo pipefail
ACCOUNT=${1:?Usage: $0 ACCOUNT '<0|1>'}
SHARD=${2:?Usage: $0 ACCOUNT '<0|1>'}
[[ "${SHARD}" == 0 || "${SHARD}" == 1 ]] || exit 2
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch
INDEX=0
cd "${REPO}"
mkdir -p slurm/logs

submit() {
  local env_name=$1 name=$2 ckpt=$3 pfreq=$4 actor_lr=$5 anchor=$6
  local i=${INDEX}; INDEX=$((INDEX + 1))
  (( i % 2 == SHARD )) || return 0
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
    PREFIX=lfc_can
    U1=15
  else
    CKPT=pretrained/flow_bc/square_reflow_state275.pt
    PREFIX=lfc_sq
    U1=20
  fi
  U05=$((U1 * 2))
  U025=$((U1 * 4))

  # Pure timescale sweep at the existing actor LR.
  submit "${ENV_NAME}" "${PREFIX}_u1" "${CKPT}" "${U1}" 1e-5 0
  submit "${ENV_NAME}" "${PREFIX}_u05" "${CKPT}" "${U05}" 1e-5 0
  submit "${ENV_NAME}" "${PREFIX}_u025" "${CKPT}" "${U025}" 1e-5 0

  # Separate update frequency from update magnitude.
  submit "${ENV_NAME}" "${PREFIX}_u1_lr5" "${CKPT}" "${U1}" 5e-6 0
  submit "${ENV_NAME}" "${PREFIX}_u05_lr5" "${CKPT}" "${U05}" 5e-6 0

  # Direct policy-space conservatism around the pretrained velocity field.
  submit "${ENV_NAME}" "${PREFIX}_u1_a01" "${CKPT}" "${U1}" 1e-5 0.1
  submit "${ENV_NAME}" "${PREFIX}_u1_a1" "${CKPT}" "${U1}" 1e-5 1.0
  submit "${ENV_NAME}" "${PREFIX}_u05_a01" "${CKPT}" "${U05}" 1e-5 0.1
done
