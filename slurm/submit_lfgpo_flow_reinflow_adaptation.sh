#!/bin/bash

# Can/Square diagnostics for adapting the unchanged MuJoCo LFGPO update to
# Robomimic.  The jobs only change sampling, macro-action time scale, trust
# region, or replay distribution; the ratio and weighted-flow objectives stay
# untouched.
#
# Usage: bash slurm/submit_lfgpo_flow_reinflow_adaptation.sh ACCOUNT SHARD
# SHARD is 0 or 1 and allows the 24 jobs to be split across two allocations.

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
  local env_name=$1 name=$2 ckpt=$3
  shift 3
  local i=${INDEX}; INDEX=$((INDEX + 1))
  (( i % 2 == SHARD )) || return 0
  sbatch --account="${ACCOUNT}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=42" \
    --job-name="${name}" "${RUNNER}" lfgpo_flow "${env_name}" \
    "name=${name}" "base_policy_path=${ckpt}" \
    train.n_train_itr=61 train.n_critic_warmup_itr=20 \
    train.val_freq=3 train.save_model_freq=20 train.replay_ratio=1 \
    train.batch_size=1000 train.buffer_size=5000 \
    train.actor_lr=1e-5 train.critic_lr=3e-4 train.ratio_lr=1e-4 \
    train.scale_reward_factor=1 ++train.use_lr_scheduler=false \
    ++train.policy_update_freq=2 ++train.target_update_freq=2 \
    ++train.alpha_lr=7e-3 ++train.alpha_update_freq=250 \
    model.num_grpo_samples=16 model.ppo_eps=0.1 \
    model.max_ratio_weight=2 model.ratio_reg_lambda=0.05 \
    ++model.adaptive_sampling_noise=true ++model.noise_scale=0.1 \
    ++model.alpha_init=5.0 ++model.target_entropy_scale=0.9 \
    ++model.noise_on_critic_target=true \
    ++model.noise_on_advantage_samples=true \
    ++model.entropy_include_horizon=false \
    ++model.use_target_actor_for_sampling=false \
    ++model.grpo_include_replay_action=true "$@"
}

for ENV_NAME in can square; do
  if [[ "${ENV_NAME}" == can ]]; then
    CKPT=pretrained/flow_bc/can_reflow_state75.pt
    PREFIX=lfr_can
  else
    CKPT=pretrained/flow_bc/square_reflow_state275.pt
    PREFIX=lfr_sq
  fi
  [[ -f "${CKPT}" ]] || { echo "Missing ${CKPT}" >&2; exit 1; }

  # D0: instrumented reference. No algorithm or adapter change.
  submit "${ENV_NAME}" "${PREFIX}_d0_ref" "${CKPT}"

  # D1/D2: match Robomimic ReinFlow's conservative action noise (~0.1).
  submit "${ENV_NAME}" "${PREFIX}_d1_fix01" "${CKPT}" \
    ++model.adaptive_sampling_noise=false ++model.sampling_noise_std=0.1
  submit "${ENV_NAME}" "${PREFIX}_d2_alpha1" "${CKPT}" \
    ++model.alpha_init=1.0

  # D3/D4/D5: locate whether OOD noise enters through TD or GRPO sampling.
  submit "${ENV_NAME}" "${PREFIX}_d3_notdnoise" "${CKPT}" \
    ++model.noise_on_critic_target=false
  submit "${ENV_NAME}" "${PREFIX}_d4_nogrpnoise" "${CKPT}" \
    ++model.noise_on_advantage_samples=false
  submit "${ENV_NAME}" "${PREFIX}_d5_nobothnoise" "${CKPT}" \
    ++model.noise_on_critic_target=false ++model.noise_on_advantage_samples=false

  # D6: treat the chunk as the macro-action in the adaptive entropy dimension.
  submit "${ENV_NAME}" "${PREFIX}_d6_chunkent" "${CKPT}" \
    ++model.alpha_init=1.0 ++model.entropy_include_horizon=true

  # D7: original Q(s,a)-Q(s,a') LFGPO advantage, keeping the same objective.
  submit "${ENV_NAME}" "${PREFIX}_d7_ppoadv" "${CKPT}" \
    ++model.advantage_mode=ppo ++model.alpha_init=1.0

  # D8: sparse-reward replay adapter; retain positive transitions in minibatches.
  submit "${ENV_NAME}" "${PREFIX}_d8_pos25" "${CKPT}" \
    ++train.positive_replay_fraction=0.25 ++model.alpha_init=1.0

  # D9: ReinFlow-like conservative trust region and a two-iteration replay window.
  submit "${ENV_NAME}" "${PREFIX}_d9_rftrust" "${CKPT}" \
    train.buffer_size=600 model.ppo_eps=0.01 model.max_ratio_weight=1.5 \
    model.ratio_reg_lambda=0.05 train.ratio_lr=3e-5 \
    ++model.alpha_init=1.0 ++model.entropy_include_horizon=true

  # D10: semi-MDP bootstrap discount for four executed primitive actions.
  submit "${ENV_NAME}" "${PREFIX}_d10_gammak" "${CKPT}" \
    train.gamma=0.996005996001 ++model.alpha_init=1.0

  # D11: reduce actor feedback frequency while leaving critic UTD unchanged.
  submit "${ENV_NAME}" "${PREFIX}_d11_pfreq8" "${CKPT}" \
    ++train.policy_update_freq=8 ++model.alpha_init=1.0
done
