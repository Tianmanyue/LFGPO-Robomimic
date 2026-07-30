#!/bin/bash

# Phase 10: winner replication plus Transport actor-collapse localization.
# Intended for the external Diffusion account; do not submit on Flow accounts.
# Usage: bash slurm/submit_lfgpo_diffusion_phase10.sh <slurm-account>

set -euo pipefail
ACCOUNT=${1:?Usage: $0 '<slurm-account>'}
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch
cd "${REPO}"
mkdir -p slurm/logs

submit() {
  local name=$1 env_name=$2 seed=$3
  shift 3
  sbatch --account="${ACCOUNT}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=${seed}" \
    --job-name="${name}" "${RUNNER}" lfgpo_diffusion "${env_name}" \
    "name=${name}" train.val_freq=5 train.save_model_freq=10 "$@"
}

# Exact winner replication. Can's exact P8 peak setting was not the same as the
# earlier rs3 multiseed setting; Square's new 99.0% winner has no multiseed yet.
for SEED in 100 200; do
  submit "ld_p10_can_best_s${SEED}" can "${SEED}" train.n_train_itr=151 \
    train.actor_lr=1.25e-5 train.ratio_lr=7.5e-5 train.scale_reward_factor=3 \
    model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.035
  submit "ld_p10_sq_best_s${SEED}" square "${SEED}" train.n_train_itr=201 \
    train.actor_lr=2e-5 train.ratio_lr=1e-4 train.scale_reward_factor=2.5 \
    model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.03
done

# Transport uses one common reference setting so module ablations are causal.
# val_freq=5 exposes the first post-warmup collapse rather than only itr10 bins.
COMMON=(train.n_train_itr=51 train.n_critic_warmup_itr=10 \
  train.actor_lr=1.25e-5 train.ratio_lr=7.5e-5 train.scale_reward_factor=3 \
  model.ppo_eps=0.2 model.max_ratio_weight=2 model.ratio_reg_lambda=0.035)

submit ld_p10_tr_critic_only transport 42 "${COMMON[@]}" \
  +train.freeze_actor=true +train.freeze_ratio=true
submit ld_p10_tr_ratio_only transport 42 "${COMMON[@]}" \
  +train.freeze_actor=true +train.freeze_ratio=false
submit ld_p10_tr_uniform_actor transport 42 "${COMMON[@]}" \
  +train.freeze_actor=false +train.freeze_ratio=true
submit ld_p10_tr_full_ref transport 42 "${COMMON[@]}"

# One-factor actor-stability repairs.
submit ld_p10_tr_actor1e6 transport 42 "${COMMON[@]}" train.actor_lr=1e-6
submit ld_p10_tr_policyfreq8 transport 42 "${COMMON[@]}" \
  +train.policy_update_freq=8 +train.target_update_freq=8
submit ld_p10_tr_anchor1 transport 42 "${COMMON[@]}" +model.bc_anchor_coef=1.0
submit ld_p10_tr_anchor5 transport 42 "${COMMON[@]}" +model.bc_anchor_coef=5.0

echo "Submitted Phase 10: 4 winner replications + 8 Transport causal runs"
