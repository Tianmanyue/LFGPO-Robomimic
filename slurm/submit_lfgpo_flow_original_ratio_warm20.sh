#!/bin/bash

# Single Can/state75 validation: faithful replay path, original default LFGPO
# hyperparameters, MuJoCo RatioNet (Mish, clip 7.5), critic-only warmup 20.
# Usage: bash slurm/submit_lfgpo_flow_original_ratio_warm20.sh <account>

set -euo pipefail
ACCOUNT=${1:?Usage: $0 '<account>'}
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch
CKPT=${CAN_FLOW_CKPT:-pretrained/flow_bc/can_reflow_state75.pt}
cd "${REPO}"
mkdir -p slurm/logs
[[ -f "${CKPT}" ]] || { echo "Missing ${CKPT}" >&2; exit 1; }

sbatch --account="${ACCOUNT}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=42" \
  --job-name=lf_mjratio_w20 "${RUNNER}" lfgpo_flow can \
  name=lf_mjratio_w20 "base_policy_path=${CKPT}" \
  train.n_train_itr=51 train.n_critic_warmup_itr=20 \
  train.val_freq=5 train.save_model_freq=10 train.replay_ratio=1 \
  +train.policy_update_freq=2 +train.target_update_freq=2 \
  +train.alpha_lr=7e-3 +train.alpha_update_freq=250 \
  train.actor_lr=1e-5 train.critic_lr=3e-4 train.ratio_lr=3e-4 \
  train.scale_reward_factor=1 \
  model.num_grpo_samples=16 model.ppo_eps=0.2 \
  model.max_ratio_weight=5 model.ratio_reg_lambda=0.01 \
  +model.adaptive_sampling_noise=true +model.noise_scale=0.1 \
  +model.alpha_init=5.0 +model.target_entropy_scale=0.9 \
  +model.use_target_actor_for_sampling=false \
  +model.grpo_include_replay_action=true \
  +model.ratio_net.activation_type=Mish \
  +model.ratio_net.logit_clip=7.5
