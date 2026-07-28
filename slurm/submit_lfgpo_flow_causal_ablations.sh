#!/bin/bash

# Short Can/state75 causal ablations for Flow collapse. Each run is 31 itr,
# evaluates every 5 itr, and changes one identifiable part of the update loop.
# Usage: bash slurm/submit_lfgpo_flow_causal_ablations.sh <account> <0|1>

set -euo pipefail
ACCOUNT=${1:?Usage: $0 '<account>' '<0|1>'}
SHARD=${2:?Usage: $0 '<account>' '<0|1>'}
[[ "${SHARD}" == 0 || "${SHARD}" == 1 ]] || exit 2
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch
CKPT=${CAN_FLOW_CKPT:-pretrained/flow_bc/can_reflow_state75.pt}
INDEX=0
cd "${REPO}"
mkdir -p slurm/logs
[[ -f "${CKPT}" ]] || { echo "Missing ${CKPT}" >&2; exit 1; }

submit() {
  local name=$1
  shift
  local i=${INDEX}; INDEX=$((INDEX + 1))
  (( i % 2 == SHARD )) || return 0
  sbatch --account="${ACCOUNT}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=42" \
    --job-name="${name}" "${RUNNER}" lfgpo_flow can \
    "name=${name}" "base_policy_path=${CKPT}" \
    train.n_train_itr=31 train.n_critic_warmup_itr=5 \
    train.val_freq=5 train.save_model_freq=10 train.replay_ratio=1 \
    +train.policy_update_freq=2 +train.target_update_freq=2 \
    +train.alpha_lr=7e-3 +train.alpha_update_freq=250 \
    train.actor_lr=2e-6 train.ratio_lr=1e-5 train.scale_reward_factor=1 \
    model.num_grpo_samples=16 model.ppo_eps=0.1 \
    model.max_ratio_weight=1.5 model.ratio_reg_lambda=0.1 \
    +model.noise_scale=0.1 \
    +model.alpha_init=5.0 +model.target_entropy_scale=0.9 \
    +model.use_target_actor_for_sampling=false \
    +model.grpo_include_replay_action=true "$@"
}

# A: policy must remain unchanged; tests rollout/eval and critic in isolation.
submit lfa_critic_only +train.freeze_actor=true +train.freeze_ratio=true \
  +model.adaptive_sampling_noise=true

# B: train critic+ratio, freeze actor; detects ratio instability independently.
submit lfa_ratio_only +train.freeze_actor=true +train.freeze_ratio=false \
  +model.adaptive_sampling_noise=true

# C: train actor with ratio frozen at its initialized value 1 (uniform replay BC).
submit lfa_uniform_actor +train.freeze_actor=false +train.freeze_ratio=true \
  +model.adaptive_sampling_noise=true

# D/E: isolate noisy OOD actions in TD/GRPO targets.
submit lfa_noise_zero +model.adaptive_sampling_noise=false +model.sampling_noise_std=0.0
submit lfa_noise_fixed01 +model.adaptive_sampling_noise=false +model.sampling_noise_std=0.1

# F: match the archived MuJoCo numerical replay settings that are portable.
submit lfa_mj_numeric train.gamma=0.99 train.batch_size=256 train.buffer_size=1000000 \
  train.actor_lr=3e-4 train.critic_lr=3e-4 train.ratio_lr=3e-4 \
  train.scale_reward_factor=0.2 model.ppo_eps=0.2 \
  model.max_ratio_weight=5 model.ratio_reg_lambda=0.01 \
  +model.adaptive_sampling_noise=true

# G: explicitly constrain the fine-tuned velocity field to the pretrained BC.
submit lfa_bc_anchor1 +model.bc_anchor_coef=1.0 \
  +model.adaptive_sampling_noise=false +model.sampling_noise_std=0.1

# H: make each policy step much smaller while preserving update timing.
submit lfa_actor_lr5e7 train.actor_lr=5e-7 \
  +model.adaptive_sampling_noise=false +model.sampling_noise_std=0.1
