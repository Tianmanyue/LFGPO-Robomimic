#!/bin/bash

# Additional Can/state75 causal tests for replay, optimizer/loss numerics, and
# a combined portable MuJoCo alignment. All run through itr 50.
# Usage: bash slurm/submit_lfgpo_flow_causal_ablations_b.sh <account> <0|1>

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

submit() {
  local name=$1; shift
  local i=${INDEX}; INDEX=$((INDEX + 1))
  (( i % 2 == SHARD )) || return 0
  sbatch --account="${ACCOUNT}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=42" \
    --job-name="${name}" "${RUNNER}" lfgpo_flow can \
    "name=${name}" "base_policy_path=${CKPT}" \
    train.n_train_itr=51 train.n_critic_warmup_itr=20 \
    train.val_freq=5 train.save_model_freq=10 train.replay_ratio=1 \
    +train.policy_update_freq=2 +train.target_update_freq=2 \
    +train.alpha_lr=7e-3 +train.alpha_update_freq=250 \
    train.actor_lr=1e-5 train.critic_lr=3e-4 train.ratio_lr=3e-4 \
    train.scale_reward_factor=1 model.num_grpo_samples=16 \
    model.ppo_eps=0.2 model.max_ratio_weight=5 model.ratio_reg_lambda=0.01 \
    +model.adaptive_sampling_noise=true +model.noise_scale=0.1 \
    +model.alpha_init=5.0 +model.target_entropy_scale=0.9 \
    +model.use_target_actor_for_sampling=false \
    +model.grpo_include_replay_action=true "$@"
}

# Preserve rare successful terminal chunks in each replay minibatch.
submit lfb_posreplay25 +train.positive_replay_fraction=0.25

# Match JAX optimizer family and remove the per-iteration cosine scheduler.
submit lfb_adam_const +train.optimizer_type=adam +train.use_lr_scheduler=false

# Optax squared_error includes a 0.5 factor relative to PyTorch MSE.
submit lfb_actorloss_half +model.actor_loss_scale=0.5

# Combine all portable numeric/code alignments in one diagnostic.
submit lfb_mj_combined train.gamma=0.99 train.batch_size=256 train.buffer_size=1000000 \
  train.scale_reward_factor=0.2 +train.optimizer_type=adam \
  +train.use_lr_scheduler=false +model.actor_loss_scale=0.5 \
  +model.ratio_net.activation_type=Mish +model.ratio_net.logit_clip=7.5
