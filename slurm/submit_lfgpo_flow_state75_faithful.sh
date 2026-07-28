#!/bin/bash

# State75 Can diagnostics that restore the successful MuJoCo off-policy data path:
# UTD=1, delayed policy/target updates, current-policy noisy target/group actions,
# and GRPO normalization over the replay action plus G sampled actions.
# Usage: bash slurm/submit_lfgpo_flow_state75_faithful.sh <account>

set -euo pipefail
ACCOUNT=${1:?Usage: $0 '<account>'}
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNNER=slurm/run_finetune_seed42.sbatch
CKPT=${CAN_FLOW_CKPT:-pretrained/flow_bc/can_reflow_state75.pt}
cd "${REPO}"
mkdir -p slurm/logs
[[ -f "${CKPT}" ]] || { echo "Missing ${CKPT}" >&2; exit 1; }

submit() {
  local name=$1 noise=$2
  sbatch --account="${ACCOUNT}" --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=42" \
    --job-name="${name}" "${RUNNER}" lfgpo_flow can \
    "name=${name}" "base_policy_path=${CKPT}" \
    train.n_train_itr=151 train.val_freq=5 train.save_model_freq=10 \
    train.replay_ratio=1 +train.policy_update_freq=2 +train.target_update_freq=2 \
    train.actor_lr=1e-5 train.ratio_lr=3e-4 train.scale_reward_factor=1 \
    model.num_grpo_samples=16 model.ppo_eps=0.2 \
    model.max_ratio_weight=5 model.ratio_reg_lambda=0.01 \
    +model.sampling_noise_std="${noise}" \
    +model.use_target_actor_for_sampling=false \
    +model.grpo_include_replay_action=true
}

submit lf_faith_s75_n01 0.1
submit lf_faith_s75_n03 0.3
