#!/bin/bash

# State75 Can diagnostics that restore the successful MuJoCo off-policy data path:
# UTD=1, delayed policy/target updates, current-policy noisy target/group actions,
# and GRPO normalization over the replay action plus G sampled actions.
# Usage: bash slurm/submit_lfgpo_flow_state75_faithful.sh <account> [all|n01|n03]

set -euo pipefail
ACCOUNT=${1:?Usage: $0 '<account>'}
SELECTOR=${2:-all}
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
    train.n_train_itr=151 train.n_critic_warmup_itr=0 train.val_freq=5 train.save_model_freq=10 \
    train.replay_ratio=1 +train.policy_update_freq=2 +train.target_update_freq=2 \
    train.actor_lr=1e-5 train.ratio_lr=3e-4 train.scale_reward_factor=1 \
    model.num_grpo_samples=16 model.ppo_eps=0.2 \
    model.max_ratio_weight=5 model.ratio_reg_lambda=0.01 \
    +train.alpha_lr=7e-3 +train.alpha_update_freq=250 \
    +model.adaptive_sampling_noise=true +model.noise_scale="${noise}" \
    +model.alpha_init=5.0 +model.target_entropy_scale=0.9 \
    +model.use_target_actor_for_sampling=false \
    +model.grpo_include_replay_action=true
}

case "${SELECTOR}" in
  all)
    submit lf_faith_s75_n01 0.1
    submit lf_faith_s75_n03 0.3
    ;;
  n01) submit lf_faith_s75_n01 0.1 ;;
  n03) submit lf_faith_s75_n03 0.3 ;;
  *)
    echo "Unknown selector '${SELECTOR}' (expected all, n01, or n03)" >&2
    exit 2
    ;;
esac
