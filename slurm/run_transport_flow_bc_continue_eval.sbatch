#!/bin/bash
#SBATCH --account=p32948
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

REPO=${LFGPO_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CONDA_ROOT=${LFGPO_CONDA_ROOT:-/projects/p32948/anaconda3}
CONDA_ENV=${LFGPO_CONDA_ENV:-lfgpo_robomimic}
CURRENT_CKPT=${1:-runtime/log/robomimic/pretrain/transport_pre_reflow_mlp_ta8_td100/2026-07-26_18-39-07_42/checkpoint/state_150.pt}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO}"
mkdir -p slurm/logs
source set_env.sh

[[ -f "${CURRENT_CKPT}" ]] || { echo "Missing Transport state150 checkpoint: ${CURRENT_CKPT}" >&2; exit 1; }
echo "Continuing Transport Flow BC from ${CURRENT_CKPT}"

for TARGET_EPOCH in 175 200 225 250 275 300 325 350 375 400 425 450 475 500; do
  MARKER=$(mktemp "${TMPDIR:-/tmp}/transport-bc-${TARGET_EPOCH}.XXXXXX")
  srun python script/run.py \
    --config-dir=cfg/robomimic/pretrain/transport \
    --config-name=pre_reflow_mlp \
    "name=transport_pre_reflow_continue_e${TARGET_EPOCH}" \
    "+base_policy_path=${CURRENT_CKPT}" \
    train.n_epochs=26 train.save_model_freq=25 \
    device=cuda:0 wandb=null

  NEXT_CKPT=$(find runtime/log/robomimic/pretrain/transport_pre_reflow_mlp_ta8_td100 \
    -path "*/checkpoint/state_${TARGET_EPOCH}.pt" -newer "${MARKER}" -type f \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  rm -f "${MARKER}"
  [[ -n "${NEXT_CKPT}" && -f "${NEXT_CKPT}" ]] || {
    echo "Failed to produce state_${TARGET_EPOCH}.pt" >&2; exit 1;
  }
  CURRENT_CKPT=${NEXT_CKPT}

  echo "Evaluating Transport cumulative epoch ${TARGET_EPOCH}: ${CURRENT_CKPT}"
  srun python script/run.py \
    --config-dir=cfg/robomimic/finetune/transport \
    --config-name=ft_lfgpo_flow_mlp \
    "name=transport_flowbc_epoch${TARGET_EPOCH}_eval" \
    "base_policy_path=${CURRENT_CKPT}" \
    train.n_train_itr=1 train.n_critic_warmup_itr=0 \
    train.val_freq=1 train.save_model_freq=1 \
    device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
done

echo "Completed Transport Flow BC continuation/evaluation through epoch 500"
