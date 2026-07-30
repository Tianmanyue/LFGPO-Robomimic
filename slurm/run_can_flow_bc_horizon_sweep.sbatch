#!/bin/bash
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# Train a default-architecture Can Flow BC at one action horizon, then evaluate
# every 25-epoch checkpoint with the matching horizon. This produces comparable
# base-policy curves before any online LFGPO update is applied.

set -euo pipefail
HORIZON=${1:?Usage: $0 '<1|2|4>'}
[[ "${HORIZON}" == 1 || "${HORIZON}" == 2 || "${HORIZON}" == 4 ]] || exit 2
REPO=${LFGPO_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CONDA_ROOT=${LFGPO_CONDA_ROOT:-/projects/p32948/anaconda3}
CONDA_ENV=${LFGPO_CONDA_ENV:-lfgpo_robomimic}
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO}"
mkdir -p slurm/logs
source set_env.sh

MARKER=$(mktemp "${TMPDIR:-/tmp}/can-bc-h${HORIZON}.XXXXXX")
srun python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can --config-name=pre_reflow_mlp \
  "name=can_reflow_h${HORIZON}_matched_bc150" \
  "horizon_steps=${HORIZON}" train.n_epochs=151 train.save_model_freq=25 \
  device=cuda:0 wandb=null

RUN_DIR=$(find "runtime/log/robomimic/pretrain/can_pre_reflow_mlp_ta${HORIZON}_td100" \
  -path '*/checkpoint/state_150.pt' -newer "${MARKER}" -type f \
  -printf '%T@ %h\n' | sort -n | tail -1 | cut -d' ' -f2- | xargs dirname)
rm -f "${MARKER}"
[[ -n "${RUN_DIR}" && -f "${RUN_DIR}/checkpoint/state_150.pt" ]] || exit 1

for EPOCH in 25 50 75 100 125 150; do
  CKPT="${RUN_DIR}/checkpoint/state_${EPOCH}.pt"
  [[ -f "${CKPT}" ]] || { echo "Missing ${CKPT}" >&2; exit 1; }
  echo "MATCHED_BC_EVAL horizon=${HORIZON} epoch=${EPOCH} checkpoint=${CKPT}"
  srun python script/run.py \
    --config-dir=cfg/robomimic/finetune/can --config-name=ft_lfgpo_flow_mlp \
    "name=can_flowbc_h${HORIZON}_e${EPOCH}_eval" \
    "base_policy_path=${CKPT}" "horizon_steps=${HORIZON}" "act_steps=${HORIZON}" \
    train.n_train_itr=1 train.n_critic_warmup_itr=0 train.val_freq=1 \
    train.save_model_freq=999 device=cuda:0 +sim_device=cuda:0 wandb=null seed=42
done

echo "Completed matched Can BC sweep for horizon=${HORIZON}"
