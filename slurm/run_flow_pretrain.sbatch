#!/bin/bash
#SBATCH --account=p32948
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

ENV_NAME=${1:?Usage: $0 '<can|square|transport>'}
REPO=${LFGPO_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CONDA_ROOT=${LFGPO_CONDA_ROOT:-/projects/p32948/anaconda3}
CONDA_ENV=${LFGPO_CONDA_ENV:-lfgpo_robomimic}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO}"
mkdir -p slurm/logs
source set_env.sh

echo "Job ${SLURM_JOB_ID} on $(hostname): flow BC pretrain ${ENV_NAME}"
srun python script/run.py \
  --config-dir="cfg/robomimic/pretrain/${ENV_NAME}" \
  --config-name=pre_reflow_mlp \
  device=cuda:0 wandb=null

CKPT=$(find "${REINFLOW_LOG_DIR}/robomimic/pretrain" -path "*${ENV_NAME}_pre_reflow_mlp*/checkpoint/state_50.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "ERROR: state_50.pt was not produced for ${ENV_NAME}" >&2
  exit 1
fi

mkdir -p pretrained/flow_bc
ln -sfn "${CKPT}" "pretrained/flow_bc/${ENV_NAME}_reflow_state50.pt"
echo "Flow BC checkpoint: ${CKPT}"
