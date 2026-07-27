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

ENV_NAME=${1:?Usage: $0 '<can|square|transport>'}
RESUME_CKPT=${2:-pretrained/flow_bc/${ENV_NAME}_reflow_state50.pt}
EXTRA_EPOCHS=${3:-150}
REPO=${LFGPO_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CONDA_ROOT=${LFGPO_CONDA_ROOT:-/projects/p32948/anaconda3}
CONDA_ENV=${LFGPO_CONDA_ENV:-lfgpo_robomimic}

if [[ "${ENV_NAME}" != "can" && "${ENV_NAME}" != "square" && "${ENV_NAME}" != "transport" ]]; then
  echo "Only can, square, and transport are supported by this tuning job" >&2
  exit 2
fi
if [[ ! -f "${REPO}/${RESUME_CKPT}" ]]; then
  echo "Missing resume checkpoint: ${REPO}/${RESUME_CKPT}" >&2
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO}"
mkdir -p slurm/logs
source set_env.sh

echo "Job ${SLURM_JOB_ID}: resume ${ENV_NAME} Flow BC from ${RESUME_CKPT} for ${EXTRA_EPOCHS} epochs"
srun python script/run.py \
  --config-dir="cfg/robomimic/pretrain/${ENV_NAME}" \
  --config-name=pre_reflow_mlp \
  "name=${ENV_NAME}_pre_reflow_mlp_long200" \
  "resume_path=${RESUME_CKPT}" \
  "train.n_epochs=${EXTRA_EPOCHS}" \
  train.save_model_freq=25 \
  device=cuda:0 wandb=null
