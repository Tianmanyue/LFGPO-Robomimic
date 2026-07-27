#!/bin/bash
#SBATCH --account=p32948
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --signal=B:TERM@120

set -euo pipefail

METHOD=${1:?Usage: $0 '<lfgpo_diffusion|dppo|lfgpo_flow|reinflow> <can|square|transport>'}
ENV_NAME=${2:?Usage: $0 '<method> <can|square|transport>'}
EXTRA_OVERRIDES=("${@:3}")
SEED=${LFGPO_SEED:-42}
REPO=${LFGPO_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CONDA_ROOT=${LFGPO_CONDA_ROOT:-/projects/p32948/anaconda3}
CONDA_ENV=${LFGPO_CONDA_ENV:-lfgpo_robomimic}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO}"
mkdir -p slurm/logs
source set_env.sh

case "${METHOD}" in
  lfgpo_diffusion)
    CONFIG=ft_lfgpo_diffusion_mlp
    EXTRA=()
    ;;
  dppo)
    CONFIG=ft_ppo_diffusion_mlp
    EXTRA=("_target_=agent.finetune.dppo.train_ppo_diffusion_agent.TrainPPODiffusionAgent")
    ;;
  lfgpo_flow)
    CONFIG=ft_lfgpo_flow_mlp
    EXTRA=()
    ;;
  reinflow)
    CONFIG=ft_ppo_reflow_mlp
    EXTRA=()
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

if [[ "${METHOD}" == "lfgpo_flow" || "${METHOD}" == "reinflow" ]]; then
  FLOW_CKPT="pretrained/flow_bc/${ENV_NAME}_reflow_state50.pt"
  if [[ ! -f "${FLOW_CKPT}" ]]; then
    echo "ERROR: missing flow checkpoint ${FLOW_CKPT}" >&2
    exit 1
  fi
fi

echo "Job ${SLURM_JOB_ID} on $(hostname): ${METHOD} ${ENV_NAME}, seed=${SEED}"
srun python script/run.py \
  --config-dir="cfg/robomimic/finetune/${ENV_NAME}" \
  --config-name="${CONFIG}" \
  "${EXTRA[@]}" \
  device=cuda:0 +sim_device=cuda:0 wandb=null "seed=${SEED}" \
  "${EXTRA_OVERRIDES[@]}"
