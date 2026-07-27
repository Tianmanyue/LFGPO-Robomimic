#!/bin/bash

set -euo pipefail

MODE=${1:?Usage: $0 '<pretrain|diffusion|flow>' [seed ...]}
SEEDS=("${@:2}")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(42)
fi
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO}"
mkdir -p slurm/logs pretrained/flow_bc

ENVS=(can square transport)

case "${MODE}" in
  pretrain)
    for env_name in "${ENVS[@]}"; do
      sbatch --job-name="bc_${env_name}" slurm/run_flow_pretrain.sbatch "${env_name}"
    done
    ;;
  diffusion)
    for seed in "${SEEDS[@]}"; do
      for env_name in "${ENVS[@]}"; do
        sbatch --export="ALL,LFGPO_SEED=${seed}" --job-name="ld_${env_name}_s${seed}" slurm/run_finetune_seed42.sbatch lfgpo_diffusion "${env_name}"
        sbatch --export="ALL,LFGPO_SEED=${seed}" --job-name="dp_${env_name}_s${seed}" slurm/run_finetune_seed42.sbatch dppo "${env_name}"
      done
    done
    ;;
  flow)
    for seed in "${SEEDS[@]}"; do
      for env_name in "${ENVS[@]}"; do
        ckpt="pretrained/flow_bc/${env_name}_reflow_state50.pt"
        if [[ ! -f "${ckpt}" ]]; then
          echo "Refusing to submit flow: missing ${ckpt}" >&2
          exit 1
        fi
        sbatch --export="ALL,LFGPO_SEED=${seed}" --job-name="lf_${env_name}_s${seed}" slurm/run_finetune_seed42.sbatch lfgpo_flow "${env_name}"
        sbatch --export="ALL,LFGPO_SEED=${seed}" --job-name="rf_${env_name}_s${seed}" slurm/run_finetune_seed42.sbatch reinflow "${env_name}"
      done
    done
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
