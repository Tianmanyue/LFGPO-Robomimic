#!/bin/bash

# Submit one matched experiment on Slurm allocation p32876 with seed 100.
# Usage:
#   bash slurm/submit_p32876_seed100.sh <method> <can|square|transport> [Hydra override ...]
# Example:
#   bash slurm/submit_p32876_seed100.sh lfgpo_diffusion transport train.actor_lr=1e-5

set -euo pipefail

METHOD=${1:?Usage: $0 '<lfgpo_diffusion|dppo|lfgpo_flow|reinflow> <can|square|transport>' [Hydra override ...]}
ENV_NAME=${2:?Usage: $0 '<lfgpo_diffusion|dppo|lfgpo_flow|reinflow> <can|square|transport>' [Hydra override ...]}
EXTRA_OVERRIDES=("${@:3}")
ACCOUNT=p32876
SEED=100
REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

case "${METHOD}" in
  lfgpo_diffusion) METHOD_TAG=ld ;;
  dppo) METHOD_TAG=dp ;;
  lfgpo_flow) METHOD_TAG=lf ;;
  reinflow) METHOD_TAG=rf ;;
  *) echo "Unsupported method: ${METHOD}" >&2; exit 2 ;;
esac
case "${ENV_NAME}" in
  can|square|transport) ;;
  *) echo "Unsupported task: ${ENV_NAME}" >&2; exit 2 ;;
esac

cd "${REPO}"
mkdir -p slurm/logs

echo "Submitting ${METHOD} ${ENV_NAME}, seed=${SEED}, account=${ACCOUNT}"
sbatch \
  --account="${ACCOUNT}" \
  --export="ALL,LFGPO_REPO=${REPO},LFGPO_SEED=${SEED}" \
  --job-name="${METHOD_TAG}_${ENV_NAME}_s${SEED}" \
  slurm/run_finetune_seed42.sbatch "${METHOD}" "${ENV_NAME}" \
  "${EXTRA_OVERRIDES[@]}"
