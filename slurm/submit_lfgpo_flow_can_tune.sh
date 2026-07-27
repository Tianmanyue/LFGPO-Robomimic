#!/bin/bash

set -euo pipefail

RATIO_LR=${1:-1e-4}
ACTOR_LR=${2:-1e-5}
GRPO_SAMPLES=${3:-32}
TAG=${4:-ratio1e4}
ADVANTAGE_MODE=${5:-grpo}

REPO=${LFGPO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "${REPO}"
mkdir -p slurm/logs

sbatch \
  --job-name="lf_can_${TAG}" \
  slurm/run_finetune_seed42.sbatch lfgpo_flow can \
  "name=can_lfgpo_flow_tune_${TAG}" \
  train.n_train_itr=30 \
  train.val_freq=3 \
  train.save_model_freq=3 \
  "train.ratio_lr=${RATIO_LR}" \
  "train.actor_lr=${ACTOR_LR}" \
  "model.num_grpo_samples=${GRPO_SAMPLES}" \
  "model.advantage_mode=${ADVANTAGE_MODE}"
