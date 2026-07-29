#!/bin/bash
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# Train a compatible BC from scratch, evaluate it, then run a short faithful
# LFGPO-Flow test. Modes: arch256 (MuJoCo-sized Mish MLP) or horizon1.

set -euo pipefail
MODE=${1:?Usage: $0 '<arch256|horizon1>'}
REPO=${LFGPO_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CONDA_ROOT=${LFGPO_CONDA_ROOT:-/projects/p32948/anaconda3}
CONDA_ENV=${LFGPO_CONDA_ENV:-lfgpo_robomimic}
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO}"
mkdir -p slurm/logs
source set_env.sh

case "${MODE}" in
  arch256)
    PRE=(model.network.mlp_dims='[256,256,256]' model.network.activation_type=Mish model.network.residual_style=false)
    FT=(model.network.mlp_dims='[256,256,256]' model.network.activation_type=Mish model.network.residual_style=false)
    ;;
  horizon1)
    PRE=(horizon_steps=1)
    FT=(horizon_steps=1 act_steps=1)
    ;;
  *) echo "Unknown mode ${MODE}" >&2; exit 2 ;;
esac

MARKER=$(mktemp "${TMPDIR:-/tmp}/flow-struct-${MODE}.XXXXXX")
srun python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can --config-name=pre_reflow_mlp \
  "name=can_reflow_${MODE}_bc150" train.n_epochs=151 train.save_model_freq=150 \
  device=cuda:0 wandb=null "${PRE[@]}"

CKPT=$(find runtime/log/robomimic/pretrain -path '*/checkpoint/state_150.pt' \
  -newer "${MARKER}" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
rm -f "${MARKER}"
[[ -n "${CKPT}" && -f "${CKPT}" ]] || { echo "No state150 produced for ${MODE}" >&2; exit 1; }

echo "Structural ablation ${MODE} checkpoint: ${CKPT}"
srun python script/run.py \
  --config-dir=cfg/robomimic/finetune/can --config-name=ft_lfgpo_flow_mlp \
  "name=lf_struct_${MODE}" "base_policy_path=${CKPT}" \
  train.n_train_itr=31 train.n_critic_warmup_itr=20 train.val_freq=5 \
  train.save_model_freq=10 train.replay_ratio=1 \
  +train.policy_update_freq=2 +train.target_update_freq=2 \
  +train.alpha_lr=7e-3 +train.alpha_update_freq=250 \
  train.actor_lr=1e-5 train.critic_lr=3e-4 train.ratio_lr=3e-4 \
  train.scale_reward_factor=1 model.num_grpo_samples=16 \
  model.ppo_eps=0.2 model.max_ratio_weight=5 model.ratio_reg_lambda=0.01 \
  +model.adaptive_sampling_noise=true +model.noise_scale=0.1 \
  +model.alpha_init=5.0 +model.target_entropy_scale=0.9 \
  +model.use_target_actor_for_sampling=false \
  +model.grpo_include_replay_action=true \
  +model.ratio_net.activation_type=Mish +model.ratio_net.logit_clip=7.5 \
  device=cuda:0 +sim_device=cuda:0 wandb=null seed=42 "${FT[@]}"
