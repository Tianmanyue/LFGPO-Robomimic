# LFGPO-Robomimic

PyTorch code for the **Robomimic** manipulation experiments in the LFGPO paper
(*Likelihood-Free Generative Policy Optimization*). LFGPO is an RL fine-tuning framework for
pretrained **diffusion** and **flow** policies: it learns a lightweight PPO/GRPO-clipped ratio
network and distills its policy improvement into the generative policy via ratio-reweighted
drift / velocity matching — **without evaluating exact action likelihoods**.

This repo is a fork of [ReinFlow](https://github.com/ReinFlow/ReinFlow) (which builds on
[DPPO](https://github.com/irom-lab/dppo)); we add the LFGPO agents, models, and configs.
It reproduces the state-based Robomimic comparison on **Can / Square / Transport**:

|                  | Diffusion policy                | Flow (rectified-flow) policy |
| ---------------- | ------------------------------- | ---------------------------- |
| **Our method**   | **LFGPO-Diffusion** (PPO ratio) | **LFGPO-Flow** (GRPO ratio)  |
| **Baseline**     | DPPO                            | ReinFlow                     |

All experiments are **state-based** (low-dim observations). Metric: task **success rate** vs
environment steps.

---

## 1. Installation

```bash
conda create --name lfgpo_robomimic python=3.8 -y && conda activate lfgpo_robomimic
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .                       # this repo
pip install "robosuite==1.4.1" "robomimic==0.3.0" "mujoco==3.1.6" hydra-core gdown beautifulsoup4
python $(python -c "import robosuite,os;print(os.path.dirname(robosuite.__file__))")/scripts/setup_macros.py
```

> ⚠️ **`mujoco==3.1.6` is required** (the version DPPO pins alongside `robosuite==1.4.1`). With
> newer mujoco (e.g. 3.2.3) the Robomimic contact physics differ and RL fails to learn to *hold*
> the object → success rate stays at 0. If `import mujoco_py` fails, add
> `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia`.

Set environment variables (checkpoint/data/log dirs, rendering libs) once per shell:
```bash
source set_env.sh      # edit CKPT_ROOT and WANDB_ENTITY inside for your machine
```

---

## 2. Reproducing the experiments

Common flags: `device=cuda:0 +sim_device=cuda:0` (the `+sim_device` enables EGL offscreen
rendering; omit it to fall back to osmesa, ~3× slower). Add `wandb=null` to disable logging, or set
`DPPO_WANDB_ENTITY`. Use `seed=42` (43, 44 for additional seeds). Replace `can` with `square` /
`transport` throughout.

### 2.1 Diffusion (LFGPO-Diffusion vs DPPO)
Pretrained BC checkpoints + normalization **download automatically** on first run.
```bash
# Ours — LFGPO-Diffusion
python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_lfgpo_diffusion_mlp \
    device=cuda:0 +sim_device=cuda:0 seed=42
# Baseline — DPPO
python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_diffusion_mlp \
    _target_=agent.finetune.dppo.train_ppo_diffusion_agent.TrainPPODiffusionAgent \
    device=cuda:0 +sim_device=cuda:0 seed=42
```

### 2.2 Flow (LFGPO-Flow vs ReinFlow)
Flow needs a state BC checkpoint — **pretrain it first** (rectified-flow BC, ~50 epochs, fast):
```bash
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_reflow_mlp device=cuda:0
```
This writes `.../<env>_pre_reflow_mlp_ta*_td100/<TIMESTAMP>_42/checkpoint/state_50.pt`. Set that
path as `base_policy_path` in `cfg/robomimic/finetune/can/{ft_lfgpo_flow_mlp,ft_ppo_reflow_mlp}.yaml`
(replace the `PRETRAINED_42` placeholder). Then:
```bash
# Ours — LFGPO-Flow (GRPO)
python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_lfgpo_flow_mlp \
    device=cuda:0 +sim_device=cuda:0 seed=42
# Baseline — ReinFlow
python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_reflow_mlp \
    device=cuda:0 +sim_device=cuda:0 seed=42
```

### 2.3 Results
Each run periodically evaluates (deterministic policy) and logs **`success rate`** to wandb (if an
entity is set) and to a local `.pkl` under its `logdir`. Plot success-rate vs env-step per task,
overlaying the four methods. Seed variability means exact numbers differ across runs; the training
*procedure* here is fully reproducible.

---

## 3. What we add on top of ReinFlow

- `model/diffusion/diffusion_lfgpo.py` — `RatioNet` + `LFGPODiffusion` (off-policy: twin-Q critic,
  PPO-clip ratio net, ratio-reweighted score/drift matching).
- `model/flow/ft_lfgpo/lfgpo_flow.py` — `LFGPOFlow` (off-policy: twin-Q, **GRPO** group-relative
  advantage, ratio-reweighted velocity matching).
- `agent/finetune/lfgpo/` — the LFGPO training agents.
- `cfg/robomimic/{pretrain,finetune}/{can,square,transport}/…` — LFGPO configs + our state flow
  pretrain / baseline configs.

## Acknowledgements
Built on **[ReinFlow](https://github.com/ReinFlow/ReinFlow)**, **[DPPO](https://github.com/irom-lab/dppo)**,
and **[Robomimic](https://github.com/ARISE-Initiative/robomimic)** — thanks to their authors.
