"""
LFGPO-Flow-GRPO (off-policy) fine-tuning agent for robomimic.

The off-policy LFGPO training loop is identical for diffusion and flow policies -- it
only touches the model through a shared interface (forward / loss_critic /
compute_advantage / loss_ratio / loss_actor / update_target_critic /
update_target_policy, and .actor / .critic_q / .ratio_net). So this agent simply
reuses TrainLFGPODiffusionAgent; the flow-specific behaviour (velocity matching,
flow ODE sampling, GRPO group advantage) lives in model/flow/ft_lfgpo/lfgpo_flow.py.
"""

from agent.finetune.lfgpo.train_lfgpo_diffusion_agent import TrainLFGPODiffusionAgent


class TrainLFGPOFlowAgent(TrainLFGPODiffusionAgent):
    pass
