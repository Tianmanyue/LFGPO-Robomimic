"""
LFGPO-Flow-GRPO (off-policy): Likelihood-Free Generative Policy Optimization for
rectified-flow policies via the Doob h-transform with a learned PPO-clipped ratio net
and a GRPO-style group-relative advantage.

PyTorch port of relax/algorithm/lfgpo_flow_grpo.py (JAX). Structure mirrors the
off-policy flow SAC agent (replay buffer + twin-Q critic) but:
  - advantage is group-relative (GRPO): sample G actions a_1..a_G ~ pi(.|s),
      A(s,a) = (Q(s,a) - mean(group_Q)) / (std(group_Q) + eps)
  - ratio net r_beta = exp(MLP), PPO-clip surrogate + unbiased double-sample regularizer
  - policy update = ratio-reweighted WEIGHTED VELOCITY MATCHING (flow-matching MSE loss
      from ReFlow.generate_target, weighted per-sample by r_beta) -- the flow analog of
      the diffusion drift matching in model/diffusion/diffusion_lfgpo.py.

reward_scale: lfgpo MuJoCo used ~0.2-0.5; robomimic sparse reward -> RETUNE.
"""

import copy
import logging

import torch
import torch.nn.functional as F

from model.flow.reflow import ReFlow
from model.diffusion.diffusion_lfgpo import RatioNet  # reuse exp(MLP) ratio net

log = logging.getLogger(__name__)


class LFGPOFlow(ReFlow):

    def __init__(
        self,
        network,          # FlowMLP (velocity field)
        critic,           # CriticObsAct (twin Q)
        ratio_net,        # RatioNet
        device,
        horizon_steps,
        action_dim,
        act_min,
        act_max,
        obs_dim,
        max_denoising_steps,
        seed,
        inference_steps,
        network_path=None,
        ppo_eps=0.2,
        max_ratio_weight=5.0,
        ratio_reg_lambda=0.01,
        num_grpo_samples=32,
        grpo_batch_norm_adv=False,
        advantage_mode="grpo",
        adv_norm=True,
        adv_eps=1e-4,
        use_target_policy=True,
        sample_t_type="uniform",
        **kwargs,
    ):
        super().__init__(
            network=network, device=device, horizon_steps=horizon_steps,
            action_dim=action_dim, act_min=act_min, act_max=act_max, obs_dim=obs_dim,
            max_denoising_steps=max_denoising_steps, seed=seed, sample_t_type=sample_t_type,
        )
        # load flow BC checkpoint into self.network (ReinFlow format: data["model"], "network." prefix)
        if network_path:
            data = torch.load(network_path, map_location=device, weights_only=True)
            net_sd = {k.replace("network.", ""): v for k, v in data["model"].items()}
            self.network.load_state_dict(net_sd)
            log.info(f"Loaded flow BC policy from {network_path}")

        self.critic_q = critic.to(device)
        self.target_q = copy.deepcopy(critic).to(device)
        self.ratio_net = ratio_net.to(device)
        self.actor = self.network

        self.inference_steps = inference_steps
        self.ppo_eps = ppo_eps
        self.max_ratio_weight = max_ratio_weight
        self.ratio_reg_lambda = ratio_reg_lambda
        self.num_grpo_samples = num_grpo_samples
        self.grpo_batch_norm_adv = grpo_batch_norm_adv
        if advantage_mode not in {"grpo", "ppo"}:
            raise ValueError(f"Unknown advantage_mode={advantage_mode!r}")
        self.advantage_mode = advantage_mode
        self.adv_norm = adv_norm
        self.adv_eps = adv_eps

        self.use_target_policy = use_target_policy
        if use_target_policy:
            self.target_actor = copy.deepcopy(self.network).to(device)

    # ------------------------------------------------------------------ #
    # rollout interface: self.model(cond=..., deterministic=...) -> actions
    # (matches the diffusion/SAC agents so the LFGPO agent loop is shared)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, cond, deterministic=False):
        # deterministic is accepted for API parity; flow sampling integrates the
        # ODE from x0~N(0,I). Low-variance for a trained flow; eval uses the same path.
        return self.sample(cond=cond, inference_steps=self.inference_steps).trajectories

    # ------------------------------------------------------------------ #
    # sampling (flow ODE integration) from online / target policy
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_action(self, cond, use_target=False):
        net = self.target_actor if (use_target and self.use_target_policy) else self.network
        orig = self.network
        self.network = net
        try:
            s = self.sample(cond=cond, inference_steps=self.inference_steps)
        finally:
            self.network = orig
        return s.trajectories  # (B, horizon, act)

    # ------------------------------------------------------------------ #
    # 1. critic: twin-Q TD (a' ~ flow policy)
    # ------------------------------------------------------------------ #
    def loss_critic(self, obs, next_obs, actions, rewards, terminated, gamma):
        current_q1, current_q2 = self.critic_q(obs, actions)
        next_actions = self.sample_action(next_obs, use_target=True)
        with torch.no_grad():
            next_q1, next_q2 = self.target_q(next_obs, next_actions)
        next_q = torch.min(next_q1, next_q2)
        mask = 1 - terminated
        target_q = rewards.view(-1) + gamma * next_q.view(-1) * mask.view(-1)
        return torch.mean((current_q1.view(-1) - target_q) ** 2) + torch.mean(
            (current_q2.view(-1) - target_q) ** 2
        )

    # ------------------------------------------------------------------ #
    # 2. GRPO group-relative advantage
    #    A(s,a) = (Q(s,a) - mean(group_Q)) / (std(group_Q) + eps),  group ~ pi
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_advantage(self, obs, actions):
        q1, q2 = self.target_q(obs, actions)
        q_sa = torch.min(q1, q2).view(-1)  # (B,)

        if self.advantage_mode == "ppo":
            # Match LFGPO-Diffusion: estimate V(s) with one independent
            # target-policy action, then optionally normalize the minibatch.
            v_action = self.sample_action(obs, use_target=True)
            vq1, vq2 = self.target_q(obs, v_action)
            adv = q_sa - torch.min(vq1, vq2).view(-1)
            if self.adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + self.adv_eps)
            return adv

        B = actions.shape[0]
        G = self.num_grpo_samples
        state = obs["state"]  # (B, cond_steps, obs_dim)
        rep = {"state": state.repeat_interleave(G, dim=0)}  # (B*G, ...)
        a_g = self.sample_action(rep, use_target=True)       # (B*G, horizon, act)
        gq1, gq2 = self.target_q(rep, a_g)
        group_q = torch.min(gq1, gq2).view(B, G)             # (B, G)

        mean = group_q.mean(dim=1)
        if self.grpo_batch_norm_adv:
            adv = q_sa - mean
            adv = (adv - adv.mean()) / (adv.std() + self.adv_eps)
        else:
            std = group_q.std(dim=1)
            adv = (q_sa - mean) / (std + self.adv_eps)
        return adv

    # ------------------------------------------------------------------ #
    # 3. ratio net: PPO-clip surrogate + unbiased double-sample regularizer
    # ------------------------------------------------------------------ #
    def loss_ratio(self, obs, actions, adv_norm):
        adv_sg = adv_norm.detach()
        r_beta = self.ratio_net(obs, actions)
        r_clip = torch.clamp(r_beta, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
        ppo_obj = torch.minimum(r_beta * adv_sg, r_clip * adv_sg)

        action_prime = self.sample_action(obs, use_target=True)
        r_beta_prime = self.ratio_net(obs, action_prime)
        g_hat_1 = r_beta.mean() - 1.0
        g_hat_2 = r_beta_prime.mean() - 1.0
        regularizer = self.ratio_reg_lambda * g_hat_1 * g_hat_2

        return -ppo_obj.mean() + regularizer, r_beta.detach().mean()

    # ------------------------------------------------------------------ #
    # 4. policy: ratio-reweighted WEIGHTED VELOCITY MATCHING
    # ------------------------------------------------------------------ #
    def loss_actor(self, obs, actions):
        with torch.no_grad():
            r_raw = self.ratio_net(obs, actions)
            r_norm = r_raw / (r_raw.mean() + 1e-8)
            weight = torch.clamp(r_norm, 0.0, self.max_ratio_weight)  # (B,)
        (xt, t), v = self.generate_target(actions)   # flow-matching target v = x1 - x0
        v_hat = self.network(xt, t, obs)
        per_sample = F.mse_loss(v_hat, v, reduction="none").mean(dim=list(range(1, v.dim())))  # (B,)
        return (weight * per_sample).mean()

    # ------------------------------------------------------------------ #
    # 5. Polyak target updates
    # ------------------------------------------------------------------ #
    def update_target_critic(self, tau):
        for tp, sp in zip(self.target_q.parameters(), self.critic_q.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)

    def update_target_policy(self, tau):
        if not self.use_target_policy:
            return
        for tp, sp in zip(self.target_actor.parameters(), self.network.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)
