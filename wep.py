import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

from .model import Policy_flow, QNetwork, ScoreNetwork
from .utils import hard_update, soft_update


@contextmanager
def frozen(module):
    states = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, state in zip(module.parameters(), states):
            p.requires_grad_(state)


class WEP(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.action_space = action_space
        self.sample_count = 0
        self.target_update_interval = args.target_update_interval
        self.device = torch.device(f"cuda:{args.device}" if args.cuda else "cpu")
        self.amp_enabled = args.cuda and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16
        self.scaler = GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        self.action_dim = action_space.shape[0]
        self.action_low = torch.as_tensor(action_space.low, device=self.device, dtype=torch.float32)
        self.action_high = torch.as_tensor(action_space.high, device=self.device, dtype=torch.float32)
        self.transport_steps = max(1, int(getattr(args, "transport_steps", getattr(args, "steps", 1))))
        self.refinement_steps = max(0, int(getattr(args, "refinement_steps", getattr(args, "lang_steps", 5))))
        self.refinement_step_size = float(getattr(args, "wep_step_size", getattr(args, "lan_lr", 5e-3)))
        self.temperature_max = float(getattr(args, "temperature_max", getattr(args, "tau_max", 1.0)))
        self.lambda_q = float(getattr(args, "lambda_q", 1.0))
        self.grad_clip_norm = float(getattr(args, "grad_clip_norm", 10.0))
        self.num_action_candidates = max(1, int(getattr(args, "num_action_candidates", 1)))

        if getattr(args, "policy", "Flow") != "Flow":
            raise ValueError("WEP requires Flow policy")

        critic_lr = float(getattr(args, "critic_lr", args.lr))
        actor_lr = float(getattr(args, "actor_lr", args.lr))
        score_lr = float(getattr(args, "score_lr", args.lr))

        self.critic = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.policy = Policy_flow(num_inputs, self.action_dim, args.hidden_size, self.transport_steps, action_space).to(self.device)
        self.policy_target = Policy_flow(num_inputs, self.action_dim, args.hidden_size, self.transport_steps, action_space).to(self.device)
        hard_update(self.policy_target, self.policy)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=actor_lr)

        self.score = ScoreNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.score_target = ScoreNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        hard_update(self.score_target, self.score)
        self.score_optim = optim.Adam(self.score.parameters(), lr=score_lr)

    def _clamp_action(self, action):
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def _base_action(self, batch_size):
        action = torch.randn(batch_size, self.action_dim, device=self.device)
        return self._clamp_action(action)

    def _critic_value(self, critic, state, action):
        q1, q2 = critic(state, action)
        return torch.min(q1, q2)

    def _clip_vector(self, vector):
        norm = vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (self.grad_clip_norm / norm).clamp(max=1.0)
        return vector * scale

    def _transport_rollout(self, state, action, policy=None):
        policy = self.policy if policy is None else policy
        time_start = torch.zeros(action.shape[0], 1, device=self.device, dtype=action.dtype)
        time_step = 1.0 / self.transport_steps
        for _ in range(self.transport_steps):
            time_end = time_start + time_step
            action = policy.step(state, action, time_start, time_end)
            action = self._clamp_action(action)
            time_start = time_end
        return action

    def _refinement_rollout(self, state, action, stochastic=True, score_model=None):
        if self.refinement_steps == 0:
            return self._clamp_action(action)
        score_model = self.score if score_model is None else score_model
        sigma = action.new_tensor(2.0 * self.refinement_step_size * self.temperature_max).sqrt()
        for _ in range(self.refinement_steps):
            score = self._clip_vector(score_model(state, action))
            noise = torch.randn_like(action) * sigma if stochastic else torch.zeros_like(action)
            action = action + self.refinement_step_size * score + noise
            action = self._clamp_action(action)
        return action

    def _generate_actions(self, state, n_samples=1, stochastic=True, policy=None, score_model=None, value_model=None):
        batch_size = state.shape[0]
        value_model = self.critic_target if value_model is None else value_model
        expanded_state = state.unsqueeze(1).expand(-1, n_samples, -1).reshape(batch_size * n_samples, -1)
        action = self._base_action(batch_size * n_samples)
        action = self._transport_rollout(expanded_state, action, policy=policy)
        action = self._refinement_rollout(expanded_state, action, stochastic=stochastic, score_model=score_model)
        if n_samples == 1:
            return action, torch.zeros(batch_size, 1, device=self.device), action
        with torch.no_grad():
            q_value = self._critic_value(value_model, expanded_state, action).view(batch_size, n_samples)
            best_idx = torch.argmax(q_value, dim=1, keepdim=True)
            action_candidates = action.view(batch_size, n_samples, self.action_dim)
            gather_idx = best_idx.unsqueeze(-1).expand(-1, -1, self.action_dim)
            best_action = torch.gather(action_candidates, dim=1, index=gather_idx).squeeze(1)
            max_q = torch.gather(q_value, dim=1, index=best_idx)
        return best_action, max_q, action

    def sample_with_wep(self, state, n_samples=1, stochastic=True, target=False):
        if target:
            return self._generate_actions(
                state,
                n_samples=n_samples,
                stochastic=stochastic,
                policy=self.policy_target,
                score_model=self.score_target,
                value_model=self.critic_target,
            )
        return self._generate_actions(state, n_samples=n_samples, stochastic=stochastic)

    def sample_with_BFN(self, state, n_samples=1):
        return self.sample_with_wep(state, n_samples=n_samples, stochastic=True)

    def select_action(self, state, evaluate=False):
        self.sample_count += int(not evaluate)
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.sample_with_wep(
                state,
                n_samples=self.num_action_candidates,
                stochastic=not evaluate,
            )
        return action.detach().cpu().numpy()[0].clip(self.action_space.low, self.action_space.high)

    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        with torch.no_grad():
            next_action, _, _ = self.sample_with_wep(next_state_batch, n_samples=1, stochastic=True, target=True)
            next_q = self._critic_value(self.critic_target, next_state_batch, next_action)
            target_q = reward_batch + mask_batch * self.gamma * next_q

        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            q1, q2 = self.critic(state_batch, action_batch)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optim.zero_grad(set_to_none=True)
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optim)
        self.scaler.update()
        return critic_loss.detach()

    def _top_action_targets(self, state_batch, action_batch):
        with torch.no_grad():
            generated_action, _, _ = self.sample_with_wep(state_batch, n_samples=1, stochastic=True)
            candidates = torch.stack((action_batch, generated_action), dim=1)
            flat_state = state_batch.unsqueeze(1).expand(-1, 2, -1).reshape(-1, state_batch.shape[-1])
            flat_action = candidates.reshape(-1, self.action_dim)
            values = self._critic_value(self.critic, flat_state, flat_action).view(state_batch.shape[0], 2)
            best_idx = torch.argmax(values, dim=1, keepdim=True)
            gather_idx = best_idx.unsqueeze(-1).expand(-1, -1, self.action_dim)
            return torch.gather(candidates, dim=1, index=gather_idx).squeeze(1)

    def update_policy(self, state_batch, action_batch):
        base_action = self._base_action(action_batch.shape[0])
        target_action = self._top_action_targets(state_batch, action_batch)
        t = torch.rand(action_batch.shape[0], 1, device=self.device)
        path_action = (1.0 - t) * base_action + t * target_action
        target_velocity = target_action - base_action

        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            predicted_velocity = self.policy(state_batch, path_action, t)
            geo_loss = F.mse_loss(predicted_velocity, target_velocity)
            with frozen(self.critic):
                transported_action = self._transport_rollout(state_batch, base_action)
                task_value = self._critic_value(self.critic, state_batch, transported_action)
                policy_loss = geo_loss - self.lambda_q * task_value.mean()

        self.policy_optim.zero_grad(set_to_none=True)
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.policy_optim)
        self.scaler.update()
        return policy_loss.detach()

    def _direct_value_gradient(self, state, action):
        action = action.detach().requires_grad_(True)
        q_value = self._critic_value(self.critic, state, action)
        grad = torch.autograd.grad(q_value.sum(), action, retain_graph=False, create_graph=False)[0]
        return grad.detach()

    def update_score(self, state_batch, action_batch):
        with torch.no_grad():
            generated_action, _, _ = self.sample_with_wep(state_batch, n_samples=1, stochastic=True)
        state = torch.cat((state_batch, state_batch), dim=0)
        action = torch.cat((action_batch, generated_action), dim=0)
        target_grad = self._direct_value_gradient(state, action)

        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            predicted_grad = self.score(state, action.detach())
            score_loss = F.mse_loss(predicted_grad, target_grad)

        self.score_optim.zero_grad(set_to_none=True)
        self.scaler.scale(score_loss).backward()
        self.scaler.step(self.score_optim)
        self.scaler.update()
        return score_loss.detach()

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.as_tensor(state_batch, device=self.device, dtype=torch.float32)
        next_state_batch = torch.as_tensor(next_state_batch, device=self.device, dtype=torch.float32)
        action_batch = torch.as_tensor(action_batch, device=self.device, dtype=torch.float32)
        reward_batch = torch.as_tensor(reward_batch, device=self.device, dtype=torch.float32).unsqueeze(1)
        mask_batch = torch.as_tensor(mask_batch, device=self.device, dtype=torch.float32).unsqueeze(1)
        action_batch = self._clamp_action(action_batch)

        self.update_policy(state_batch, action_batch)
        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        self.update_score(state_batch, action_batch)

        if updates % self.target_update_interval == 0:
            with torch.no_grad():
                soft_update(self.critic_target, self.critic, self.tau)
                soft_update(self.policy_target, self.policy, self.tau)
                soft_update(self.score_target, self.score, self.tau)

    def save_checkpoint(self, path, i_episode):
        ckpt_path = os.path.join(path, f"{i_episode}.torch")
        print(f"Saving models to {ckpt_path}")
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "policy_target_state_dict": self.policy_target.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "score_state_dict": self.score.state_dict(),
                "score_target_state_dict": self.score_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
                "score_optimizer_state_dict": self.score_optim.state_dict(),
            },
            ckpt_path,
        )

    def load_checkpoint(self, path, i_episode, evaluate=False):
        ckpt_path = os.path.join(path, "checkpoint", "best.torch")
        if i_episode is not None:
            candidate_path = os.path.join(path, f"{i_episode}.torch")
            if os.path.exists(candidate_path):
                ckpt_path = candidate_path
        print(f"Loading models from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "policy_target_state_dict" in checkpoint:
            self.policy_target.load_state_dict(checkpoint["policy_target_state_dict"])
        else:
            hard_update(self.policy_target, self.policy)
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        if "score_state_dict" in checkpoint:
            self.score.load_state_dict(checkpoint["score_state_dict"])
        if "score_target_state_dict" in checkpoint:
            self.score_target.load_state_dict(checkpoint["score_target_state_dict"])
        else:
            hard_update(self.score_target, self.score)
        if "score_optimizer_state_dict" in checkpoint:
            self.score_optim.load_state_dict(checkpoint["score_optimizer_state_dict"])

        modules = (self.policy, self.policy_target, self.critic, self.critic_target, self.score, self.score_target)
        for module in modules:
            module.eval() if evaluate else module.train()

