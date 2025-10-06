"""Lightweight PPO trainer tailored for DecisionEnv."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta

from .adapters import DecisionEnvAdapter


def _init_layer(layer: nn.Module, std: float = 1.0) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=std)
        nn.init.zeros_(layer.bias)
    return layer


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = obs_dim
        for size in hidden_sizes:
            layers.append(_init_layer(nn.Linear(last_dim, size), std=math.sqrt(2)))
            layers.append(nn.Tanh())
            last_dim = size
        self.body = nn.Sequential(*layers)
        self.alpha_head = _init_layer(nn.Linear(last_dim, action_dim), std=0.01)
        self.beta_head = _init_layer(nn.Linear(last_dim, action_dim), std=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(obs)
        alpha = torch.nn.functional.softplus(self.alpha_head(hidden)) + 1.0
        beta = torch.nn.functional.softplus(self.beta_head(hidden)) + 1.0
        return alpha, beta


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = obs_dim
        for size in hidden_sizes:
            layers.append(_init_layer(nn.Linear(last_dim, size), std=math.sqrt(2)))
            layers.append(nn.Tanh())
            last_dim = size
        layers.append(_init_layer(nn.Linear(last_dim, 1), std=1.0))
        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs).squeeze(-1)


@dataclass
class PPOConfig:
    total_timesteps: int = 4096
    rollout_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    epochs: int = 8
    minibatch_size: int = 128
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: Sequence[int] = (128, 128)
    device: str = "cpu"
    seed: Optional[int] = None


@dataclass
class TrainingSummary:
    timesteps: int
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    diagnostics: List[Dict[str, float]] = field(default_factory=list)


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.size = size
        self.device = device
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)
        self._pos = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ) -> None:
        if self._pos >= self.size:
            raise RuntimeError("rollout buffer overflow; check rollout_steps")
        self.obs[self._pos].copy_(obs)
        self.actions[self._pos].copy_(action)
        self.log_probs[self._pos] = log_prob
        self.rewards[self._pos] = reward
        self.dones[self._pos] = float(done)
        self.values[self._pos] = value
        self._pos += 1

    def finish(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        last_advantage = 0.0
        for idx in reversed(range(self._pos)):
            mask = 1.0 - float(self.dones[idx])
            value = float(self.values[idx])
            delta = float(self.rewards[idx]) + gamma * last_value * mask - value
            last_advantage = delta + gamma * gae_lambda * mask * last_advantage
            self.advantages[idx] = last_advantage
            self.returns[idx] = last_advantage + value
            last_value = value

        if self._pos:
            adv_view = self.advantages[: self._pos]
            adv_mean = adv_view.mean()
            adv_std = adv_view.std(unbiased=False) + 1e-8
            adv_view.sub_(adv_mean).div_(adv_std)

    def get_minibatches(self, batch_size: int) -> Iterable[Tuple[torch.Tensor, ...]]:
        if self._pos == 0:
            return []
        indices = torch.randperm(self._pos, device=self.device)
        for start in range(0, self._pos, batch_size):
            end = min(start + batch_size, self._pos)
            batch_idx = indices[start:end]
            yield (
                self.obs[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.advantages[batch_idx],
                self.returns[batch_idx],
                self.values[batch_idx],
            )

    def reset(self) -> None:
        self._pos = 0


class PPOTrainer:
    def __init__(self, adapter: DecisionEnvAdapter, config: PPOConfig) -> None:
        self.adapter = adapter
        self.config = config
        device = torch.device(config.device)
        obs_dim = adapter.observation_dim
        action_dim = adapter.action_dim
        self.actor = ActorNetwork(obs_dim, action_dim, config.hidden_sizes).to(device)
        self.critic = CriticNetwork(obs_dim, config.hidden_sizes).to(device)
        self.policy_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.value_lr)
        self.device = device
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

    def train(self) -> TrainingSummary:
        cfg = self.config
        obs_array, _ = self.adapter.reset()
        obs = torch.from_numpy(obs_array).to(self.device)
        rollout = RolloutBuffer(cfg.rollout_steps, self.adapter.observation_dim, self.adapter.action_dim, self.device)
        timesteps = 0
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        diagnostics: List[Dict[str, float]] = []
        current_return = 0.0
        current_length = 0

        while timesteps < cfg.total_timesteps:
            rollout.reset()
            steps_to_collect = min(cfg.rollout_steps, cfg.total_timesteps - timesteps)
            for _ in range(steps_to_collect):
                with torch.no_grad():
                    alpha, beta = self.actor(obs.unsqueeze(0))
                    dist = Beta(alpha, beta)
                    action = dist.rsample().squeeze(0)
                    log_prob = dist.log_prob(action).sum()
                    value = self.critic(obs.unsqueeze(0)).squeeze(0)
                action_np = action.cpu().numpy()
                next_obs_array, reward, done, info, _ = self.adapter.step(action_np)
                next_obs = torch.from_numpy(next_obs_array).to(self.device)

                rollout.add(obs, action, log_prob, reward, done, value)
                timesteps += 1
                current_return += reward
                current_length += 1

                if done:
                    episode_rewards.append(current_return)
                    episode_lengths.append(current_length)
                    current_return = 0.0
                    current_length = 0
                    next_obs_array, _ = self.adapter.reset()
                    next_obs = torch.from_numpy(next_obs_array).to(self.device)

                obs = next_obs

                if timesteps >= cfg.total_timesteps or rollout._pos >= steps_to_collect:
                    break

            with torch.no_grad():
                next_value = self.critic(obs.unsqueeze(0)).squeeze(0).item()
            rollout.finish(last_value=next_value, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

            for _ in range(cfg.epochs):
                for (mb_obs, mb_actions, mb_log_probs, mb_adv, mb_returns, _) in rollout.get_minibatches(
                    cfg.minibatch_size
                ):
                    alpha, beta = self.actor(mb_obs)
                    dist = Beta(alpha, beta)
                    new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)
                    ratios = torch.exp(new_log_probs - mb_log_probs)
                    surrogate1 = ratios * mb_adv
                    surrogate2 = torch.clamp(ratios, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_adv
                    policy_loss = -torch.min(surrogate1, surrogate2).mean() - cfg.entropy_coef * entropy.mean()

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                    self.policy_optimizer.step()

                    values = self.critic(mb_obs)
                    value_loss = torch.nn.functional.mse_loss(values, mb_returns)
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                    self.value_optimizer.step()

                    diagnostics.append(
                        {
                            "policy_loss": float(policy_loss.detach().cpu()),
                            "value_loss": float(value_loss.detach().cpu()),
                            "entropy": float(entropy.mean().detach().cpu()),
                        }
                    )

        return TrainingSummary(
            timesteps=timesteps,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            diagnostics=diagnostics,
        )


def train_ppo(adapter: DecisionEnvAdapter, config: PPOConfig) -> TrainingSummary:
    """Convenience helper to run PPO training with defaults."""

    trainer = PPOTrainer(adapter, config)
    return trainer.train()
