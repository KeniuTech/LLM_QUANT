"""Reinforcement learning utilities for DecisionEnv."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .adapters import DecisionEnvAdapter

TORCH_AVAILABLE = True

try:  # pragma: no cover - exercised via integration
    from .ppo import PPOConfig, PPOTrainer, train_ppo
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    if exc.name != "torch":
        raise
    TORCH_AVAILABLE = False

    @dataclass
    class PPOConfig:
        """Placeholder PPOConfig used when torch is unavailable."""

        total_timesteps: int = 0
        rollout_steps: int = 0
        gamma: float = 0.0
        gae_lambda: float = 0.0
        clip_range: float = 0.0
        policy_lr: float = 0.0
        value_lr: float = 0.0
        epochs: int = 0
        minibatch_size: int = 0
        entropy_coef: float = 0.0
        value_coef: float = 0.0
        max_grad_norm: float = 0.0
        hidden_sizes: Sequence[int] = field(default_factory=tuple)
        seed: Optional[int] = None

    class PPOTrainer:  # pragma: no cover - simply raises when used
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "torch is required for PPO training. Please install torch before using this feature."
            )

    def train_ppo(*_args, **_kwargs):  # pragma: no cover - simply raises when used
        raise ModuleNotFoundError(
            "torch is required for PPO training. Please install torch before using this feature."
        )

__all__ = [
    "DecisionEnvAdapter",
    "PPOConfig",
    "PPOTrainer",
    "train_ppo",
    "TORCH_AVAILABLE",
]
