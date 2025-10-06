from __future__ import annotations

import math

from app.rl.adapters import DecisionEnvAdapter
from app.rl.ppo import PPOConfig, train_ppo


class _DummyDecisionEnv:
    action_dim = 1

    def __init__(self) -> None:
        self._step = 0
        self._episode = 0

    def reset(self):
        self._step = 0
        self._episode += 1
        return {
            "day_index": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "turnover": 0.0,
            "sharpe_like": 0.0,
            "trade_count": 0.0,
            "risk_count": 0.0,
            "nav": 1.0,
            "cash": 1.0,
            "market_value": 0.0,
            "done": 0.0,
        }

    def step(self, action):
        value = float(action[0])
        reward = 1.0 - abs(value - 0.8)
        self._step += 1
        done = self._step >= 3
        obs = {
            "day_index": float(self._step),
            "total_return": reward,
            "max_drawdown": 0.1,
            "volatility": 0.05,
            "turnover": 0.1,
            "sharpe_like": reward / 0.05,
            "trade_count": float(self._step),
            "risk_count": 0.0,
            "nav": 1.0 + 0.01 * self._step,
            "cash": 1.0,
            "market_value": 0.0,
            "done": 1.0 if done else 0.0,
        }
        info = {}
        return obs, reward, done, info


def test_train_ppo_completes_with_dummy_env():
    adapter = DecisionEnvAdapter(_DummyDecisionEnv())
    config = PPOConfig(
        total_timesteps=64,
        rollout_steps=16,
        epochs=2,
        minibatch_size=8,
        hidden_sizes=(32, 32),
        seed=123,
    )
    summary = train_ppo(adapter, config)

    assert summary.timesteps == config.total_timesteps
    assert summary.episode_rewards
    assert not math.isnan(summary.episode_rewards[-1])
    assert summary.diagnostics
