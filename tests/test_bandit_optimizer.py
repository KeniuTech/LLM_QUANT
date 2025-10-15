"""Tests for global parameter search optimizers."""
from __future__ import annotations

import pytest

from app.backtest.decision_env import EpisodeMetrics, ParameterSpec
from app.backtest.optimizer import (
    BanditConfig,
    EpsilonGreedyBandit,
    BayesianBandit,
    SuccessiveHalvingOptimizer,
)
from app.utils import tuning


class DummyEnv:
    def __init__(self) -> None:
        self._specs = [
            ParameterSpec(name="w1", target="agent_weights.A_mom", minimum=0.0, maximum=1.0)
        ]
        self._last_metrics: EpisodeMetrics | None = None
        self._episode = 0

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def last_metrics(self) -> EpisodeMetrics | None:
        return self._last_metrics

    def reset(self) -> dict:
        self._episode += 1
        return {"episode": float(self._episode)}

    def step(self, action):
        value = float(action[0])
        reward = 1.0 - abs(value - 0.7)
        sharpe_like = reward / 0.05 if 0.05 else 0.0
        calmar_like = reward / 0.1 if 0.1 else reward
        metrics = EpisodeMetrics(
            total_return=reward,
            max_drawdown=0.1,
            volatility=0.05,
            sharpe_like=sharpe_like,
            calmar_like=calmar_like,
            nav_series=[],
            trades=[],
            turnover=0.1,
            turnover_value=1000.0,
            trade_count=0,
            risk_count=1,
            risk_breakdown={"test": 1},
        )
        self._last_metrics = metrics
        obs = {
            "total_return": reward,
            "max_drawdown": 0.1,
            "volatility": 0.05,
            "sharpe_like": reward / 0.05,
            "calmar_like": reward / 0.1,
            "turnover": 0.1,
            "turnover_value": 1000.0,
            "trade_count": 0.0,
            "risk_count": 1.0,
        }
        info = {
            "nav_series": [],
            "trades": [],
            "weights": {"A_mom": value},
            "risk_breakdown": metrics.risk_breakdown,
            "risk_events": [],
            "department_controls": {"momentum": {"prompt": "baseline"}},
        }
        return obs, reward, True, info


@pytest.fixture(autouse=True)
def patch_logging(monkeypatch):
    records = []

    def fake_log_tuning_result(**kwargs):
        records.append(kwargs)

    monkeypatch.setattr(tuning, "log_tuning_result", fake_log_tuning_result)
    from app.backtest import optimizer as optimizer_module

    monkeypatch.setattr(optimizer_module, "log_tuning_result", fake_log_tuning_result)
    return records


def test_epsilon_greedy_optimizer(patch_logging):
    env = DummyEnv()
    optimizer = EpsilonGreedyBandit(
        env,
        BanditConfig(experiment_id="exp_eps", episodes=5, epsilon=0.5, seed=42),
    )
    summary = optimizer.run()

    assert len(summary.episodes) == 5
    assert summary.best_episode is not None
    assert patch_logging and len(patch_logging) == 5
    payload = patch_logging[0]["metrics"]
    assert isinstance(payload, dict)
    assert "risk_breakdown" in payload
    assert summary.best_episode.department_controls == {"momentum": {"prompt": "baseline"}}


def test_bayesian_optimizer(patch_logging):
    env = DummyEnv()
    optimizer = BayesianBandit(
        env,
        BanditConfig(
            experiment_id="exp_bayes",
            strategy="bayesian",
            episodes=6,
            candidate_pool=32,
            exploration_weight=0.01,
            seed=123,
        ),
    )
    summary = optimizer.run()
    assert summary.best_episode is not None
    assert summary.best_episode.reward > 0.3
    assert len(patch_logging) >= 6


def test_successive_halving_optimizer(patch_logging):
    env = DummyEnv()
    optimizer = SuccessiveHalvingOptimizer(
        env,
        BanditConfig(
            experiment_id="exp_bohb",
            strategy="bohb",
            initial_candidates=9,
            eta=3,
            max_rounds=2,
            seed=7,
        ),
    )
    summary = optimizer.run()
    assert summary.best_episode is not None
    assert summary.best_episode.reward > 0.3
    assert len(patch_logging) >= 9
