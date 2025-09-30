"""Tests for DecisionEnv risk-aware reward and info outputs."""
from __future__ import annotations

from datetime import date

import pytest

from app.backtest.decision_env import DecisionEnv, EpisodeMetrics, ParameterSpec
from app.backtest.engine import BacktestResult, BtConfig


class _StubEngine:
    def __init__(self, cfg: BtConfig) -> None:  # noqa: D401
        self.cfg = cfg
        self.weights = {}
        self.department_manager = None

    def run(self) -> BacktestResult:
        result = BacktestResult()
        result.nav_series = [
            {
                "trade_date": "2025-01-10",
                "nav": 102.0,
                "cash": 50.0,
                "market_value": 52.0,
                "realized_pnl": 1.0,
                "unrealized_pnl": 1.0,
                "turnover": 20000.0,
            }
        ]
        result.trades = [
            {
                "trade_date": "2025-01-10",
                "ts_code": "000001.SZ",
                "action": "buy",
                "quantity": 100.0,
                "price": 100.0,
                "value": 10000.0,
                "fee": 5.0,
            }
        ]
        result.risk_events = [
            {
                "trade_date": "2025-01-10",
                "ts_code": "000002.SZ",
                "reason": "limit_up",
                "action": "buy_l",
                "confidence": 0.7,
                "target_weight": 0.2,
            }
        ]
        return result


def test_decision_env_returns_risk_metrics(monkeypatch):
    cfg = BtConfig(
        id="stub",
        name="stub",
        start_date=date(2025, 1, 10),
        end_date=date(2025, 1, 10),
        universe=["000001.SZ"],
        params={},
    )
    specs = [ParameterSpec(name="w_mom", target="agent_weights.A_mom", minimum=0.0, maximum=1.0)]
    env = DecisionEnv(bt_config=cfg, parameter_specs=specs, baseline_weights={"A_mom": 0.5})

    monkeypatch.setattr("app.backtest.decision_env.BacktestEngine", _StubEngine)

    obs, reward, done, info = env.step([0.8])

    assert done is True
    assert "risk_count" in obs and obs["risk_count"] == 1.0
    assert obs["turnover"] == pytest.approx(20000.0)
    assert info["risk_events"][0]["reason"] == "limit_up"
    assert info["risk_breakdown"]["limit_up"] == 1
    assert reward < obs["total_return"]


def test_default_reward_penalizes_metrics():
    metrics = EpisodeMetrics(
        total_return=0.1,
        max_drawdown=0.2,
        volatility=0.05,
        nav_series=[],
        trades=[],
        turnover=1000.0,
        trade_count=0,
        risk_count=2,
        risk_breakdown={"foo": 2},
    )
    reward = DecisionEnv._default_reward(metrics)
    assert reward == pytest.approx(0.1 - (0.5 * 0.2 + 0.05 * 2 + 0.00001 * 1000.0))
