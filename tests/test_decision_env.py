"""Tests for DecisionEnv risk-aware reward and info outputs."""
from __future__ import annotations

from datetime import date

import pytest

from app.backtest.decision_env import DecisionEnv, EpisodeMetrics, ParameterSpec
from app.backtest.engine import BacktestResult, BtConfig
from app.utils.config import DepartmentSettings, LLMConfig, LLMEndpoint


class _StubDepartmentAgent:
    def __init__(self) -> None:
        self._tool_choice = "auto"
        self._max_rounds = 3
        endpoint = LLMEndpoint(provider="openai", model="mock", temperature=0.2)
        self.settings = DepartmentSettings(
            code="momentum",
            title="Momentum",
            description="baseline",
            prompt="baseline",
            llm=LLMConfig(primary=endpoint),
        )

    @property
    def tool_choice(self) -> str:
        return self._tool_choice

    @tool_choice.setter
    def tool_choice(self, value) -> None:
        normalized = str(value).strip().lower()
        if normalized not in {"auto", "none", "required"}:
            raise ValueError("invalid tool choice")
        self._tool_choice = normalized

    @property
    def max_rounds(self) -> int:
        return self._max_rounds

    @max_rounds.setter
    def max_rounds(self, value) -> None:
        numeric = int(round(float(value)))
        if numeric < 1:
            numeric = 1
        if numeric > 6:
            numeric = 6
        self._max_rounds = numeric


class _StubManager:
    def __init__(self) -> None:
        self.agents = {"momentum": _StubDepartmentAgent()}


class _StubEngine:
    def __init__(self, cfg: BtConfig) -> None:  # noqa: D401
        self.cfg = cfg
        self.weights = {}
        self.department_manager = _StubManager()
        _StubEngine.last_instance = self

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
                "turnover_ratio": 0.2,
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


_StubEngine.last_instance: _StubEngine | None = None


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
    monkeypatch.setattr(DecisionEnv, "_clear_portfolio_records", lambda self: None)
    monkeypatch.setattr(DecisionEnv, "_fetch_portfolio_records", lambda self: ([], []))

    obs, reward, done, info = env.step([0.8])

    assert done is True
    assert "risk_count" in obs and obs["risk_count"] == 1.0
    assert obs["turnover"] == pytest.approx(0.2)
    assert obs["turnover_value"] == pytest.approx(20000.0)
    assert info["risk_events"][0]["reason"] == "limit_up"
    assert info["risk_breakdown"]["limit_up"] == 1
    assert info["nav_series"][0]["turnover_ratio"] == pytest.approx(0.2)
    assert reward < obs["total_return"]


def test_default_reward_penalizes_metrics():
    metrics = EpisodeMetrics(
        total_return=0.1,
        max_drawdown=0.2,
        volatility=0.05,
        nav_series=[],
        trades=[],
        turnover=0.3,
        turnover_value=5000.0,
        trade_count=0,
        risk_count=2,
        risk_breakdown={"foo": 2},
    )
    reward = DecisionEnv._default_reward(metrics)
    assert reward == pytest.approx(0.1 - (0.5 * 0.2 + 0.05 * 2 + 0.1 * 0.3))


def test_decision_env_department_controls(monkeypatch):
    cfg = BtConfig(
        id="stub",
        name="stub",
        start_date=date(2025, 1, 10),
        end_date=date(2025, 1, 10),
        universe=["000001.SZ"],
        params={},
    )
    specs = [
        ParameterSpec(name="w_mom", target="agent_weights.A_mom", minimum=0.0, maximum=1.0),
        ParameterSpec(
            name="dept_prompt",
            target="department.momentum.prompt",
            values=["baseline", "aggressive"],
        ),
        ParameterSpec(
            name="dept_temp",
            target="department.momentum.temperature",
            minimum=0.1,
            maximum=0.9,
        ),
        ParameterSpec(
            name="dept_tool",
            target="department.momentum.function_policy",
            values=["none", "auto", "required"],
        ),
        ParameterSpec(
            name="dept_rounds",
            target="department.momentum.max_rounds",
            minimum=1,
            maximum=5,
        ),
    ]

    env = DecisionEnv(bt_config=cfg, parameter_specs=specs, baseline_weights={"A_mom": 0.5})

    monkeypatch.setattr("app.backtest.decision_env.BacktestEngine", _StubEngine)
    monkeypatch.setattr(DecisionEnv, "_clear_portfolio_records", lambda self: None)
    monkeypatch.setattr(DecisionEnv, "_fetch_portfolio_records", lambda self: ([], []))

    obs, reward, done, info = env.step([0.3, 1.0, 0.75, 0.0, 1.0])

    assert done is True
    assert obs["total_return"] == pytest.approx(0.0)

    controls = info["department_controls"]
    assert "momentum" in controls
    momentum_ctrl = controls["momentum"]
    assert momentum_ctrl["prompt"] == "aggressive"
    assert momentum_ctrl["temperature"] == pytest.approx(0.7, abs=1e-6)
    assert momentum_ctrl["tool_choice"] == "none"
    assert momentum_ctrl["max_rounds"] == 5

    assert env.last_department_controls == controls

    engine = _StubEngine.last_instance
    assert engine is not None
    agent = engine.department_manager.agents["momentum"]
    assert agent.settings.prompt == "aggressive"
    assert agent.settings.llm.primary.temperature == pytest.approx(0.7, abs=1e-6)
    assert agent.tool_choice == "none"
    assert agent.max_rounds == 5
