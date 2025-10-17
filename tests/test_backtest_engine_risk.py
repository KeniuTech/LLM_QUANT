"""Tests for BacktestEngine risk-aware execution."""
from __future__ import annotations

from datetime import date

import pytest

import json

from app.agents.base import AgentAction, AgentContext
from app.agents.game import Decision, RiskAssessment
from app.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    BtConfig,
    PortfolioState,
    _persist_backtest_results,
)
from app.data.schema import initialize_database
from app.utils.config import DataPaths, get_config
from app.utils import alerts
from app.utils.db import db_session


def _make_context(price: float, features: dict | None = None) -> AgentContext:
    scope_values = {"daily.close": price}
    return AgentContext(
        ts_code="000001.SZ",
        trade_date="2025-01-10",
        features=features or {},
        market_snapshot={},
        raw={"scope_values": scope_values},
    )


def _make_decision(action: AgentAction, target_weight: float = 0.0) -> Decision:
    return Decision(
        action=action,
        confidence=0.8,
        target_weight=target_weight,
        feasible_actions=[],
        utilities={},
    )


def _engine_with_params(params: dict[str, float]) -> BacktestEngine:
    cfg = BtConfig(
        id="test",
        name="test",
        start_date=date(2025, 1, 10),
        end_date=date(2025, 1, 10),
        universe=["000001.SZ"],
        params=params,
    )
    return BacktestEngine(cfg)


@pytest.fixture()
def isolated_db(tmp_path):
    cfg = get_config()
    original_paths = cfg.data_paths
    tmp_root = tmp_path / "data"
    tmp_root.mkdir(parents=True, exist_ok=True)
    cfg.data_paths = DataPaths(root=tmp_root)
    initialize_database()
    try:
        yield
    finally:
        cfg.data_paths = original_paths


def test_buy_respects_risk_caps():
    engine = _engine_with_params(
        {
            "max_position_weight": 0.2,
            "fee_rate": 0.0,
            "slippage_bps": 0.0,
            "max_daily_turnover_ratio": 1.0,
        }
    )
    state = PortfolioState(cash=100_000.0)
    result = BacktestResult()
    features = {
        "liquidity_score": 0.7,
        "risk_penalty": 0.25,
    }
    context = _make_context(100.0, features)
    decision = _make_decision(AgentAction.BUY_L, target_weight=0.5)

    engine._apply_portfolio_updates(
        date(2025, 1, 10),
        state,
        [("000001.SZ", context, decision)],
        result,
    )

    expected_qty = (100_000.0 * 0.2 * (1 - 0.25)) / 100.0
    assert state.holdings["000001.SZ"] == pytest.approx(expected_qty)
    assert state.cash == pytest.approx(100_000.0 - expected_qty * 100.0)
    assert result.trades and result.trades[0]["status"] == "executed"
    assert result.nav_series[0]["turnover"] == pytest.approx(expected_qty * 100.0)


def test_buy_blocked_by_limit_up_records_risk():
    engine = _engine_with_params({})
    state = PortfolioState(cash=50_000.0)
    result = BacktestResult()
    features = {"limit_up": True}
    context = _make_context(100.0, features)
    decision = _make_decision(AgentAction.BUY_M, target_weight=0.1)

    engine._apply_portfolio_updates(
        date(2025, 1, 10),
        state,
        [("000001.SZ", context, decision)],
        result,
    )

    assert "000001.SZ" not in state.holdings
    assert not result.trades
    assert result.risk_events
    assert result.risk_events[0]["reason"] == "limit_up"


def test_position_limit_triggers_risk_event_and_adjusts_execution():
    alerts.clear_warnings()
    engine = _engine_with_params(
        {
            "max_position_weight": 0.3,
            "fee_rate": 0.0,
            "slippage_bps": 0.0,
            "max_daily_turnover_ratio": 1.0,
        }
    )
    state = PortfolioState(cash=100_000.0)
    result = BacktestResult()
    context = _make_context(100.0, {"position_limit": True})
    decision = _make_decision(AgentAction.BUY_L, target_weight=0.4)
    decision.risk_assessment = RiskAssessment(
        status="pending_review",
        reason="position_limit",
        recommended_action=AgentAction.BUY_S,
        notes={"trigger": "position_limit"},
    )

    engine._apply_portfolio_updates(
        date(2025, 1, 10),
        state,
        [("000001.SZ", context, decision)],
        result,
    )

    assert not result.trades, "position limit should block execution despite adjustment"
    assert result.risk_events
    event_with_status = next(
        (event for event in result.risk_events if event.get("risk_status")),
        None,
    )
    assert event_with_status is not None
    assert event_with_status["reason"] == "position_limit"
    assert event_with_status.get("risk_status") == "pending_review"
    warning_messages = [item["message"] for item in alerts.get_warnings()]
    assert any("风险提示" in msg for msg in warning_messages)
    alerts.clear_warnings()


def test_blacklist_blocks_execution_and_warns():
    alerts.clear_warnings()
    engine = _engine_with_params({})
    state = PortfolioState(cash=50_000.0)
    result = BacktestResult()
    context = _make_context(100.0, {"is_blacklisted": True})
    decision = _make_decision(AgentAction.BUY_M, target_weight=0.2)
    decision.risk_assessment = RiskAssessment(
        status="blocked",
        reason="blacklist",
        recommended_action=AgentAction.HOLD,
        notes={"trigger": "is_blacklisted"},
    )

    engine._apply_portfolio_updates(
        date(2025, 1, 10),
        state,
        [("000001.SZ", context, decision)],
        result,
    )

    assert not result.trades
    assert result.risk_events
    event = result.risk_events[0]
    assert event["reason"] == "blacklist"
    assert event.get("risk_status") == "blocked"
    warning_messages = [item["message"] for item in alerts.get_warnings()]
    assert any("风险阻断" in msg for msg in warning_messages)
    alerts.clear_warnings()


def test_sell_applies_slippage_and_fee():
    engine = _engine_with_params(
        {
            "max_position_weight": 0.3,
            "fee_rate": 0.001,
            "slippage_bps": 20.0,
            "max_daily_turnover_ratio": 1.0,
        }
    )
    state = PortfolioState(
        cash=0.0,
        holdings={"000001.SZ": 100.0},
        cost_basis={"000001.SZ": 90.0},
        opened_dates={"000001.SZ": "2024-12-01"},
    )
    result = BacktestResult()
    context = _make_context(100.0, {})
    decision = _make_decision(AgentAction.SELL)

    engine._apply_portfolio_updates(
        date(2025, 1, 10),
        state,
        [("000001.SZ", context, decision)],
        result,
    )

    trade = result.trades[0]
    assert pytest.approx(trade["price"], rel=1e-6) == 100.0 * (1 - 0.002)
    assert pytest.approx(trade["fee"], rel=1e-6) == trade["value"] * 0.001
    assert state.cash == pytest.approx(trade["value"] - trade["fee"])
    assert state.realized_pnl == pytest.approx((trade["price"] - 90.0) * 100 - trade["fee"])
    assert not state.holdings
    assert result.nav_series[0]["turnover"] == pytest.approx(trade["value"])
    assert not result.risk_events


def test_persist_backtest_results_saves_risk_events(isolated_db):
    cfg = BtConfig(
        id="risk_cfg",
        name="risk",
        start_date=date(2025, 1, 10),
        end_date=date(2025, 1, 10),
        universe=["000001.SZ"],
        params={},
    )
    result = BacktestResult()
    result.nav_series = [
        {
            "trade_date": "2025-01-10",
            "nav": 100.0,
            "cash": 100.0,
            "market_value": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "turnover": 0.0,
        }
    ]
    result.risk_events = [
        {
            "trade_date": "2025-01-10",
            "ts_code": "000001.SZ",
            "reason": "limit_up",
            "action": "buy_l",
            "target_weight": 0.3,
            "confidence": 0.8,
        }
    ]

    _persist_backtest_results(cfg, result)

    with db_session(read_only=True) as conn:
        risk_row = conn.execute(
            "SELECT reason, metadata FROM bt_risk_events WHERE cfg_id = ?",
            (cfg.id,),
        ).fetchone()
        assert risk_row is not None
        assert risk_row["reason"] == "limit_up"
        metadata = json.loads(risk_row["metadata"])
        assert metadata["action"] == "buy_l"

        summary_row = conn.execute(
            "SELECT summary FROM bt_report WHERE cfg_id = ?",
            (cfg.id,),
        ).fetchone()
        summary = json.loads(summary_row["summary"])
        assert summary["risk_events"] == 1
        assert summary["risk_breakdown"]["limit_up"] == 1
