"""Tests for BacktestEngine risk-aware execution."""
from __future__ import annotations

from datetime import date

import pytest

from app.agents.base import AgentAction, AgentContext
from app.agents.game import Decision
from app.backtest.engine import BacktestEngine, BacktestResult, BtConfig, PortfolioState


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
