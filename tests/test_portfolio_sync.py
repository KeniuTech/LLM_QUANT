"""Tests for live portfolio sync utilities."""
from __future__ import annotations

import pytest

from app.utils.db import db_session
from app.utils.portfolio_sync import (
    RealtimePosition,
    RealtimeSnapshot,
    RealtimeTrade,
    sync_portfolio_state,
)


def _fetch_one(sql: str, params: tuple | None = None):
    with db_session(read_only=True) as conn:
        return conn.execute(sql, params or ()).fetchone()


def _fetch_all(sql: str, params: tuple | None = None):
    with db_session(read_only=True) as conn:
        return conn.execute(sql, params or ()).fetchall()


def test_sync_portfolio_state_inserts_records(isolated_db):
    snapshot = RealtimeSnapshot(
        trade_date="2025-01-10",
        total_value=100_000.0,
        cash=40_000.0,
        invested_value=60_000.0,
        unrealized_pnl=600.0,
        realized_pnl=250.0,
        net_flow=-5_000.0,
        exposure=0.6,
        notes="intraday sync",
        metadata={"source": "broker_api"},
    )
    positions = [
        RealtimePosition(
            ts_code="000001.SZ",
            opened_date="2025-01-03",
            quantity=1_500,
            cost_price=12.5,
            market_price=13.2,
            realized_pnl=200.0,
            unrealized_pnl=1050.0,
            target_weight=0.3,
            metadata={"account": "live"},
        )
    ]
    trades = [
        RealtimeTrade(
            trade_date="2025-01-10",
            ts_code="000001.SZ",
            action="buy",
            quantity=500,
            price=13.2,
            fee=4.5,
            order_id="order-001",
            source="broker",
            notes="increase position",
            metadata={"account": "live"},
        )
    ]

    sync_portfolio_state(snapshot, positions, trades)

    snap_row = _fetch_one("SELECT * FROM portfolio_snapshots WHERE trade_date = ?", ("2025-01-10",))
    assert snap_row is not None
    assert snap_row["total_value"] == pytest.approx(100_000.0)
    assert snap_row["net_flow"] == pytest.approx(-5_000.0)

    pos_row = _fetch_one("SELECT * FROM portfolio_positions WHERE ts_code = '000001.SZ'")
    assert pos_row is not None
    assert pos_row["quantity"] == pytest.approx(1_500.0)
    assert pos_row["status"] == "open"
    assert pos_row["target_weight"] == pytest.approx(0.3)
    assert pos_row["metadata"] is not None

    trade_row = _fetch_one("SELECT * FROM portfolio_trades WHERE order_id = 'order-001'")
    assert trade_row is not None
    assert trade_row["quantity"] == pytest.approx(500.0)
    assert trade_row["price"] == pytest.approx(13.2)
    assert trade_row["source"] == "broker"


def test_sync_portfolio_state_updates_and_closes(isolated_db):
    # prime database with initial state
    initial_snapshot = RealtimeSnapshot(
        trade_date="2025-01-10",
        total_value=90_000.0,
        cash=50_000.0,
        invested_value=40_000.0,
    )
    initial_positions = [
        RealtimePosition(
            ts_code="000001.SZ",
            opened_date="2025-01-02",
            quantity=800,
            cost_price=11.0,
            market_price=11.5,
            status="open",
        ),
        RealtimePosition(
            ts_code="000002.SZ",
            opened_date="2025-01-04",
            quantity=600,
            cost_price=8.0,
            market_price=8.2,
            status="open",
        ),
    ]
    sync_portfolio_state(initial_snapshot, initial_positions, [])

    # update next day with only one open position and revised trade info
    followup_snapshot = RealtimeSnapshot(
        trade_date="2025-01-11",
        total_value=95_000.0,
        cash=54_000.0,
        invested_value=41_000.0,
        net_flow=1_000.0,
    )
    followup_positions = [
        RealtimePosition(
            ts_code="000001.SZ",
            opened_date="2025-01-02",
            quantity=500,
            cost_price=11.0,
            market_price=12.0,
            status="open",
            realized_pnl=300.0,
            target_weight=0.25,
        )
    ]
    trades = [
        RealtimeTrade(
            trade_date="2025-01-11",
            ts_code="000001.SZ",
            action="sell",
            quantity=300,
            price=12.0,
            order_id="order-xyz",
            source="broker",
        ),
        # update existing trade by reusing order id
        RealtimeTrade(
            trade_date="2025-01-11",
            ts_code="000001.SZ",
            action="sell",
            quantity=300,
            price=12.1,
            fee=3.2,
            order_id="order-xyz",
            source="broker",
            notes="amended fill",
        ),
    ]
    sync_portfolio_state(followup_snapshot, followup_positions, trades)

    open_rows = _fetch_all("SELECT ts_code, status, closed_date FROM portfolio_positions")
    status_map = {row["ts_code"]: (row["status"], row["closed_date"]) for row in open_rows}
    assert status_map["000001.SZ"][0] == "open"
    assert status_map["000001.SZ"][1] in (None, "2025-01-11")
    assert status_map["000002.SZ"][0] == "closed"
    assert status_map["000002.SZ"][1] == "2025-01-11"

    trade_rows = _fetch_all("SELECT id FROM portfolio_trades")
    assert len(trade_rows) == 1, "duplicate trades should be merged via order_id"
    trade_record = _fetch_one("SELECT price, fee FROM portfolio_trades WHERE order_id = 'order-xyz'")
    assert trade_record["price"] == pytest.approx(12.1)
    assert trade_record["fee"] == pytest.approx(3.2)
