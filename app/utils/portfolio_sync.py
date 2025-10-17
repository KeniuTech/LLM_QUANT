"""Persist live portfolio snapshots, positions, and trades into SQLite tables."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from .db import db_session
from .logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "portfolio_sync"}


def _utc_now() -> str:
    """Return current UTC timestamp formatted like the DB triggers."""

    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _normalize_date(value: str | date | datetime | None, *, field: str) -> str | None:
    """Accept ISO/date/yyyymmdd inputs and convert to ISO strings."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.date().isoformat()
    except ValueError:
        return text


def _json_dumps(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        LOGGER.debug("metadata JSON 序列化失败 field_payload=%s", payload, extra=LOG_EXTRA)
        return None


def _to_float(value: Any, *, field: str, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if allow_none:
            LOGGER.debug("字段 %s 非法浮点数值：%s", field, value, extra=LOG_EXTRA)
            return None
        raise ValueError(f"{field} expects numeric value, got {value!r}") from None


@dataclass(frozen=True)
class RealtimeSnapshot:
    trade_date: str | date | datetime
    total_value: float | None = None
    cash: float | None = None
    invested_value: float | None = None
    unrealized_pnl: float | None = None
    realized_pnl: float | None = None
    net_flow: float | None = None
    exposure: float | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RealtimePosition:
    ts_code: str
    opened_date: str | date | datetime
    quantity: float
    cost_price: float
    market_price: float | None = None
    market_value: float | None = None
    realized_pnl: float | None = 0.0
    unrealized_pnl: float | None = 0.0
    target_weight: float | None = None
    status: str = "open"
    closed_date: str | date | datetime | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RealtimeTrade:
    trade_date: str | date | datetime
    ts_code: str
    action: str
    quantity: float
    price: float
    fee: float | None = 0.0
    order_id: str | None = None
    source: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] | None = None


def sync_portfolio_state(
    snapshot: RealtimeSnapshot,
    positions: Sequence[RealtimePosition] | None = None,
    trades: Sequence[RealtimeTrade] | None = None,
) -> None:
    """Upsert live portfolio data for monitoring and offline analysis.

    Args:
        snapshot: Summary metrics for the current trading day.
        positions: Current open positions to upsert (missing ones will be closed).
        trades: Optional trade executions to record/update (dedup via order_id if present).
    """

    trade_date = _normalize_date(snapshot.trade_date, field="trade_date")
    if not trade_date:
        raise ValueError("snapshot.trade_date is required")

    snapshot_payload = (
        trade_date,
        _to_float(snapshot.total_value, field="total_value"),
        _to_float(snapshot.cash, field="cash"),
        _to_float(snapshot.invested_value, field="invested_value"),
        _to_float(snapshot.unrealized_pnl, field="unrealized_pnl"),
        _to_float(snapshot.realized_pnl, field="realized_pnl"),
        _to_float(snapshot.net_flow, field="net_flow"),
        _to_float(snapshot.exposure, field="exposure"),
        snapshot.notes,
        _json_dumps(snapshot.metadata),
    )

    now_ts = _utc_now()
    positions = list(positions or [])
    trades = list(trades or [])

    with db_session() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO portfolio_snapshots
            (trade_date, total_value, cash, invested_value, unrealized_pnl, realized_pnl, net_flow, exposure, notes, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            snapshot_payload,
        )

        existing_rows = conn.execute(
            """
            SELECT id, ts_code
            FROM portfolio_positions
            WHERE status = 'open'
            """
        ).fetchall()
        existing_map = {row["ts_code"]: row for row in existing_rows}

        seen_codes: set[str] = set()
        for position in positions:
            ts_code = position.ts_code.strip()
            if not ts_code:
                raise ValueError("position.ts_code is required")
            if ts_code in seen_codes:
                raise ValueError(f"duplicate position payload for {ts_code}")
            seen_codes.add(ts_code)

            opened_date = _normalize_date(position.opened_date, field="opened_date")
            if not opened_date:
                opened_date = trade_date
            closed_date = _normalize_date(position.closed_date, field="closed_date")
            quantity = _to_float(position.quantity, field="quantity", allow_none=False)
            cost_price = _to_float(position.cost_price, field="cost_price", allow_none=False)
            market_price = _to_float(position.market_price, field="market_price")
            market_value = _to_float(position.market_value, field="market_value")
            if market_value is None and market_price is not None:
                market_value = market_price * quantity
            unrealized = _to_float(position.unrealized_pnl, field="unrealized_pnl")
            if unrealized is None and market_value is not None:
                unrealized = market_value - cost_price * quantity
            realized = _to_float(position.realized_pnl, field="realized_pnl")
            target_weight = _to_float(position.target_weight, field="target_weight")
            status = (position.status or "open").strip()
            notes = position.notes
            metadata = _json_dumps(position.metadata)

            existing = existing_map.get(ts_code)
            if existing:
                conn.execute(
                    """
                    UPDATE portfolio_positions
                    SET opened_date = ?, closed_date = ?, quantity = ?, cost_price = ?, market_price = ?,
                        market_value = ?, realized_pnl = ?, unrealized_pnl = ?, target_weight = ?, status = ?,
                        notes = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        opened_date,
                        closed_date,
                        quantity,
                        cost_price,
                        market_price,
                        market_value,
                        realized,
                        unrealized,
                        target_weight,
                        status,
                        notes,
                        metadata,
                        now_ts,
                        existing["id"],
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO portfolio_positions
                    (ts_code, opened_date, closed_date, quantity, cost_price, market_price, market_value,
                     realized_pnl, unrealized_pnl, target_weight, status, notes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts_code,
                        opened_date,
                        closed_date,
                        quantity,
                        cost_price,
                        market_price,
                        market_value,
                        realized,
                        unrealized,
                        target_weight,
                        status,
                        notes,
                        metadata,
                    ),
                )

        stale_codes = set(existing_map) - seen_codes
        for ts_code in stale_codes:
            row_id = existing_map[ts_code]["id"]
            conn.execute(
                """
                UPDATE portfolio_positions
                SET status = 'closed',
                    closed_date = COALESCE(closed_date, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (trade_date, now_ts, row_id),
            )

        for trade in trades:
            trade_ts = _normalize_date(trade.trade_date, field="trade.trade_date")
            if not trade_ts:
                raise ValueError("trade.trade_date is required")
            ts_code = trade.ts_code.strip()
            if not ts_code:
                raise ValueError("trade.ts_code is required")
            action = trade.action.strip()
            if not action:
                raise ValueError("trade.action is required")
            quantity = _to_float(trade.quantity, field="trade.quantity", allow_none=False)
            price = _to_float(trade.price, field="trade.price", allow_none=False)
            fee = _to_float(trade.fee, field="trade.fee")
            metadata_json = _json_dumps(trade.metadata)
            order_id = (trade.order_id or "").strip() or None

            if order_id:
                existing_trade = conn.execute(
                    "SELECT id FROM portfolio_trades WHERE order_id = ?",
                    (order_id,),
                ).fetchone()
                if existing_trade:
                    conn.execute(
                        """
                        UPDATE portfolio_trades
                        SET trade_date = ?, ts_code = ?, action = ?, quantity = ?, price = ?, fee = ?,
                            source = ?, notes = ?, metadata = ?
                        WHERE id = ?
                        """,
                        (
                            trade_ts,
                            ts_code,
                            action,
                            quantity,
                            price,
                            fee,
                            trade.source,
                            trade.notes,
                            metadata_json,
                            existing_trade["id"],
                        ),
                    )
                    continue

            conn.execute(
                """
                INSERT INTO portfolio_trades
                (trade_date, ts_code, action, quantity, price, fee, order_id, source, notes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_ts,
                    ts_code,
                    action,
                    quantity,
                    price,
                    fee,
                    order_id,
                    trade.source,
                    trade.notes,
                    metadata_json,
                ),
            )

        LOGGER.info(
            "实时持仓写入完成 trade_date=%s positions=%s trades=%s",
            trade_date,
            len(positions),
            len(trades),
            extra=LOG_EXTRA,
        )


__all__ = [
    "RealtimeSnapshot",
    "RealtimePosition",
    "RealtimeTrade",
    "sync_portfolio_state",
]
