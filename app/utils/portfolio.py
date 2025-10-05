"""Portfolio data access helpers for candidate pool, positions, and PnL tracking."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .db import db_session
from .logging import get_logger
from .portfolio_init import get_portfolio_config

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "portfolio"}


def _loads_or_default(payload: Optional[str], default: Any) -> Any:
    if not payload:
        return default
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        LOGGER.debug("JSON 解析失败 payload=%s", payload, extra=LOG_EXTRA)
        return default


@dataclass
class InvestmentCandidate:
    trade_date: str
    ts_code: str
    score: Optional[float]
    status: str
    rationale: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]


def list_investment_pool(
    *,
    trade_date: Optional[str] = None,
    status: Optional[Iterable[str]] = None,
    limit: int = 200,
) -> List[InvestmentCandidate]:
    """Return investment candidates for the given trade date (latest if None)."""

    query = [
        "SELECT trade_date, ts_code, score, status, rationale, tags, metadata",
        "FROM investment_pool",
    ]
    params: List[Any] = []

    if trade_date:
        query.append("WHERE trade_date = ?")
        params.append(trade_date)
    else:
        query.append(
            "WHERE trade_date = (SELECT MAX(trade_date) FROM investment_pool)"
        )

    if status:
        placeholders = ", ".join("?" for _ in status)
        query.append(f"AND status IN ({placeholders})")
        params.extend(list(status))

    query.append("ORDER BY (score IS NULL), score DESC, ts_code")
    query.append("LIMIT ?")
    params.append(int(limit))

    sql = "\n".join(query)
    with db_session(read_only=True) as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:  # noqa: BLE001
            LOGGER.exception("查询 investment_pool 失败", extra=LOG_EXTRA)
            return []

    candidates: List[InvestmentCandidate] = []
    for row in rows:
        candidates.append(
            InvestmentCandidate(
                trade_date=row["trade_date"],
                ts_code=row["ts_code"],
                score=row["score"],
                status=row["status"] or "unknown",
                rationale=row["rationale"],
                tags=list(_loads_or_default(row["tags"], [])),
                metadata=dict(_loads_or_default(row["metadata"], {})),
            )
        )
    return candidates


@dataclass
class PortfolioPosition:
    id: int
    ts_code: str
    opened_date: str
    closed_date: Optional[str]
    quantity: float
    cost_price: float
    market_price: Optional[float]
    market_value: Optional[float]
    realized_pnl: float
    unrealized_pnl: float
    target_weight: Optional[float]
    status: str
    notes: Optional[str]
    metadata: Dict[str, Any]


def list_positions(*, active_only: bool = True) -> List[PortfolioPosition]:
    """Return current portfolio positions."""

    sql = """
    SELECT id, ts_code, opened_date, closed_date, quantity, cost_price,
           market_price, market_value, realized_pnl, unrealized_pnl,
           target_weight, status, notes, metadata
    FROM portfolio_positions
    {where_clause}
    ORDER BY status DESC, opened_date DESC
    """

    where_clause = ""
    params: List[Any] = []
    if active_only:
        where_clause = "WHERE status = 'open'"

    sql = sql.format(where_clause=where_clause)
    with db_session(read_only=True) as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:  # noqa: BLE001
            LOGGER.exception("查询 portfolio_positions 失败", extra=LOG_EXTRA)
            return []

    positions: List[PortfolioPosition] = []
    for row in rows:
        positions.append(
            PortfolioPosition(
                id=row["id"],
                ts_code=row["ts_code"],
                opened_date=row["opened_date"],
                closed_date=row["closed_date"],
                quantity=float(row["quantity"]),
                cost_price=float(row["cost_price"]),
                market_price=row["market_price"],
                market_value=row["market_value"],
                realized_pnl=row["realized_pnl"],
                unrealized_pnl=row["unrealized_pnl"],
                target_weight=row["target_weight"],
                status=row["status"],
                notes=row["notes"],
                metadata=dict(_loads_or_default(row["metadata"], {})),
            )
        )
    return positions


@dataclass
class PortfolioSnapshot:
    trade_date: str
    total_value: Optional[float]
    cash: Optional[float]
    invested_value: Optional[float]
    unrealized_pnl: Optional[float]
    realized_pnl: Optional[float]
    net_flow: Optional[float]
    exposure: Optional[float]
    notes: Optional[str]
    metadata: Dict[str, Any]


def get_latest_snapshot() -> Optional[PortfolioSnapshot]:
    """Fetch the most recent portfolio snapshot.
    
    Returns:
        最新的投资组合快照，如果没有数据则返回初始快照（仅包含初始资金）
    """
    sql = """
    SELECT trade_date, total_value, cash, invested_value, unrealized_pnl,
           realized_pnl, net_flow, exposure, notes, metadata
    FROM portfolio_snapshots
    ORDER BY trade_date DESC
    LIMIT 1
    """
    with db_session(read_only=True) as conn:
        try:
            row = conn.execute(sql).fetchone()
        except Exception:  # noqa: BLE001
            LOGGER.exception("查询 portfolio_snapshots 失败", extra=LOG_EXTRA)
            return None

    if not row:
        # 如果没有快照，返回初始状态（只有初始资金）
        config = get_portfolio_config()
        initial_capital = config["initial_capital"]
        return PortfolioSnapshot(
            trade_date="",  # 空日期表示初始状态
            total_value=initial_capital,
            cash=initial_capital,
            invested_value=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            net_flow=0.0,
            exposure=0.0,
            notes="Initial portfolio state",
            metadata={"initial_capital": initial_capital, "currency": config["currency"]},
        )
        
    return PortfolioSnapshot(
        trade_date=row["trade_date"],
        total_value=row["total_value"],
        cash=row["cash"],
        invested_value=row["invested_value"],
        unrealized_pnl=row["unrealized_pnl"],
        realized_pnl=row["realized_pnl"],
        net_flow=row["net_flow"],
        exposure=row["exposure"],
        notes=row["notes"],
        metadata=dict(_loads_or_default(row["metadata"], {})),
    )


def list_recent_trades(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent trades for monitoring purposes."""

    sql = """
    SELECT trade_date, ts_code, action, quantity, price, fee, order_id, source, notes, metadata
    FROM portfolio_trades
    ORDER BY trade_date DESC, id DESC
    LIMIT ?
    """
    with db_session(read_only=True) as conn:
        try:
            rows = conn.execute(sql, (int(limit),)).fetchall()
        except Exception:  # noqa: BLE001
            LOGGER.exception("查询 portfolio_trades 失败", extra=LOG_EXTRA)
            return []

    trades: List[Dict[str, Any]] = []
    for row in rows:
        trades.append(
            {
                "trade_date": row["trade_date"],
                "ts_code": row["ts_code"],
                "action": row["action"],
                "quantity": row["quantity"],
                "price": row["price"],
                "fee": row["fee"],
                "order_id": row["order_id"],
                "source": row["source"],
                "notes": row["notes"],
                "metadata": _loads_or_default(row["metadata"], {}),
            }
        )
    return trades
