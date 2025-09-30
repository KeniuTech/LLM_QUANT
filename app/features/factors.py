"""Feature engineering for signals and indicator computation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Dict, Iterable, List, Optional, Sequence

from app.core.indicators import momentum, rolling_mean, volatility
from app.data.schema import initialize_database
from app.utils.data_access import DataBroker
from app.utils.db import db_session
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "factor_compute"}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class FactorSpec:
    name: str
    window: int


@dataclass
class FactorResult:
    ts_code: str
    trade_date: date
    values: Dict[str, float | None]


DEFAULT_FACTORS: List[FactorSpec] = [
    FactorSpec("mom_5", 5),
    FactorSpec("mom_20", 20),
    FactorSpec("mom_60", 60),
    FactorSpec("volat_20", 20),
    FactorSpec("turn_20", 20),
    FactorSpec("turn_5", 5),
    FactorSpec("val_pe_score", 0),
    FactorSpec("val_pb_score", 0),
    FactorSpec("volume_ratio_score", 0),
]


def compute_factors(
    trade_date: date,
    factors: Iterable[FactorSpec] = DEFAULT_FACTORS,
    *,
    ts_codes: Optional[Sequence[str]] = None,
    skip_existing: bool = False,
) -> List[FactorResult]:
    """Calculate and persist factor values for the requested date.

    ``ts_codes`` can be supplied to restrict computation to a subset of the
    universe. When ``skip_existing`` is True, securities that already have an
    entry for ``trade_date`` will be ignored.
    """

    specs = [spec for spec in factors if spec.window > 0]
    if not specs:
        return []

    initialize_database()
    trade_date_str = trade_date.strftime("%Y%m%d")

    _ensure_factor_columns(specs)

    allowed = {code.strip().upper() for code in ts_codes or () if code.strip()}
    universe = _load_universe(trade_date_str, allowed if allowed else None)
    if not universe:
        LOGGER.info("无可用标的生成因子 trade_date=%s", trade_date_str, extra=LOG_EXTRA)
        return []

    if skip_existing:
        existing = _existing_factor_codes(trade_date_str)
        universe = [code for code in universe if code not in existing]
        if not universe:
            LOGGER.debug(
                "目标交易日因子已存在 trade_date=%s universe_size=%s",
                trade_date_str,
                len(existing),
                extra=LOG_EXTRA,
            )
            return []

    broker = DataBroker()
    results: List[FactorResult] = []
    rows_to_persist: List[tuple[str, Dict[str, float | None]]] = []
    for ts_code in universe:
        values = _compute_security_factors(broker, ts_code, trade_date_str, specs)
        if not values:
            continue
        results.append(FactorResult(ts_code=ts_code, trade_date=trade_date, values=values))
        rows_to_persist.append((ts_code, values))

    if rows_to_persist:
        _persist_factor_rows(trade_date_str, rows_to_persist, specs)
    return results


def compute_factor_range(
    start: date,
    end: date,
    *,
    factors: Iterable[FactorSpec] = DEFAULT_FACTORS,
    ts_codes: Optional[Sequence[str]] = None,
    skip_existing: bool = True,
) -> List[FactorResult]:
    """Compute factors for all trading days within ``[start, end]`` inclusive."""

    if end < start:
        raise ValueError("end date must not precede start date")

    initialize_database()
    allowed = None
    if ts_codes:
        allowed = tuple(dict.fromkeys(code.strip().upper() for code in ts_codes if code.strip()))
        if not allowed:
            allowed = None

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    trade_dates = _list_trade_dates(start_str, end_str, allowed)

    aggregated: List[FactorResult] = []
    for trade_date_str in trade_dates:
        trade_day = datetime.strptime(trade_date_str, "%Y%m%d").date()
        aggregated.extend(
            compute_factors(
                trade_day,
                factors,
                ts_codes=allowed,
                skip_existing=skip_existing,
            )
        )
    return aggregated


def _load_universe(trade_date: str, allowed: Optional[set[str]] = None) -> List[str]:
    query = "SELECT ts_code FROM daily WHERE trade_date = ? ORDER BY ts_code"
    with db_session(read_only=True) as conn:
        rows = conn.execute(query, (trade_date,)).fetchall()
    codes = [row["ts_code"] for row in rows if row["ts_code"]]
    if allowed:
        allowed_upper = {code.upper() for code in allowed}
        return [code for code in codes if code.upper() in allowed_upper]
    return codes


def _existing_factor_codes(trade_date: str) -> set[str]:
    with db_session(read_only=True) as conn:
        rows = conn.execute(
            "SELECT ts_code FROM factors WHERE trade_date = ?",
            (trade_date,),
        ).fetchall()
    return {row["ts_code"] for row in rows if row["ts_code"]}


def _list_trade_dates(
    start_date: str,
    end_date: str,
    allowed: Optional[Sequence[str]],
) -> List[str]:
    params: List[str] = [start_date, end_date]
    if allowed:
        placeholders = ", ".join("?" for _ in allowed)
        query = (
            "SELECT DISTINCT trade_date FROM daily "
            "WHERE trade_date BETWEEN ? AND ? "
            f"AND ts_code IN ({placeholders}) "
            "ORDER BY trade_date"
        )
        params.extend(allowed)
    else:
        query = (
            "SELECT DISTINCT trade_date FROM daily "
            "WHERE trade_date BETWEEN ? AND ? "
            "ORDER BY trade_date"
        )
    with db_session(read_only=True) as conn:
        rows = conn.execute(query, params).fetchall()
    return [row["trade_date"] for row in rows if row["trade_date"]]


def _compute_security_factors(
    broker: DataBroker,
    ts_code: str,
    trade_date: str,
    specs: Sequence[FactorSpec],
) -> Dict[str, float | None]:
    close_windows = [spec.window for spec in specs if _factor_prefix(spec.name) in {"mom", "volat"}]
    turnover_windows = [spec.window for spec in specs if _factor_prefix(spec.name) == "turn"]
    max_close_window = max(close_windows) if close_windows else 0
    max_turn_window = max(turnover_windows) if turnover_windows else 0

    close_series = _fetch_series_values(
        broker,
        "daily",
        "close",
        ts_code,
        trade_date,
        max_close_window,
    )
    turnover_series = _fetch_series_values(
        broker,
        "daily_basic",
        "turnover_rate",
        ts_code,
        trade_date,
        max_turn_window,
    )

    latest_fields = broker.fetch_latest(
        ts_code,
        trade_date,
        [
            "daily_basic.pe",
            "daily_basic.pb",
            "daily_basic.ps",
            "daily_basic.volume_ratio",
            "daily.amount",
        ],
    )

    results: Dict[str, float | None] = {}
    for spec in specs:
        prefix = _factor_prefix(spec.name)
        if prefix == "mom":
            if len(close_series) >= spec.window:
                results[spec.name] = momentum(close_series, spec.window)
            else:
                results[spec.name] = None
        elif prefix == "volat":
            if len(close_series) >= 2:
                results[spec.name] = volatility(close_series, spec.window)
            else:
                results[spec.name] = None
        elif prefix == "turn":
            if len(turnover_series) >= spec.window:
                results[spec.name] = rolling_mean(turnover_series, spec.window)
            else:
                results[spec.name] = None
        elif spec.name == "val_pe_score":
            pe = latest_fields.get("daily_basic.pe")
            results[spec.name] = _valuation_score(pe, scale=12.0)
        elif spec.name == "val_pb_score":
            pb = latest_fields.get("daily_basic.pb")
            results[spec.name] = _valuation_score(pb, scale=2.5)
        elif spec.name == "volume_ratio_score":
            volume_ratio = latest_fields.get("daily_basic.volume_ratio")
            results[spec.name] = _volume_ratio_score(volume_ratio)
        else:
            LOGGER.debug(
                "忽略未识别的因子 name=%s ts_code=%s",
                spec.name,
                ts_code,
                extra=LOG_EXTRA,
            )
    return results


def _persist_factor_rows(
    trade_date: str,
    rows: Sequence[tuple[str, Dict[str, float | None]]],
    specs: Sequence[FactorSpec],
) -> None:
    columns = sorted({spec.name for spec in specs})
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    insert_columns = ["ts_code", "trade_date", "updated_at", *columns]
    placeholders = ", ".join(["?"] * len(insert_columns))
    update_clause = ", ".join(
        f"{column}=excluded.{column}" for column in ["updated_at", *columns]
    )
    sql = (
        f"INSERT INTO factors ({', '.join(insert_columns)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(ts_code, trade_date) DO UPDATE SET {update_clause}"
    )

    with db_session() as conn:
        for ts_code, values in rows:
            payload = [ts_code, trade_date, timestamp]
            payload.extend(values.get(column) for column in columns)
            conn.execute(sql, payload)


def _ensure_factor_columns(specs: Sequence[FactorSpec]) -> None:
    pending = {spec.name for spec in specs if _IDENTIFIER_RE.match(spec.name)}
    if not pending:
        return
    with db_session() as conn:
        existing_rows = conn.execute("PRAGMA table_info(factors)").fetchall()
        existing = {row["name"] for row in existing_rows}
        for column in sorted(pending - existing):
            conn.execute(f"ALTER TABLE factors ADD COLUMN {column} REAL")


def _fetch_series_values(
    broker: DataBroker,
    table: str,
    column: str,
    ts_code: str,
    trade_date: str,
    window: int,
) -> List[float]:
    if window <= 0:
        return []
    series = broker.fetch_series(table, column, ts_code, trade_date, window)
    values: List[float] = []
    for _dt, raw in series:
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


def _factor_prefix(name: str) -> str:
    return name.split("_", 1)[0] if name else ""


def _valuation_score(value: object, *, scale: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric <= 0:
        return 0.0
    score = scale / (scale + numeric)
    return max(0.0, min(1.0, score))


def _volume_ratio_score(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0:
        numeric = 0.0
    return max(0.0, min(1.0, numeric / 10.0))
