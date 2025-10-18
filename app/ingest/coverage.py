"""Data coverage orchestration separated from TuShare API calls."""
from __future__ import annotations

import sqlite3
from inspect import signature
from datetime import date
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from app.data.schema import initialize_database
from app.utils.db import db_session
from app.utils.config import get_config
from app.utils.logging import get_logger

from .api_client import (
    ETF_CODES,
    FUND_CODES,
    FUTURE_CODES,
    FX_CODES,
    HK_CODES,
    INDEX_CODES,
    LOG_EXTRA,
    US_CODES,
    _expected_trading_days,
    _format_date,
    _listing_window,
    ensure_stock_basic,
    ensure_trade_calendar,
    fetch_adj_factor,
    fetch_daily_basic,
    fetch_daily_bars,
    fetch_fund_basic,
    fetch_fund_nav,
    fetch_fut_basic,
    fetch_fut_daily,
    fetch_fx_daily,
    fetch_hk_daily,
    fetch_index_basic,
    fetch_index_daily,
    fetch_index_dailybasic,
    fetch_index_weight,
    fetch_suspensions,
    fetch_stk_limit,
    fetch_trade_calendar,
    fetch_us_daily,
    save_records,
)

LOGGER = get_logger(__name__)


def _range_stats(
    table: str,
    date_col: str,
    start_str: str,
    end_str: str,
    ts_code: str | None = None,
) -> Dict[str, Optional[str]]:
    sql = (
        f"SELECT MIN({date_col}) AS min_d, MAX({date_col}) AS max_d, "
        f"COUNT(DISTINCT {date_col}) AS distinct_days FROM {table} "
        f"WHERE {date_col} BETWEEN ? AND ?"
    )
    params: List[object] = [start_str, end_str]
    if ts_code:
        sql += " AND ts_code = ?"
        params.append(ts_code)
    try:
        with db_session(read_only=True) as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
    except sqlite3.OperationalError:
        return {"min": None, "max": None, "distinct": 0}
    return {
        "min": row["min_d"] if row else None,
        "max": row["max_d"] if row else None,
        "distinct": row["distinct_days"] if row else 0,
    }


def _range_needs_refresh(
    table: str,
    date_col: str,
    start_str: str,
    end_str: str,
    expected_days: int = 0,
    **filters: object,
) -> bool:
    ts_code = filters.get("ts_code") or filters.get("index_code")
    stats = _range_stats(table, date_col, start_str, end_str, ts_code=ts_code)  # type: ignore[arg-type]
    if stats["min"] is None or stats["max"] is None:
        return True
    if stats["min"] > start_str or stats["max"] < end_str:
        return True
    if expected_days and (stats["distinct"] or 0) < expected_days:
        return True
    return False


def _should_skip_range(
    table: str,
    date_col: str,
    start: date,
    end: date,
    ts_code: str | None = None,
) -> bool:
    start_str = _format_date(start)
    end_str = _format_date(end)

    effective_start = start_str
    effective_end = end_str

    if ts_code:
        list_date, delist_date = _listing_window(ts_code)
        if list_date:
            effective_start = max(effective_start, list_date)
        if delist_date:
            effective_end = min(effective_end, delist_date)
        if effective_start > effective_end:
            LOGGER.debug(
                "股票 %s 在目标区间之外，跳过补数",
                ts_code,
                extra=LOG_EXTRA,
            )
            return True
        stats = _range_stats(table, date_col, effective_start, effective_end, ts_code=ts_code)
    else:
        stats = _range_stats(table, date_col, effective_start, effective_end)

    if stats["min"] is None or stats["max"] is None:
        return False
    if stats["min"] > effective_start or stats["max"] < effective_end:
        return False

    if ts_code is None:
        expected_days = _expected_trading_days(effective_start, effective_end)
        if expected_days and (stats["distinct"] or 0) < expected_days:
            return False

    return True


def ensure_index_weights(start: date, end: date, index_codes: Optional[Sequence[str]] = None) -> None:
    if index_codes is None:
        index_codes = [code for code in INDEX_CODES if code.endswith(".SH") or code.endswith(".SZ")]

    for index_code in index_codes:
        start_str = _format_date(start)
        end_str = _format_date(end)
        if _range_needs_refresh("index_weight", "trade_date", start_str, end_str, index_code=index_code):
            LOGGER.info("指数 %s 的成分股权重数据不完整，开始拉取 %s-%s", index_code, start_str, end_str)
            save_records("index_weight", fetch_index_weight(start, end, index_code))
        else:
            LOGGER.info("指数 %s 的成分股权重数据已完整，跳过", index_code)


def ensure_index_dailybasic(start: date, end: date, index_codes: Optional[Sequence[str]] = None) -> None:
    if index_codes is None:
        index_codes = [code for code in INDEX_CODES if code.endswith(".SH") or code.endswith(".SZ")]

    for index_code in index_codes:
        start_str = _format_date(start)
        end_str = _format_date(end)
        if _range_needs_refresh("index_dailybasic", "trade_date", start_str, end_str, ts_code=index_code):
            LOGGER.info("指数 %s 的每日指标数据不完整，开始拉取 %s-%s", index_code, start_str, end_str)
            save_records("index_dailybasic", fetch_index_dailybasic(start, end, index_code))
        else:
            LOGGER.info("指数 %s 的每日指标数据已完整，跳过", index_code)


def ensure_data_coverage(
    start: date,
    end: date,
    ts_codes: Optional[Sequence[str]] = None,
    include_limits: bool = True,
    include_extended: bool = True,
    force: bool = False,
    progress_hook: Callable[[str, float], None] | None = None,
) -> None:
    initialize_database()
    start_str = _format_date(start)
    end_str = _format_date(end)
    cfg = get_config()
    disabled_tables = {
        name.strip().lower()
        for name in getattr(cfg, "disabled_ingest_tables", set())
        if isinstance(name, str) and name.strip()
    }

    def _is_disabled(table: str) -> bool:
        return table.lower() in disabled_tables

    extra_steps = 0
    if include_limits:
        extra_steps += 1
    if include_extended:
        extra_steps += 4
    total_steps = 5 + extra_steps
    current_step = 0

    def advance(message: str) -> None:
        nonlocal current_step
        current_step += 1
        progress = min(current_step / total_steps, 1.0)
        if progress_hook:
            progress_hook(message, progress)
        LOGGER.info(message, extra=LOG_EXTRA)

    advance("准备股票基础信息与交易日历")
    ensure_stock_basic()
    ensure_trade_calendar(start, end)

    codes = tuple(dict.fromkeys(ts_codes)) if ts_codes else tuple()
    expected_days = _expected_trading_days(start_str, end_str)

    advance("处理日线行情数据")
    if codes:
        pending_codes: List[str] = []
        for code in codes:
            if not force and _should_skip_range("daily", "trade_date", start, end, code):
                LOGGER.info("股票 %s 的日线已覆盖 %s-%s，跳过", code, start_str, end_str)
                continue
            pending_codes.append(code)
        if pending_codes:
            LOGGER.info("开始拉取日线行情：%s-%s（待补股票 %d 支）", start_str, end_str, len(pending_codes))
            save_records(
                "daily",
                fetch_daily_bars(start, end, pending_codes, skip_existing=not force),
            )
    else:
        needs_daily = force or _range_needs_refresh("daily", "trade_date", start_str, end_str, expected_days)
        if not needs_daily:
            LOGGER.info("日线数据已覆盖 %s-%s，跳过拉取", start_str, end_str)
        else:
            LOGGER.info("开始拉取日线行情：%s-%s", start_str, end_str)
            save_records(
                "daily",
                fetch_daily_bars(start, end, skip_existing=not force),
            )

    advance("处理指数成分股权重数据")
    ensure_index_weights(start, end)

    advance("处理指数每日指标数据")
    ensure_index_dailybasic(start, end)

    date_cols: Dict[str, str] = {
        "daily_basic": "trade_date",
        "adj_factor": "trade_date",
        "stk_limit": "trade_date",
        "suspend": "suspend_date",
        "index_daily": "trade_date",
        "index_dailybasic": "trade_date",
        "index_weight": "trade_date",
        "fund_nav": "nav_date",
        "fut_daily": "trade_date",
        "fx_daily": "trade_date",
        "hk_daily": "trade_date",
        "us_daily": "trade_date",
    }

    incremental_tables = {"daily_basic", "adj_factor", "stk_limit", "suspend"}
    expected_tables = {"daily_basic", "adj_factor", "stk_limit"}

    def _save_with_codes(
        table: str,
        fetch_fn,
        *,
        targets: Optional[Iterable[str]] = None,
    ) -> None:
        if _is_disabled(table):
            LOGGER.info("表 %s 已在禁用列表，跳过拉取", table, extra=LOG_EXTRA)
            return
        date_col = date_cols.get(table, "trade_date")
        incremental = table in incremental_tables
        sig = signature(fetch_fn)
        has_ts_code = "ts_code" in sig.parameters
        has_skip = "skip_existing" in sig.parameters

        def _call_fetch(code: Optional[str] = None):
            kwargs: Dict[str, object] = {}
            if has_ts_code and code is not None:
                kwargs["ts_code"] = code
            if has_skip:
                kwargs["skip_existing"] = (not force) and incremental
            return fetch_fn(start, end, **kwargs)

        if targets is not None:
            target_iter = list(dict.fromkeys(targets))
        elif codes:
            target_iter = list(codes)
        else:
            target_iter = []

        if target_iter:
            if not has_ts_code:
                LOGGER.info("拉取 %s 表数据（全市场）%s-%s", table, start_str, end_str)
                rows = _call_fetch()
                save_records(table, rows)
                return
            for code in target_iter:
                if not force and _should_skip_range(table, date_col, start, end, code):
                    LOGGER.info("表 %s 股票 %s 已覆盖 %s-%s，跳过", table, code, start_str, end_str)
                    continue
                LOGGER.info("拉取 %s 表数据（股票：%s）%s-%s", table, code, start_str, end_str)
                rows = _call_fetch(code)
                save_records(table, rows)
            return

        if not force:
            if table == "suspend":
                needs_refresh = True
            else:
                expected = expected_days if table in expected_tables else 0
                needs_refresh = _range_needs_refresh(
                    table,
                    date_col,
                    start_str,
                    end_str,
                    expected_days=expected,
                )
            if not needs_refresh:
                LOGGER.info("表 %s 已覆盖 %s-%s，跳过", table, start_str, end_str)
                return
        LOGGER.info("拉取 %s 表数据（全市场）%s-%s", table, start_str, end_str)
        rows = _call_fetch()
        save_records(table, rows)

    advance("处理日线基础指标数据")
    _save_with_codes("daily_basic", fetch_daily_basic)

    advance("处理复权因子数据")
    _save_with_codes("adj_factor", fetch_adj_factor)

    if include_limits:
        advance("处理涨跌停价格数据")
        _save_with_codes("stk_limit", fetch_stk_limit)

    advance("处理停复牌信息")
    _save_with_codes("suspend", fetch_suspensions)

    if include_extended:
        advance("同步指数/基金/期货基础信息")
        save_records("index_basic", fetch_index_basic())
        save_records("fund_basic", fetch_fund_basic())
        save_records("fut_basic", fetch_fut_basic())

        advance("拉取指数行情数据")
        _save_with_codes("index_daily", fetch_index_daily, targets=INDEX_CODES)

        advance("拉取基金净值数据")
        fund_targets = tuple(dict.fromkeys(ETF_CODES + FUND_CODES))
        _save_with_codes("fund_nav", fetch_fund_nav, targets=fund_targets)

        advance("拉取期货/外汇行情数据")
        _save_with_codes("fut_daily", fetch_fut_daily, targets=FUTURE_CODES)
        _save_with_codes("fx_daily", fetch_fx_daily, targets=FX_CODES)

        advance("拉取港/美股行情数据（已暂时关闭）")
        _save_with_codes("hk_daily", fetch_hk_daily, targets=HK_CODES)
        _save_with_codes("us_daily", fetch_us_daily, targets=US_CODES)

    if progress_hook:
        progress_hook("数据覆盖检查完成", 1.0)


def collect_data_coverage(start: date, end: date) -> Dict[str, Dict[str, object]]:
    start_str = _format_date(start)
    end_str = _format_date(end)
    expected_days = _expected_trading_days(start_str, end_str)

    coverage: Dict[str, Dict[str, object]] = {
        "period": {
            "start": start_str,
            "end": end_str,
            "expected_trading_days": expected_days,
        }
    }

    def add_table(name: str, date_col: str, require_days: bool = True) -> None:
        stats = _range_stats(name, date_col, start_str, end_str)
        coverage[name] = {
            "min": stats["min"],
            "max": stats["max"],
            "distinct_days": stats["distinct"],
            "meets_expectation": (
                stats["min"] is not None
                and stats["max"] is not None
                and stats["min"] <= start_str
                and stats["max"] >= end_str
                and ((not require_days) or (stats["distinct"] or 0) >= expected_days)
            ),
        }

    add_table("daily", "trade_date")
    add_table("daily_basic", "trade_date")
    add_table("adj_factor", "trade_date")
    add_table("stk_limit", "trade_date")
    add_table("suspend", "suspend_date", require_days=False)
    add_table("index_daily", "trade_date")
    add_table("fund_nav", "nav_date", require_days=False)
    add_table("fut_daily", "trade_date", require_days=False)
    add_table("fx_daily", "trade_date", require_days=False)
    add_table("hk_daily", "trade_date", require_days=False)
    add_table("us_daily", "trade_date", require_days=False)

    with db_session(read_only=True) as conn:
        stock_tot = conn.execute("SELECT COUNT(*) AS cnt FROM stock_basic").fetchone()
        stock_sse = conn.execute(
            "SELECT COUNT(*) AS cnt FROM stock_basic WHERE exchange = 'SSE' AND list_status = 'L'"
        ).fetchone()
        stock_szse = conn.execute(
            "SELECT COUNT(*) AS cnt FROM stock_basic WHERE exchange = 'SZSE' AND list_status = 'L'"
        ).fetchone()
    coverage["stock_basic"] = {
        "total": stock_tot["cnt"] if stock_tot else 0,
        "sse_listed": stock_sse["cnt"] if stock_sse else 0,
        "szse_listed": stock_szse["cnt"] if stock_szse else 0,
    }

    return coverage


__all__ = [
    "collect_data_coverage",
    "ensure_data_coverage",
    "ensure_index_dailybasic",
    "ensure_index_weights",
]
