"""TuShare 数据拉取与数据覆盖检查工具。"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

try:
    import tushare as ts
except ImportError:  # pragma: no cover - 运行时提示
    ts = None  # type: ignore[assignment]

from app.utils.config import get_config
from app.utils.db import db_session
from app.data.schema import initialize_database

LOGGER = logging.getLogger(__name__)


@dataclass
class FetchJob:
    name: str
    start: date
    end: date
    granularity: str = "daily"
    ts_codes: Optional[Sequence[str]] = None


_TABLE_SCHEMAS: Dict[str, str] = {
    "stock_basic": """
        CREATE TABLE IF NOT EXISTS stock_basic (
            ts_code TEXT PRIMARY KEY,
            symbol TEXT,
            name TEXT,
            area TEXT,
            industry TEXT,
            market TEXT,
            exchange TEXT,
            list_status TEXT,
            list_date TEXT,
            delist_date TEXT
        );
    """,
    "daily": """
        CREATE TABLE IF NOT EXISTS daily (
            ts_code TEXT,
            trade_date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            pre_close REAL,
            change REAL,
            pct_chg REAL,
            vol REAL,
            amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
    """,
    "daily_basic": """
        CREATE TABLE IF NOT EXISTS daily_basic (
            ts_code TEXT,
            trade_date TEXT,
            close REAL,
            turnover_rate REAL,
            turnover_rate_f REAL,
            volume_ratio REAL,
            pe REAL,
            pe_ttm REAL,
            pb REAL,
            ps REAL,
            ps_ttm REAL,
            dv_ratio REAL,
            dv_ttm REAL,
            total_share REAL,
            float_share REAL,
            free_share REAL,
            total_mv REAL,
            circ_mv REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
    """,
    "adj_factor": """
        CREATE TABLE IF NOT EXISTS adj_factor (
            ts_code TEXT,
            trade_date TEXT,
            adj_factor REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
    """,
    "suspend": """
        CREATE TABLE IF NOT EXISTS suspend (
            ts_code TEXT,
            suspend_date TEXT,
            resume_date TEXT,
            suspend_type TEXT,
            ann_date TEXT,
            suspend_timing TEXT,
            resume_timing TEXT,
            reason TEXT,
            PRIMARY KEY (ts_code, suspend_date)
        );
    """,
    "trade_calendar": """
        CREATE TABLE IF NOT EXISTS trade_calendar (
            exchange TEXT,
            cal_date TEXT,
            is_open INTEGER,
            pretrade_date TEXT,
            PRIMARY KEY (exchange, cal_date)
        );
    """,
    "stk_limit": """
        CREATE TABLE IF NOT EXISTS stk_limit (
            ts_code TEXT,
            trade_date TEXT,
            up_limit REAL,
            down_limit REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
    """,
}

_TABLE_COLUMNS: Dict[str, List[str]] = {
    "stock_basic": [
        "ts_code",
        "symbol",
        "name",
        "area",
        "industry",
        "market",
        "exchange",
        "list_status",
        "list_date",
        "delist_date",
    ],
    "daily": [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ],
    "daily_basic": [
        "ts_code",
        "trade_date",
        "close",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ],
    "adj_factor": [
        "ts_code",
        "trade_date",
        "adj_factor",
    ],
    "suspend": [
        "ts_code",
        "suspend_date",
        "resume_date",
        "suspend_type",
        "ann_date",
        "suspend_timing",
        "resume_timing",
        "reason",
    ],
    "trade_calendar": [
        "exchange",
        "cal_date",
        "is_open",
        "pretrade_date",
    ],
    "stk_limit": [
        "ts_code",
        "trade_date",
        "up_limit",
        "down_limit",
    ],
}


def _ensure_client():
    if ts is None:
        raise RuntimeError("未安装 tushare，请先在环境中安装 tushare 包")
    token = get_config().tushare_token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("未配置 TuShare Token，请在配置文件或环境变量 TUSHARE_TOKEN 中设置")
    if not hasattr(_ensure_client, "_client") or _ensure_client._client is None:  # type: ignore[attr-defined]
        ts.set_token(token)
        _ensure_client._client = ts.pro_api(token)  # type: ignore[attr-defined]
        LOGGER.info("完成 TuShare 客户端初始化")
    return _ensure_client._client  # type: ignore[attr-defined]


def _format_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def _df_to_records(df: pd.DataFrame, allowed_cols: List[str]) -> List[Dict]:
    if df is None or df.empty:
        return []
    reindexed = df.reindex(columns=allowed_cols)
    return reindexed.where(pd.notnull(reindexed), None).to_dict("records")


def _range_stats(table: str, date_col: str, start_str: str, end_str: str) -> Dict[str, Optional[str]]:
    sql = (
        f"SELECT MIN({date_col}) AS min_d, MAX({date_col}) AS max_d, "
        f"COUNT(DISTINCT {date_col}) AS distinct_days FROM {table} "
        f"WHERE {date_col} BETWEEN ? AND ?"
    )
    with db_session(read_only=True) as conn:
        row = conn.execute(sql, (start_str, end_str)).fetchone()
    return {
        "min": row["min_d"],
        "max": row["max_d"],
        "distinct": row["distinct_days"] if row else 0,
    }


def _range_needs_refresh(
    table: str,
    date_col: str,
    start_str: str,
    end_str: str,
    expected_days: int = 0,
) -> bool:
    stats = _range_stats(table, date_col, start_str, end_str)
    if stats["min"] is None or stats["max"] is None:
        return True
    if stats["min"] > start_str or stats["max"] < end_str:
        return True
    if expected_days and (stats["distinct"] or 0) < expected_days:
        return True
    return False


def _calendar_needs_refresh(exchange: str, start_str: str, end_str: str) -> bool:
    sql = """
        SELECT MIN(cal_date) AS min_d, MAX(cal_date) AS max_d, COUNT(*) AS cnt
        FROM trade_calendar
        WHERE exchange = ? AND cal_date BETWEEN ? AND ?
    """
    with db_session(read_only=True) as conn:
        row = conn.execute(sql, (exchange, start_str, end_str)).fetchone()
    if row is None or row["min_d"] is None:
        return True
    if row["min_d"] > start_str or row["max_d"] < end_str:
        return True
    # 交易日历允许不连续（节假日），此处不比较天数
    return False


def _expected_trading_days(start_str: str, end_str: str, exchange: str = "SSE") -> int:
    sql = """
        SELECT COUNT(*) AS cnt
        FROM trade_calendar
        WHERE exchange = ? AND cal_date BETWEEN ? AND ? AND is_open = 1
    """
    with db_session(read_only=True) as conn:
        row = conn.execute(sql, (exchange, start_str, end_str)).fetchone()
    return int(row["cnt"]) if row and row["cnt"] is not None else 0


def fetch_stock_basic(exchange: Optional[str] = None, list_status: str = "L") -> Iterable[Dict]:
    client = _ensure_client()
    LOGGER.info("拉取股票基础信息（交易所：%s，状态：%s）", exchange or "全部", list_status)
    fields = "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date"
    df = client.stock_basic(exchange=exchange, list_status=list_status, fields=fields)
    return _df_to_records(df, _TABLE_COLUMNS["stock_basic"])


def fetch_daily_bars(job: FetchJob) -> Iterable[Dict]:
    client = _ensure_client()
    start_date = _format_date(job.start)
    end_date = _format_date(job.end)
    frames: List[pd.DataFrame] = []

    if job.granularity != "daily":
        raise ValueError(f"暂不支持的粒度：{job.granularity}")

    if job.ts_codes:
        for code in job.ts_codes:
            LOGGER.info("拉取 %s 的日线行情（%s-%s）", code, start_date, end_date)
            frames.append(client.daily(ts_code=code, start_date=start_date, end_date=end_date))
    else:
        LOGGER.info("按全市场拉取日线行情（%s-%s）", start_date, end_date)
        frames.append(client.daily(start_date=start_date, end_date=end_date))

    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return _df_to_records(df, _TABLE_COLUMNS["daily"])


def fetch_daily_basic(start: date, end: date, ts_code: Optional[str] = None) -> Iterable[Dict]:
    client = _ensure_client()
    start_date = _format_date(start)
    end_date = _format_date(end)
    LOGGER.info("拉取日线基础指标（%s-%s，股票：%s）", start_date, end_date, ts_code or "全部")
    df = client.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
    return _df_to_records(df, _TABLE_COLUMNS["daily_basic"])


def fetch_adj_factor(start: date, end: date, ts_code: Optional[str] = None) -> Iterable[Dict]:
    client = _ensure_client()
    start_date = _format_date(start)
    end_date = _format_date(end)
    LOGGER.info("拉取复权因子（%s-%s，股票：%s）", start_date, end_date, ts_code or "全部")
    df = client.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
    return _df_to_records(df, _TABLE_COLUMNS["adj_factor"])


def fetch_suspensions(start: date, end: date, ts_code: Optional[str] = None) -> Iterable[Dict]:
    client = _ensure_client()
    start_date = _format_date(start)
    end_date = _format_date(end)
    LOGGER.info("拉取停复牌信息（%s-%s）", start_date, end_date)
    df = client.suspend_d(ts_code=ts_code, start_date=start_date, end_date=end_date)
    return _df_to_records(df, _TABLE_COLUMNS["suspend"])


def fetch_trade_calendar(start: date, end: date, exchange: str = "SSE") -> Iterable[Dict]:
    client = _ensure_client()
    start_date = _format_date(start)
    end_date = _format_date(end)
    LOGGER.info("拉取交易日历（交易所：%s，区间：%s-%s）", exchange, start_date, end_date)
    df = client.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date)
    return _df_to_records(df, _TABLE_COLUMNS["trade_calendar"])


def fetch_stk_limit(start: date, end: date, ts_code: Optional[str] = None) -> Iterable[Dict]:
    client = _ensure_client()
    start_date = _format_date(start)
    end_date = _format_date(end)
    LOGGER.info("拉取涨跌停价格（%s-%s）", start_date, end_date)
    df = client.stk_limit(ts_code=ts_code, start_date=start_date, end_date=end_date)
    return _df_to_records(df, _TABLE_COLUMNS["stk_limit"])


def save_records(table: str, rows: Iterable[Dict]) -> None:
    items = list(rows)
    if not items:
        LOGGER.info("表 %s 没有新增记录，跳过写入", table)
        return

    schema = _TABLE_SCHEMAS.get(table)
    columns = _TABLE_COLUMNS.get(table)
    if not schema or not columns:
        raise ValueError(f"不支持写入的表：{table}")

    placeholders = ",".join([f":{col}" for col in columns])
    col_clause = ",".join(columns)

    LOGGER.info("表 %s 写入 %d 条记录", table, len(items))
    with db_session() as conn:
        conn.executescript(schema)
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({col_clause}) VALUES ({placeholders})",
            items,
        )


def ensure_stock_basic(list_status: str = "L") -> None:
    exchanges = ("SSE", "SZSE")
    with db_session(read_only=True) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM stock_basic WHERE exchange IN (?, ?) AND list_status = ?",
            (*exchanges, list_status),
        ).fetchone()
    if row and row["cnt"]:
        LOGGER.info("股票基础信息已存在 %d 条记录，跳过拉取", row["cnt"])
        return

    for exch in exchanges:
        save_records("stock_basic", fetch_stock_basic(exchange=exch, list_status=list_status))


def ensure_trade_calendar(start: date, end: date, exchanges: Sequence[str] = ("SSE", "SZSE")) -> None:
    start_str = _format_date(start)
    end_str = _format_date(end)
    for exch in exchanges:
        if _calendar_needs_refresh(exch, start_str, end_str):
            save_records("trade_calendar", fetch_trade_calendar(start, end, exchange=exch))


def ensure_data_coverage(
    start: date,
    end: date,
    ts_codes: Optional[Sequence[str]] = None,
    include_limits: bool = True,
    force: bool = False,
) -> None:
    initialize_database()
    start_str = _format_date(start)
    end_str = _format_date(end)

    ensure_stock_basic()
    ensure_trade_calendar(start, end)

    codes = tuple(dict.fromkeys(ts_codes)) if ts_codes else tuple()
    expected_days = _expected_trading_days(start_str, end_str)
    job = FetchJob("daily_autofill", start=start, end=end, ts_codes=codes)

    if force or _range_needs_refresh("daily", "trade_date", start_str, end_str, expected_days):
        save_records("daily", fetch_daily_bars(job))

    def _save_with_codes(table: str, fetch_fn) -> None:
        if codes:
            for code in codes:
                save_records(table, fetch_fn(start, end, ts_code=code))
        else:
            save_records(table, fetch_fn(start, end))

    if force or _range_needs_refresh("daily_basic", "trade_date", start_str, end_str, expected_days):
        _save_with_codes("daily_basic", fetch_daily_basic)

    if force or _range_needs_refresh("adj_factor", "trade_date", start_str, end_str, expected_days):
        _save_with_codes("adj_factor", fetch_adj_factor)

    if include_limits and (force or _range_needs_refresh("stk_limit", "trade_date", start_str, end_str, expected_days)):
        _save_with_codes("stk_limit", fetch_stk_limit)

    if force or _range_needs_refresh("suspend", "suspend_date", start_str, end_str):
        _save_with_codes("suspend", fetch_suspensions)


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


def run_ingestion(job: FetchJob, include_limits: bool = True) -> None:
    LOGGER.info("启动 TuShare 拉取任务：%s", job.name)
    ensure_data_coverage(job.start, job.end, ts_codes=job.ts_codes, include_limits=include_limits, force=True)
    LOGGER.info("任务 %s 完成", job.name)
