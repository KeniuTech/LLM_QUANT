"""TuShare 数据拉取管线实现。"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

try:
    import tushare as ts
except ImportError as exc:  # pragma: no cover - dependency error surfaced at runtime
    ts = None  # type: ignore[assignment]

from app.utils.config import get_config
from app.utils.db import db_session

LOGGER = logging.getLogger(__name__)


@dataclass
class FetchJob:
    name: str
    start: date
    end: date
    granularity: str = "daily"
    ts_codes: Optional[Sequence[str]] = None


_TABLE_SCHEMAS: Dict[str, str] = {
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
            cal_date TEXT PRIMARY KEY,
            is_open INTEGER,
            pretrade_date TEXT
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
    # 对缺失列进行补全，防止写库时缺少绑定参数
    reindexed = df.reindex(columns=allowed_cols)
    return reindexed.where(pd.notnull(reindexed), None).to_dict("records")


def fetch_daily_bars(job: FetchJob) -> Iterable[Dict]:
    """拉取日线行情。"""

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
    """将拉取的数据写入 SQLite。"""

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


def run_ingestion(job: FetchJob, include_limits: bool = True) -> None:
    """按任务配置拉取 TuShare 数据。"""

    LOGGER.info("启动 TuShare 拉取任务：%s", job.name)

    daily_rows = fetch_daily_bars(job)
    save_records("daily", daily_rows)

    suspend_rows = fetch_suspensions(job.start, job.end)
    save_records("suspend", suspend_rows)

    calendar_rows = fetch_trade_calendar(job.start, job.end)
    save_records("trade_calendar", calendar_rows)

    if include_limits:
        limit_rows = fetch_stk_limit(job.start, job.end)
        save_records("stk_limit", limit_rows)

    LOGGER.info("任务 %s 完成", job.name)
