"""TuShare 数据拉取与数据覆盖检查工具。"""
from __future__ import annotations

import os
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

try:
    import tushare as ts
except ImportError:  # pragma: no cover - 运行时提示
    ts = None  # type: ignore[assignment]

from app.utils import alerts
from app.utils.config import get_config
from app.utils.db import db_session
from app.data.schema import initialize_database
from app.utils.logging import get_logger
from app.features.factors import compute_factor_range


LOGGER = get_logger(__name__)

API_DEFAULT_LIMIT = 5000
LOG_EXTRA = {"stage": "data_ingest"}

_CALL_BUCKETS: Dict[str, deque] = defaultdict(deque)

RATE_LIMIT_ERROR_PATTERNS: Tuple[str, ...] = (
    "最多访问该接口",
    "超过接口限制",
    "Frequency limit",
)

API_RATE_LIMITS: Dict[str, int] = {
    "stock_basic": 180,
    "daily": 480,
    "daily_basic": 200,
    "adj_factor": 200,
    "suspend_d": 180,
    "suspend": 180,
    "stk_limit": 200,
    "trade_cal": 200,
    "index_basic": 120,
    "index_daily": 240,
    "fund_basic": 120,
    "fund_nav": 200,
    "fut_basic": 120,
    "fut_daily": 200,
    "fx_daily": 200,
    "hk_daily": 2,
    "us_daily": 200,
}


INDEX_CODES: Tuple[str, ...] = (
    "000001.SH",  # 上证综指
    "000300.SH",  # 沪深300
    "000016.SH",  # 上证50
    "000905.SH",  # 中证500
    "399001.SZ",  # 深证成指
    "399005.SZ",  # 中小板指
    "399006.SZ",  # 创业板指
    "HSI.HI",     # 恒生指数
    "SPX.GI",     # 标普500
    "DJI.GI",     # 道琼斯工业指数
    "IXIC.GI",    # 纳斯达克综合指数
    "GDAXI.GI",   # 德国DAX
    "FTSE.GI",    # 英国富时100
)

ETF_CODES: Tuple[str, ...] = (
    "510300.SH",  # 华泰柏瑞沪深300ETF
    "510500.SH",  # 南方中证500ETF
    "159915.SZ",  # 易方达创业板ETF
)

FUND_CODES: Tuple[str, ...] = (
    "000001.OF",  # 华夏成长
    "110022.OF",  # 易方达消费行业
)

FUTURE_CODES: Tuple[str, ...] = (
    "IF9999.CFE",  # 沪深300股指期货主力
    "IC9999.CFE",  # 中证500股指期货主力
    "IH9999.CFE",  # 上证50股指期货主力
)

FX_CODES: Tuple[str, ...] = (
    "USDCNY",  # 美元人民币
    "EURCNY",  # 欧元人民币
)

HK_CODES: Tuple[str, ...] = (
    "00700.HK",  # 腾讯控股
    "00941.HK",  # 中国移动
    "09618.HK",  # 京东集团-SW
    "09988.HK",  # 阿里巴巴-SW
    "03690.HK",  # 美团-W
)

US_CODES: Tuple[str, ...] = (
    "AAPL.O",  # 苹果
    "MSFT.O",  # 微软
    "BABA.N",  # 阿里巴巴美股
    "JD.O",    # 京东美股
    "PDD.O",   # 拼多多
    "BIDU.O",  # 百度
    "BILI.O",  # 哔哩哔哩
)


def _normalize_date_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _respect_rate_limit(endpoint: str | None) -> None:
    def _throttle(queue: deque, limit: int) -> None:
        if limit <= 0:
            return
        now = time.time()
        window = 60.0
        while queue and now - queue[0] > window:
            queue.popleft()
        if len(queue) >= limit:
            sleep_time = window - (now - queue[0]) + 0.1
            LOGGER.debug(
                "触发限频控制（limit=%s）休眠 %.2f 秒 endpoint=%s",
                limit,
                sleep_time,
                endpoint,
                extra=LOG_EXTRA,
            )
            time.sleep(max(0.1, sleep_time))
        queue.append(time.time())

    bucket_key = endpoint or "_default"
    endpoint_limit = API_RATE_LIMITS.get(bucket_key, 60)
    _throttle(_CALL_BUCKETS[bucket_key], endpoint_limit or 0)


def _df_to_records(df: pd.DataFrame, allowed_cols: List[str]) -> List[Dict]:
    if df is None or df.empty:
        return []
    reindexed = df.reindex(columns=allowed_cols)
    return reindexed.where(pd.notnull(reindexed), None).to_dict("records")


def _fetch_paginated(endpoint: str, params: Dict[str, object], limit: int | None = None) -> pd.DataFrame:
    client = _ensure_client()
    limit = limit or API_DEFAULT_LIMIT
    frames: List[pd.DataFrame] = []
    offset = 0
    clean_params = {k: v for k, v in params.items() if v is not None}
    LOGGER.info(
        "开始调用 TuShare 接口：%s，参数=%s，limit=%s",
        endpoint,
        clean_params,
        limit,
        extra=LOG_EXTRA,
    )
    while True:
        _respect_rate_limit(endpoint)
        call = getattr(client, endpoint)
        try:
            df = call(limit=limit, offset=offset, **clean_params)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if any(pattern in message for pattern in RATE_LIMIT_ERROR_PATTERNS):
                per_minute = API_RATE_LIMITS.get(endpoint or "", 0)
                wait_time = 60.0 / per_minute + 1 if per_minute else 30.0
                wait_time = max(wait_time, 30.0)
                LOGGER.warning(
                    "接口限频触发：%s，原因=%s，等待 %.1f 秒后重试",
                    endpoint,
                    message,
                    wait_time,
                    extra=LOG_EXTRA,
                )
                time.sleep(wait_time)
                continue

            LOGGER.exception(
                "TuShare 接口调用异常：endpoint=%s offset=%s params=%s",
                endpoint,
                offset,
                clean_params,
                extra=LOG_EXTRA,
            )
            raise
        if df is None or df.empty:
            LOGGER.debug(
                "TuShare 返回空数据：endpoint=%s offset=%s",
                endpoint,
                offset,
                extra=LOG_EXTRA,
            )
            break
        LOGGER.debug(
            "TuShare 返回 %s 行：endpoint=%s offset=%s",
            len(df),
            endpoint,
            offset,
            extra=LOG_EXTRA,
        )
        frames.append(df)
        if len(df) < limit:
            break
        offset += limit
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    LOGGER.info(
        "TuShare 调用完成：endpoint=%s 总行数=%s",
        endpoint,
        len(merged),
        extra=LOG_EXTRA,
    )
    return merged


from .job_logger import JobLogger


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
    "index_basic": """
        CREATE TABLE IF NOT EXISTS index_basic (
            ts_code TEXT PRIMARY KEY,
            name TEXT,
            fullname TEXT,
            market TEXT,
            publisher TEXT,
            index_type TEXT,
            category TEXT,
            base_date TEXT,
            base_point REAL,
            list_date TEXT,
            weight_rule TEXT,
            desc TEXT,
            exp_date TEXT
        );
    """,
    "index_daily": """
        CREATE TABLE IF NOT EXISTS index_daily (
            ts_code TEXT,
            trade_date TEXT,
            close REAL,
            open REAL,
            high REAL,
            low REAL,
            pre_close REAL,
            change REAL,
            pct_chg REAL,
            vol REAL,
            amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
    """,
    "index_dailybasic": """
        CREATE TABLE IF NOT EXISTS index_dailybasic (
            ts_code TEXT,
            trade_date TEXT,
            turnover REAL,
            turnover_ratio REAL,
            pe_ttm REAL,
            pb REAL,
            ps_ttm REAL,
            dv_ttm REAL,
            total_mv REAL,
            circ_mv REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
    """,
    "index_weight": """
        CREATE TABLE IF NOT EXISTS index_weight (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_code VARCHAR(10) NOT NULL,
            trade_date VARCHAR(8) NOT NULL,
            ts_code VARCHAR(10) NOT NULL,
            weight FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_index_weight_lookup (index_code, trade_date)
        );
    """,
    "fund_basic": """
        CREATE TABLE IF NOT EXISTS fund_basic (
            ts_code TEXT PRIMARY KEY,
            name TEXT,
            management TEXT,
            custodian TEXT,
            fund_type TEXT,
            found_date TEXT,
            due_date TEXT,
            list_date TEXT,
            issue_date TEXT,
            delist_date TEXT,
            issue_amount REAL,
            m_fee REAL,
            c_fee REAL,
            benchmark TEXT,
            status TEXT,
            invest_type TEXT,
            type TEXT,
            trustee TEXT,
            purc_start_date TEXT,
            redm_start_date TEXT,
            market TEXT
        );
    """,
    "fund_nav": """
        CREATE TABLE IF NOT EXISTS fund_nav (
            ts_code TEXT,
            nav_date TEXT,
            ann_date TEXT,
            unit_nav REAL,
            accum_nav REAL,
            accum_div REAL,
            net_asset REAL,
            total_netasset REAL,
            adj_nav REAL,
            update_flag TEXT,
            PRIMARY KEY (ts_code, nav_date)
        );
    """,
    "fut_basic": """
        CREATE TABLE IF NOT EXISTS fut_basic (
            ts_code TEXT PRIMARY KEY,
            symbol TEXT,
            name TEXT,
            exchange TEXT,
            exchange_full_name TEXT,
            product TEXT,
            product_name TEXT,
            variety TEXT,
            list_date TEXT,
            delist_date TEXT,
            trade_unit REAL,
            per_unit REAL,
            quote_unit TEXT,
            settle_month TEXT,
            contract_size REAL,
            tick_size REAL,
            margin_rate REAL,
            margin_ratio REAL,
            delivery_month TEXT,
            delivery_day TEXT
        );
    """,
    "fut_daily": """
        CREATE TABLE IF NOT EXISTS fut_daily (
            ts_code TEXT,
            trade_date TEXT,
            pre_settle REAL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            settle REAL,
            change1 REAL,
            change2 REAL,
            vol REAL,
            amount REAL,
            oi REAL,
            oi_chg REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
    """,
    "fx_daily": """
        CREATE TABLE IF NOT EXISTS fx_daily (
            ts_code TEXT,
            trade_date TEXT,
            bid REAL,
            ask REAL,
            mid REAL,
            high REAL,
            low REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
    """,
    "hk_daily": """
        CREATE TABLE IF NOT EXISTS hk_daily (
            ts_code TEXT,
            trade_date TEXT,
            close REAL,
            open REAL,
            high REAL,
            low REAL,
            pre_close REAL,
            change REAL,
            pct_chg REAL,
            vol REAL,
            amount REAL,
            exchange TEXT,
            PRIMARY KEY (ts_code, trade_date)
        );
    """,
    "us_daily": """
        CREATE TABLE IF NOT EXISTS us_daily (
            ts_code TEXT,
            trade_date TEXT,
            close REAL,
            open REAL,
            high REAL,
            low REAL,
            pre_close REAL,
            change REAL,
            pct_chg REAL,
            vol REAL,
            amount REAL,
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
    "index_basic": [
        "ts_code",
        "name",
        "fullname",
        "market",
        "publisher",
        "index_type",
        "category",
        "base_date",
        "base_point",
        "list_date",
        "weight_rule",
        "desc",
        "exp_date",
    ],
    "index_daily": [
        "ts_code",
        "trade_date",
        "close",
        "open",
        "high",
        "low",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ],
    "index_dailybasic": [
        "ts_code",
        "trade_date",
        "turnover",
        "turnover_ratio",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "dv_ttm",
        "total_mv",
        "circ_mv",
    ],
    "index_weight": [
        "index_code",
        "trade_date",
        "ts_code",
        "weight",
    ],
    "fund_basic": [
        "ts_code",
        "name",
        "management",
        "custodian",
        "fund_type",
        "found_date",
        "due_date",
        "list_date",
        "issue_date",
        "delist_date",
        "issue_amount",
        "m_fee",
        "c_fee",
        "benchmark",
        "status",
        "invest_type",
        "type",
        "trustee",
        "purc_start_date",
        "redm_start_date",
        "market",
    ],
    "fund_nav": [
        "ts_code",
        "nav_date",
        "ann_date",
        "unit_nav",
        "accum_nav",
        "accum_div",
        "net_asset",
        "total_netasset",
        "adj_nav",
        "update_flag",
    ],
    "fut_basic": [
        "ts_code",
        "symbol",
        "name",
        "exchange",
        "exchange_full_name",
        "product",
        "product_name",
        "variety",
        "list_date",
        "delist_date",
        "trade_unit",
        "per_unit",
        "quote_unit",
        "settle_month",
        "contract_size",
        "tick_size",
        "margin_rate",
        "margin_ratio",
        "delivery_month",
        "delivery_day",
    ],
    "fut_daily": [
        "ts_code",
        "trade_date",
        "pre_settle",
        "open",
        "high",
        "low",
        "close",
        "settle",
        "change1",
        "change2",
        "vol",
        "amount",
        "oi",
        "oi_chg",
    ],
    "fx_daily": [
        "ts_code",
        "trade_date",
        "bid",
        "ask",
        "mid",
        "high",
        "low",
    ],
    "hk_daily": [
        "ts_code",
        "trade_date",
        "close",
        "open",
        "high",
        "low",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "exchange",
    ],
    "us_daily": [
        "ts_code",
        "trade_date",
        "close",
        "open",
        "high",
        "low",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
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


def _load_trade_dates(start: date, end: date, exchange: str = "SSE") -> List[str]:
    start_str = _format_date(start)
    end_str = _format_date(end)
    query = (
        "SELECT cal_date FROM trade_calendar "
        "WHERE exchange = ? AND cal_date BETWEEN ? AND ? AND is_open = 1 ORDER BY cal_date"
    )
    with db_session(read_only=True) as conn:
        rows = conn.execute(query, (exchange, start_str, end_str)).fetchall()
    return [row["cal_date"] for row in rows]


def _record_exists(
    table: str,
    date_col: str,
    trade_date: str,
    ts_code: Optional[str] = None,
) -> bool:
    query = f"SELECT 1 FROM {table} WHERE {date_col} = ?"
    params: Tuple = (trade_date,)
    if ts_code:
        query += " AND ts_code = ?"
        params = (trade_date, ts_code)
    with db_session(read_only=True) as conn:
        row = conn.execute(query, params).fetchone()
    return row is not None


def _should_skip_range(table: str, date_col: str, start: date, end: date, ts_code: str | None = None) -> bool:
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
) -> bool:
    stats = _range_stats(table, date_col, start_str, end_str)
    if stats["min"] is None or stats["max"] is None:
        return True
    if stats["min"] > start_str or stats["max"] < end_str:
        return True
    if expected_days and (stats["distinct"] or 0) < expected_days:
        return True
    return False


def _existing_suspend_dates(start_str: str, end_str: str, ts_code: str | None = None) -> Set[str]:
    sql = "SELECT DISTINCT suspend_date FROM suspend WHERE suspend_date BETWEEN ? AND ?"
    params: List[object] = [start_str, end_str]
    if ts_code:
        sql += " AND ts_code = ?"
        params.append(ts_code)
    try:
        with db_session(read_only=True) as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
    except sqlite3.OperationalError:
        return set()
    return {row["suspend_date"] for row in rows if row["suspend_date"]}


def _listing_window(ts_code: str) -> Tuple[Optional[str], Optional[str]]:
    with db_session(read_only=True) as conn:
        row = conn.execute(
            "SELECT list_date, delist_date FROM stock_basic WHERE ts_code = ?",
            (ts_code,),
        ).fetchone()
    if not row:
        return None, None
    return _normalize_date_str(row["list_date"]), _normalize_date_str(row["delist_date"])  # type: ignore[index]


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
    LOGGER.info(
        "拉取股票基础信息（交易所：%s，状态：%s）",
        exchange or "全部",
        list_status,
        extra=LOG_EXTRA,
    )
    _respect_rate_limit("stock_basic")
    fields = "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date"
    df = client.stock_basic(exchange=exchange, list_status=list_status, fields=fields)
    return _df_to_records(df, _TABLE_COLUMNS["stock_basic"])


def fetch_daily_bars(job: FetchJob, skip_existing: bool = True) -> Iterable[Dict]:
    client = _ensure_client()
    frames: List[pd.DataFrame] = []

    if job.granularity != "daily":
        raise ValueError(f"暂不支持的粒度：{job.granularity}")

    trade_dates = _load_trade_dates(job.start, job.end)
    if not trade_dates:
        LOGGER.info("本地交易日历缺失，尝试补全后再拉取日线行情", extra=LOG_EXTRA)
        ensure_trade_calendar(job.start, job.end)
        trade_dates = _load_trade_dates(job.start, job.end)

    if job.ts_codes:
        for code in job.ts_codes:
            for trade_date in trade_dates:
                if skip_existing and _record_exists("daily", "trade_date", trade_date, code):
                    LOGGER.debug(
                        "日线数据已存在，跳过 %s %s",
                        code,
                        trade_date,
                        extra=LOG_EXTRA,
                    )
                    continue
                LOGGER.debug(
                    "按交易日拉取日线行情：code=%s trade_date=%s",
                    code,
                    trade_date,
                    extra=LOG_EXTRA,
                )
                LOGGER.info(
                    "交易日拉取请求：endpoint=daily code=%s trade_date=%s",
                    code,
                    trade_date,
                    extra=LOG_EXTRA,
                )
                df = _fetch_paginated(
                    "daily",
                    {
                        "trade_date": trade_date,
                        "ts_code": code,
                    },
                )
                if not df.empty:
                    frames.append(df)
    else:
        for trade_date in trade_dates:
            if skip_existing and _record_exists("daily", "trade_date", trade_date):
                LOGGER.debug(
                    "日线数据已存在，跳过交易日 %s",
                    trade_date,
                    extra=LOG_EXTRA,
                )
                continue
            LOGGER.debug("按交易日拉取日线行情：%s", trade_date, extra=LOG_EXTRA)
            LOGGER.info(
                "交易日拉取请求：endpoint=daily trade_date=%s",
                trade_date,
                extra=LOG_EXTRA,
            )
            df = _fetch_paginated("daily", {"trade_date": trade_date})
            if not df.empty:
                frames.append(df)

    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    return _df_to_records(df, _TABLE_COLUMNS["daily"])


def fetch_daily_basic(
    start: date,
    end: date,
    ts_code: Optional[str] = None,
    skip_existing: bool = True,
) -> Iterable[Dict]:
    client = _ensure_client()
    start_date = _format_date(start)
    end_date = _format_date(end)
    LOGGER.info(
        "拉取日线基础指标（%s-%s，股票：%s）",
        start_date,
        end_date,
        ts_code or "全部",
        extra=LOG_EXTRA,
    )

    trade_dates = _load_trade_dates(start, end)
    frames: List[pd.DataFrame] = []
    for trade_date in trade_dates:
        if skip_existing and _record_exists("daily_basic", "trade_date", trade_date, ts_code):
            LOGGER.info(
                "日线基础指标已存在，跳过交易日 %s",
                trade_date,
                extra=LOG_EXTRA,
            )
            continue
        params = {"trade_date": trade_date}
        if ts_code:
            params["ts_code"] = ts_code
        LOGGER.info(
            "交易日拉取请求：endpoint=daily_basic params=%s",
            params,
            extra=LOG_EXTRA,
        )
        df = _fetch_paginated("daily_basic", params)
        if not df.empty:
            frames.append(df)

    if not frames:
        return []

    merged = pd.concat(frames, ignore_index=True)
    return _df_to_records(merged, _TABLE_COLUMNS["daily_basic"])


def fetch_adj_factor(
    start: date,
    end: date,
    ts_code: Optional[str] = None,
    skip_existing: bool = True,
) -> Iterable[Dict]:
    client = _ensure_client()
    start_date = _format_date(start)
    end_date = _format_date(end)
    LOGGER.info(
        "拉取复权因子（%s-%s，股票：%s）",
        start_date,
        end_date,
        ts_code or "全部",
        extra=LOG_EXTRA,
    )

    trade_dates = _load_trade_dates(start, end)
    frames: List[pd.DataFrame] = []
    for trade_date in trade_dates:
        if skip_existing and _record_exists("adj_factor", "trade_date", trade_date, ts_code):
            LOGGER.debug(
                "复权因子已存在，跳过 %s %s",
                ts_code or "ALL",
                trade_date,
                extra=LOG_EXTRA,
            )
            continue
        params = {"trade_date": trade_date}
        if ts_code:
            params["ts_code"] = ts_code
        LOGGER.info("交易日拉取请求：endpoint=adj_factor params=%s", params, extra=LOG_EXTRA)
        df = _fetch_paginated("adj_factor", params)
        if not df.empty:
            frames.append(df)

    if not frames:
        return []

    merged = pd.concat(frames, ignore_index=True)
    return _df_to_records(merged, _TABLE_COLUMNS["adj_factor"])


def fetch_suspensions(
    start: date,
    end: date,
    ts_code: Optional[str] = None,
    skip_existing: bool = True,
) -> Iterable[Dict]:
    client = _ensure_client()
    start_date = _format_date(start)
    end_date = _format_date(end)
    LOGGER.info(
        "拉取停复牌信息（逐日循环）%s-%s 股票=%s",
        start_date,
        end_date,
        ts_code or "全部",
        extra=LOG_EXTRA,
    )
    trade_dates = _load_trade_dates(start, end)
    existing_dates: Set[str] = set()
    if skip_existing:
        existing_dates = _existing_suspend_dates(start_date, end_date, ts_code)
        if existing_dates:
            LOGGER.debug(
                "停复牌已有覆盖日期数量=%s 示例=%s",
                len(existing_dates),
                sorted(existing_dates)[:5],
                extra=LOG_EXTRA,
            )
    frames: List[pd.DataFrame] = []
    for trade_date in trade_dates:
        if skip_existing and trade_date in existing_dates:
            LOGGER.debug(
                "停复牌信息已存在，跳过 %s %s",
                ts_code or "ALL",
                trade_date,
                extra=LOG_EXTRA,
            )
            continue
        params: Dict[str, object] = {"trade_date": trade_date}
        if ts_code:
            params["ts_code"] = ts_code
        LOGGER.info(
            "交易日拉取请求：endpoint=suspend_d params=%s",
            params,
            extra=LOG_EXTRA,
        )
        df = _fetch_paginated("suspend_d", params, limit=2000)
        if not df.empty:
            if "suspend_date" not in df.columns and "trade_date" in df.columns:
                df = df.rename(columns={"trade_date": "suspend_date"})
            frames.append(df)

    if not frames:
        LOGGER.info("停复牌接口未返回数据", extra=LOG_EXTRA)
        return []

    merged = pd.concat(frames, ignore_index=True)
    missing_cols = [col for col in _TABLE_COLUMNS["suspend"] if col not in merged.columns]
    for col in missing_cols:
        merged[col] = None
    ordered = merged[_TABLE_COLUMNS["suspend"]]
    return _df_to_records(ordered, _TABLE_COLUMNS["suspend"])


def fetch_trade_calendar(start: date, end: date, exchange: str = "SSE") -> Iterable[Dict]:
    client = _ensure_client()
    start_date = _format_date(start)
    end_date = _format_date(end)
    LOGGER.info(
        "拉取交易日历（交易所：%s，区间：%s-%s）",
        exchange,
        start_date,
        end_date,
        extra=LOG_EXTRA,
    )
    _respect_rate_limit("trade_cal")
    df = client.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date)
    if df is not None and not df.empty and "is_open" in df.columns:
        df["is_open"] = pd.to_numeric(df["is_open"], errors="coerce").fillna(0).astype(int)
    return _df_to_records(df, _TABLE_COLUMNS["trade_calendar"])


def fetch_stk_limit(
    start: date,
    end: date,
    ts_code: Optional[str] = None,
    skip_existing: bool = True,
) -> Iterable[Dict]:
    client = _ensure_client()
    start_date = _format_date(start)
    end_date = _format_date(end)
    LOGGER.info("拉取涨跌停价格（%s-%s）", start_date, end_date, extra=LOG_EXTRA)
    trade_dates = _load_trade_dates(start, end)
    frames: List[pd.DataFrame] = []
    for trade_date in trade_dates:
        if skip_existing and _record_exists("stk_limit", "trade_date", trade_date, ts_code):
            LOGGER.debug(
                "涨跌停数据已存在，跳过 %s %s",
                ts_code or "ALL",
                trade_date,
                extra=LOG_EXTRA,
            )
            continue
        params = {"trade_date": trade_date}
        if ts_code:
            params["ts_code"] = ts_code
        LOGGER.info("交易日拉取请求：endpoint=stk_limit params=%s", params, extra=LOG_EXTRA)
        df = _fetch_paginated("stk_limit", params)
        if not df.empty:
            frames.append(df)

    if not frames:
        return []

    merged = pd.concat(frames, ignore_index=True)
    return _df_to_records(merged, _TABLE_COLUMNS["stk_limit"])


def fetch_index_basic(market: Optional[str] = None) -> Iterable[Dict]:
    client = _ensure_client()
    LOGGER.info("拉取指数基础信息（market=%s）", market or "all", extra=LOG_EXTRA)
    _respect_rate_limit("index_basic")
    df = client.index_basic(market=market)
    return _df_to_records(df, _TABLE_COLUMNS["index_basic"])


def fetch_index_daily(start: date, end: date, ts_code: str) -> Iterable[Dict]:
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取指数日线：%s %s-%s",
        ts_code,
        start_str,
        end_str,
        extra=LOG_EXTRA,
    )
    df = _fetch_paginated(
        "index_daily",
        {"ts_code": ts_code, "start_date": start_str, "end_date": end_str},
        limit=5000,
    )
    return _df_to_records(df, _TABLE_COLUMNS["index_daily"])


def fetch_index_weight(start: date, end: date, index_code: str) -> Iterable[Dict]:
    """拉取指定指数的成分股权重数据。
    
    Args:
        start: 开始日期
        end: 结束日期
        index_code: 指数代码，如 "000300.SH"
        
    Returns:
        成分股权重数据列表
    """
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取指数成分股权重：%s %s-%s",
        index_code,
        start_str,
        end_str,
        extra=LOG_EXTRA,
    )
    df = _fetch_paginated(
        "index_weight",
        {"index_code": index_code, "start_date": start_str, "end_date": end_str},
        limit=5000,
    )
    return _df_to_records(df, ["index_code", "trade_date", "ts_code", "weight"])


def fetch_index_dailybasic(start: date, end: date, ts_code: str) -> Iterable[Dict]:
    """拉取指定指数的每日指标数据。
    
    Args:
        start: 开始日期
        end: 结束日期
        ts_code: 指数代码，如 "000300.SH"
        
    Returns:
        指数每日指标数据
    """
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取指数每日指标：%s %s-%s",
        ts_code,
        start_str,
        end_str,
        extra=LOG_EXTRA,
    )
    df = _fetch_paginated(
        "index_dailybasic",
        {"ts_code": ts_code, "start_date": start_str, "end_date": end_str},
        limit=5000,
    )
    return _df_to_records(df, ["ts_code", "trade_date", "turnover", "turnover_ratio", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv"])


def fetch_fund_basic(asset_class: str = "E", status: str = "L") -> Iterable[Dict]:
    client = _ensure_client()
    LOGGER.info("拉取基金基础信息：asset_class=%s status=%s", asset_class, status, extra=LOG_EXTRA)
    _respect_rate_limit("fund_basic")
    df = client.fund_basic(market=asset_class, status=status)
    return _df_to_records(df, _TABLE_COLUMNS["fund_basic"])


def fetch_fund_nav(start: date, end: date, ts_code: str) -> Iterable[Dict]:
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取基金净值：%s %s-%s",
        ts_code,
        start_str,
        end_str,
        extra=LOG_EXTRA,
    )
    df = _fetch_paginated(
        "fund_nav",
        {"ts_code": ts_code, "start_date": start_str, "end_date": end_str},
        limit=5000,
    )
    return _df_to_records(df, _TABLE_COLUMNS["fund_nav"])


def fetch_fut_basic(exchange: Optional[str] = None) -> Iterable[Dict]:
    client = _ensure_client()
    LOGGER.info("拉取期货基础信息（exchange=%s）", exchange or "all", extra=LOG_EXTRA)
    _respect_rate_limit("fut_basic")
    df = client.fut_basic(exchange=exchange)
    return _df_to_records(df, _TABLE_COLUMNS["fut_basic"])


def fetch_fut_daily(start: date, end: date, ts_code: str) -> Iterable[Dict]:
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取期货日线：%s %s-%s",
        ts_code,
        start_str,
        end_str,
        extra=LOG_EXTRA,
    )
    df = _fetch_paginated(
        "fut_daily",
        {"ts_code": ts_code, "start_date": start_str, "end_date": end_str},
        limit=4000,
    )
    return _df_to_records(df, _TABLE_COLUMNS["fut_daily"])


def fetch_fx_daily(start: date, end: date, ts_code: str) -> Iterable[Dict]:
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取外汇日线：%s %s-%s",
        ts_code,
        start_str,
        end_str,
        extra=LOG_EXTRA,
    )
    df = _fetch_paginated(
        "fx_daily",
        {"ts_code": ts_code, "start_date": start_str, "end_date": end_str},
        limit=4000,
    )
    return _df_to_records(df, _TABLE_COLUMNS["fx_daily"])


def fetch_hk_daily(start: date, end: date, ts_code: str) -> Iterable[Dict]:
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取港股日线：%s %s-%s",
        ts_code,
        start_str,
        end_str,
        extra=LOG_EXTRA,
    )
    df = _fetch_paginated(
        "hk_daily",
        {"ts_code": ts_code, "start_date": start_str, "end_date": end_str},
        limit=4000,
    )
    return _df_to_records(df, _TABLE_COLUMNS["hk_daily"])


def fetch_us_daily(start: date, end: date, ts_code: str) -> Iterable[Dict]:
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取美股日线：%s %s-%s",
        ts_code,
        start_str,
        end_str,
        extra=LOG_EXTRA,
    )
    df = _fetch_paginated(
        "us_daily",
        {"ts_code": ts_code, "start_date": start_str, "end_date": end_str},
        limit=4000,
    )
    return _df_to_records(df, _TABLE_COLUMNS["us_daily"])


def save_records(table: str, rows: Iterable[Dict]) -> None:
    items = list(rows)
    if not items:
        LOGGER.info("表 %s 没有新增记录，跳过写入", table, extra=LOG_EXTRA)
        return

    schema = _TABLE_SCHEMAS.get(table)
    columns = _TABLE_COLUMNS.get(table)
    if not schema or not columns:
        raise ValueError(f"不支持写入的表：{table}")

    placeholders = ",".join([f":{col}" for col in columns])
    col_clause = ",".join(columns)

    LOGGER.info("表 %s 写入 %d 条记录", table, len(items), extra=LOG_EXTRA)
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
        LOGGER.info(
            "股票基础信息已存在 %d 条记录，跳过拉取",
            row["cnt"],
            extra=LOG_EXTRA,
        )
        return

    for exch in exchanges:
        save_records("stock_basic", fetch_stock_basic(exchange=exch, list_status=list_status))


def ensure_trade_calendar(start: date, end: date, exchanges: Sequence[str] = ("SSE", "SZSE")) -> None:
    start_str = _format_date(start)
    end_str = _format_date(end)
    for exch in exchanges:
        if _calendar_needs_refresh(exch, start_str, end_str):
            save_records("trade_calendar", fetch_trade_calendar(start, end, exchange=exch))


def ensure_index_weights(start: date, end: date, index_codes: Optional[Sequence[str]] = None) -> None:
    """确保指定指数的成分股权重数据完整。
    
    Args:
        start: 开始日期
        end: 结束日期
        index_codes: 指数代码列表，如果为 None 则使用默认的 A 股指数
    """
    if index_codes is None:
        # 默认获取 A 股指数的成分股权重
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
    """确保指定指数的每日指标数据完整。
    
    Args:
        start: 开始日期
        end: 结束日期
        index_codes: 指数代码列表，如果为 None 则使用默认的 A 股指数
    """
    if index_codes is None:
        # 默认获取 A 股指数的每日指标
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
            job = FetchJob("daily_autofill", start=start, end=end, ts_codes=tuple(pending_codes))
            LOGGER.info("开始拉取日线行情：%s-%s（待补股票 %d 支）", start_str, end_str, len(pending_codes))
            save_records("daily", fetch_daily_bars(job, skip_existing=not force))
    else:
        needs_daily = force or _range_needs_refresh("daily", "trade_date", start_str, end_str, expected_days)
        if not needs_daily:
            LOGGER.info("日线数据已覆盖 %s-%s，跳过拉取", start_str, end_str)
        else:
            job = FetchJob("daily_autofill", start=start, end=end)
            LOGGER.info("开始拉取日线行情：%s-%s", start_str, end_str)
            save_records("daily", fetch_daily_bars(job, skip_existing=not force))
    
    advance("处理指数成分股权重数据")
    # 获取默认指数列表
    default_indices = [code for code in INDEX_CODES if code.endswith(".SH") or code.endswith(".SZ")]
    for index_code in default_indices:
        if not force and _should_skip_range("index_weight", "trade_date", start, end, index_code):
            LOGGER.info("指数 %s 的成分股权重已覆盖 %s-%s，跳过", index_code, start_str, end_str)
            continue
        LOGGER.info("开始拉取指数成分股权重：%s %s-%s", index_code, start_str, end_str)
        save_records("index_weight", fetch_index_weight(start, end, index_code))
    
    advance("处理指数每日指标数据")
    for index_code in default_indices:
        if not force and _should_skip_range("index_dailybasic", "trade_date", start, end, index_code):
            LOGGER.info("指数 %s 的每日指标已覆盖 %s-%s，跳过", index_code, start_str, end_str)
            continue
        LOGGER.info("开始拉取指数每日指标：%s %s-%s", index_code, start_str, end_str)
        save_records("index_dailybasic", fetch_index_dailybasic(start, end, index_code))

    date_cols = {
        "daily_basic": "trade_date",
        "adj_factor": "trade_date",
        "stk_limit": "trade_date",
        "suspend": "suspend_date",
    }
    date_cols.update(
        {
            "index_daily": "trade_date",
            "index_dailybasic": "trade_date",
            "index_weight": "trade_date",
            "fund_nav": "nav_date",
            "fut_daily": "trade_date",
            "fx_daily": "trade_date",
            "hk_daily": "trade_date",
            "us_daily": "trade_date",
        }
    )

    def _save_with_codes(table: str, fetch_fn) -> None:
        date_col = date_cols.get(table, "trade_date")
        if codes:
            for code in codes:
                if not force and _should_skip_range(table, date_col, start, end, code):
                    LOGGER.info("表 %s 股票 %s 已覆盖 %s-%s，跳过", table, code, start_str, end_str)
                    continue
                LOGGER.info("拉取 %s 表数据（股票：%s）%s-%s", table, code, start_str, end_str)
                try:
                    kwargs = {"ts_code": code}
                    if fetch_fn in (fetch_daily_basic, fetch_adj_factor, fetch_suspensions, fetch_stk_limit):
                        kwargs["skip_existing"] = not force
                    rows = fetch_fn(start, end, **kwargs)
                except Exception:
                    LOGGER.exception("TuShare 拉取失败：table=%s code=%s", table, code)
                    raise
                save_records(table, rows)
        else:
            needs_refresh = force or table == "suspend"
            if not force and table != "suspend":
                expected = expected_days if table in {"daily_basic", "adj_factor", "stk_limit"} else 0
                needs_refresh = _range_needs_refresh(table, date_col, start_str, end_str, expected)
            if not needs_refresh:
                LOGGER.info("表 %s 已覆盖 %s-%s，跳过", table, start_str, end_str)
                return
            LOGGER.info("拉取 %s 表数据（全市场）%s-%s", table, start_str, end_str)
            try:
                kwargs = {}
                if fetch_fn in (fetch_daily_basic, fetch_adj_factor, fetch_suspensions, fetch_stk_limit):
                    kwargs["skip_existing"] = not force
                rows = fetch_fn(start, end, **kwargs)
            except Exception:
                LOGGER.exception("TuShare 拉取失败：table=%s code=全部", table)
                raise
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
        try:
            save_records("index_basic", fetch_index_basic())
            save_records("fund_basic", fetch_fund_basic())
            save_records("fut_basic", fetch_fut_basic())
        except Exception:
            LOGGER.exception("扩展基础信息拉取失败")
            raise

        advance("拉取指数行情数据")
        for code in INDEX_CODES:
            try:
                if not force and _should_skip_range("index_daily", "trade_date", start, end, code):
                    LOGGER.info("指数 %s 已覆盖 %s-%s，跳过", code, start_str, end_str)
                    continue
                save_records("index_daily", fetch_index_daily(start, end, code))
            except Exception:
                LOGGER.exception("指数行情拉取失败：%s", code)
                raise

        advance("拉取基金净值数据")
        fund_targets = tuple(dict.fromkeys(ETF_CODES + FUND_CODES))
        for code in fund_targets:
            try:
                if not force and _should_skip_range("fund_nav", "nav_date", start, end, code):
                    LOGGER.info("基金 %s 净值已覆盖 %s-%s，跳过", code, start_str, end_str)
                    continue
                save_records("fund_nav", fetch_fund_nav(start, end, code))
            except Exception:
                LOGGER.exception("基金净值拉取失败：%s", code)
                raise

        advance("拉取期货/外汇行情数据")
        for code in FUTURE_CODES:
            try:
                if not force and _should_skip_range("fut_daily", "trade_date", start, end, code):
                    LOGGER.info("期货 %s 已覆盖 %s-%s，跳过", code, start_str, end_str)
                    continue
                save_records("fut_daily", fetch_fut_daily(start, end, code))
            except Exception:
                LOGGER.exception("期货行情拉取失败：%s", code)
                raise
        for code in FX_CODES:
            try:
                if not force and _should_skip_range("fx_daily", "trade_date", start, end, code):
                    LOGGER.info("外汇 %s 已覆盖 %s-%s，跳过", code, start_str, end_str)
                    continue
                save_records("fx_daily", fetch_fx_daily(start, end, code))
            except Exception:
                LOGGER.exception("外汇行情拉取失败：%s", code)
                raise

        advance("拉取港/美股行情数据（已暂时关闭）")

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


def run_ingestion(job: FetchJob, include_limits: bool = True) -> None:
    """运行数据拉取任务。

    Args:
        job: 任务配置
        include_limits: 是否包含涨跌停数据
    """
    with JobLogger("TuShare数据获取") as logger:
        LOGGER.info("启动 TuShare 拉取任务：%s", job.name, extra=LOG_EXTRA)
        
        try:
            # 拉取基础数据
            ensure_data_coverage(
                job.start,
                job.end,
                ts_codes=job.ts_codes,
                include_limits=include_limits,
                include_extended=True,
                force=True,
            )
            
            # 记录任务元数据
            logger.update_metadata({
                "name": job.name,
                "start": str(job.start),
                "end": str(job.end),
                "codes": len(job.ts_codes) if job.ts_codes else 0
            })
            
            alerts.clear_warnings("TuShare")
            
            # 对日线数据计算因子
            if job.granularity == "daily":
                LOGGER.info("开始计算因子：%s", job.name, extra=LOG_EXTRA)
                try:
                    compute_factor_range(
                        job.start,
                        job.end,
                        ts_codes=job.ts_codes,
                        skip_existing=False,
                    )
                    alerts.clear_warnings("Factors")
                except Exception as exc:
                    LOGGER.exception("因子计算失败 job=%s", job.name, extra=LOG_EXTRA)
                    alerts.add_warning("Factors", f"因子计算失败：{job.name}", str(exc))
                    logger.update_status("failed", f"因子计算失败：{exc}")
                    raise
                LOGGER.info("因子计算完成：%s", job.name, extra=LOG_EXTRA)
                alerts.clear_warnings("Factors")
                
        except Exception as exc:
            LOGGER.exception("数据拉取失败 job=%s", job.name, extra=LOG_EXTRA)
            alerts.add_warning("TuShare", f"拉取任务失败：{job.name}", str(exc))
            logger.update_status("failed", f"数据拉取失败：{exc}")
            raise
        LOGGER.info("任务 %s 完成", job.name, extra=LOG_EXTRA)
