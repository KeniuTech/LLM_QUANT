"""TuShare API client helpers and persistence utilities."""
from __future__ import annotations

import os
import sqlite3
import time
from collections import defaultdict, deque
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

try:
    import tushare as ts
except ImportError:  # pragma: no cover - 运行时提示
    ts = None  # type: ignore[assignment]

from app.utils.config import get_config
from app.utils.db import db_session
from app.utils.logging import get_logger

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
    return False


def ensure_trade_calendar(start: date, end: date, exchanges: Sequence[str] = ("SSE", "SZSE")) -> None:
    start_str = _format_date(start)
    end_str = _format_date(end)
    for exch in exchanges:
        if _calendar_needs_refresh(exch, start_str, end_str):
            save_records("trade_calendar", fetch_trade_calendar(start, end, exchange=exch))


def _expected_trading_days(start_str: str, end_str: str, exchange: str = "SSE") -> int:
    sql = """
        SELECT COUNT(*) AS cnt
        FROM trade_calendar
        WHERE exchange = ? AND cal_date BETWEEN ? AND ? AND is_open = 1
    """
    with db_session(read_only=True) as conn:
        row = conn.execute(sql, (exchange, start_str, end_str)).fetchone()
    return int(row["cnt"]) if row and row["cnt"] is not None else 0


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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


def fetch_daily_bars(
    start: date,
    end: date,
    ts_codes: Optional[Sequence[str]] = None,
    *,
    skip_existing: bool = True,
    exchange: str = "SSE",
) -> Iterable[Dict]:
    client = _ensure_client()
    frames: List[pd.DataFrame] = []

    trade_dates = _load_trade_dates(start, end, exchange=exchange)
    if not trade_dates:
        LOGGER.info("本地交易日历缺失，尝试补全后再拉取日线行情", extra=LOG_EXTRA)
        ensure_trade_calendar(start, end, exchanges=(exchange,))
        trade_dates = _load_trade_dates(start, end, exchange=exchange)

    if ts_codes:
        for code in ts_codes:
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
    *,
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
    *,
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
            LOGGER.info(
                "复权因子已存在，跳过交易日 %s",
                trade_date,
                extra=LOG_EXTRA,
            )
            continue
        params = {"trade_date": trade_date}
        if ts_code:
            params["ts_code"] = ts_code
        LOGGER.info(
            "交易日拉取请求：endpoint=adj_factor params=%s",
            params,
            extra=LOG_EXTRA,
        )
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
    *,
    skip_existing: bool = False,
) -> Iterable[Dict]:
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取停复牌信息（%s-%s，股票：%s）",
        start_str,
        end_str,
        ts_code or "全部",
        extra=LOG_EXTRA,
    )

    params: Dict[str, object] = {
        "start_date": start_str,
        "end_date": end_str,
    }
    if ts_code:
        params["ts_code"] = ts_code
    df = _fetch_paginated("suspend_d", params)
    if df.empty:
        return []

    merged = df.rename(
        columns={
            "ann_date": "ann_date",
            "suspend_date": "suspend_date",
            "resume_date": "resume_date",
            "suspend_type": "suspend_type",
        }
    )
    if skip_existing:
        existing = _existing_suspend_dates(start_str, end_str, ts_code=ts_code)
        if existing:
            merged = merged[~merged["suspend_date"].isin(existing)]
    missing_cols = [col for col in _TABLE_COLUMNS["suspend"] if col not in merged.columns]
    for column in missing_cols:
        merged[column] = None
    ordered = merged[_TABLE_COLUMNS["suspend"]]
    return _df_to_records(ordered, _TABLE_COLUMNS["suspend"])


def fetch_trade_calendar(start: date, end: date, exchange: str = "SSE") -> Iterable[Dict]:
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取交易日历：%s %s-%s",
        exchange,
        start_str,
        end_str,
        extra=LOG_EXTRA,
    )
    df = _fetch_paginated(
        "trade_cal",
        {"exchange": exchange, "start_date": start_str, "end_date": end_str},
        limit=4000,
    )
    return _df_to_records(df, _TABLE_COLUMNS["trade_calendar"])


def fetch_stk_limit(
    start: date,
    end: date,
    ts_code: str | None = None,
    *,
    skip_existing: bool = True,
) -> Iterable[Dict]:
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取涨跌停数据（%s-%s，股票：%s）",
        start_str,
        end_str,
        ts_code or "全部",
        extra=LOG_EXTRA,
    )

    params: Dict[str, object] = {"start_date": start_str, "end_date": end_str}
    if ts_code:
        params["ts_code"] = ts_code
    df = _fetch_paginated("stk_limit", params, limit=4000)
    if df.empty:
        return []
    if skip_existing:
        df = df[
            ~df.apply(
                lambda row: _record_exists("stk_limit", "trade_date", row["trade_date"], row["ts_code"]),
                axis=1,
            )
        ]
    return _df_to_records(df, _TABLE_COLUMNS["stk_limit"])


def fetch_index_basic() -> Iterable[Dict]:
    client = _ensure_client()
    LOGGER.info("拉取指数基础信息", extra=LOG_EXTRA)
    df = _fetch_paginated("index_basic", {"market": "SSE"})
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
        limit=4000,
    )
    return _df_to_records(df, _TABLE_COLUMNS["index_daily"])


def fetch_index_dailybasic(start: date, end: date, ts_code: str) -> Iterable[Dict]:
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
        limit=4000,
    )
    return _df_to_records(df, _TABLE_COLUMNS["index_dailybasic"])


def fetch_index_weight(start: date, end: date, index_code: str) -> Iterable[Dict]:
    client = _ensure_client()
    start_str = _format_date(start)
    end_str = _format_date(end)
    LOGGER.info(
        "拉取指数权重：%s %s-%s",
        index_code,
        start_str,
        end_str,
        extra=LOG_EXTRA,
    )
    df = _fetch_paginated(
        "index_weight",
        {"index_code": index_code, "start_date": start_str, "end_date": end_str},
        limit=4000,
    )
    return _df_to_records(df, _TABLE_COLUMNS["index_weight"])


def fetch_fund_basic(market: Optional[str] = None) -> Iterable[Dict]:
    client = _ensure_client()
    LOGGER.info(
        "拉取基金基础信息（市场：%s）",
        market or "全部",
        extra=LOG_EXTRA,
    )
    params: Dict[str, object] = {}
    if market:
        params["market"] = market
    df = _fetch_paginated("fund_basic", params)
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
        limit=4000,
    )
    return _df_to_records(df, _TABLE_COLUMNS["fund_nav"])


def fetch_fut_basic(exchange: Optional[str] = None) -> Iterable[Dict]:
    client = _ensure_client()
    LOGGER.info(
        "拉取期货基础信息（交易所：%s）",
        exchange or "全部",
        extra=LOG_EXTRA,
    )
    df = _fetch_paginated("fut_basic", {"exchange": exchange})
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


__all__ = [
    "API_DEFAULT_LIMIT",
    "API_RATE_LIMITS",
    "ETF_CODES",
    "FUND_CODES",
    "FUTURE_CODES",
    "FX_CODES",
    "HK_CODES",
    "INDEX_CODES",
    "US_CODES",
    "ensure_stock_basic",
    "ensure_trade_calendar",
    "fetch_adj_factor",
    "fetch_daily_basic",
    "fetch_daily_bars",
    "fetch_fund_basic",
    "fetch_fund_nav",
    "fetch_fut_basic",
    "fetch_fut_daily",
    "fetch_fx_daily",
    "fetch_hk_daily",
    "fetch_index_basic",
    "fetch_index_daily",
    "fetch_index_dailybasic",
    "fetch_index_weight",
    "fetch_stock_basic",
    "fetch_stk_limit",
    "fetch_suspensions",
    "fetch_trade_calendar",
    "fetch_us_daily",
    "save_records",
    "LOG_EXTRA",
    "_expected_trading_days",
    "_format_date",
    "_listing_window",
    "_load_trade_dates",
    "_record_exists",
    "_existing_suspend_dates",
]

