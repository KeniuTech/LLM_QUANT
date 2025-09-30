"""Utility helpers to retrieve structured data slices for agents and departments."""
from __future__ import annotations

import re
import sqlite3
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

from .db import db_session
from .logging import get_logger
from app.core.indicators import momentum, normalize, rolling_mean, volatility

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "data_broker"}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_safe_identifier(name: str) -> bool:
    return bool(_IDENTIFIER_RE.match(name))


def _safe_split(path: str) -> Tuple[str, str] | None:
    if "." not in path:
        return None
    table, column = path.split(".", 1)
    table = table.strip()
    column = column.strip()
    if not table or not column:
        return None
    if not (_is_safe_identifier(table) and _is_safe_identifier(column)):
        LOGGER.debug("忽略非法字段：%s", path, extra=LOG_EXTRA)
        return None
    return table, column


def parse_field_path(path: str) -> Tuple[str, str] | None:
    """Validate and split a `table.column` field expression."""

    return _safe_split(path)


def _parse_trade_date(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("-", "")
    try:
        return datetime.strptime(text[:8], "%Y%m%d")
    except ValueError:
        return None


def _start_of_day(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d 00:00:00")


def _end_of_day(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d 23:59:59")


@dataclass
class DataBroker:
    """Lightweight data access helper for agent/LLM consumption."""

    FIELD_ALIASES: ClassVar[Dict[str, Dict[str, str]]] = {
        "daily": {
            "volume": "vol",
            "vol": "vol",
            "turnover": "amount",
        },
        "daily_basic": {
            "turnover": "turnover_rate",
            "turnover_rate": "turnover_rate",
            "turnover_rate_f": "turnover_rate_f",
            "volume_ratio": "volume_ratio",
            "pe": "pe",
            "pb": "pb",
            "ps": "ps",
            "ps_ttm": "ps_ttm",
            "dividend_yield": "dv_ratio",
        },
        "stk_limit": {
            "up": "up_limit",
            "down": "down_limit",
        },
    }
    MAX_WINDOW: ClassVar[int] = 120
    BENCHMARK_INDEX: ClassVar[str] = "000300.SH"

    enable_cache: bool = True
    latest_cache_size: int = 256
    series_cache_size: int = 512
    _latest_cache: OrderedDict = field(init=False, repr=False)
    _series_cache: OrderedDict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._latest_cache = OrderedDict()
        self._series_cache = OrderedDict()

    def fetch_latest(
        self,
        ts_code: str,
        trade_date: str,
        fields: Iterable[str],
    ) -> Dict[str, Any]:
        """Fetch the latest value (<= trade_date) for each requested field."""
        field_list = [str(item) for item in fields if item]
        cache_key: Optional[Tuple[Any, ...]] = None
        if self.enable_cache and field_list:
            cache_key = (ts_code, trade_date, tuple(sorted(field_list)))
            cached = self._cache_lookup(self._latest_cache, cache_key)
            if cached is not None:
                return deepcopy(cached)

        grouped: Dict[str, List[Tuple[str, str]]] = {}
        derived_cache: Dict[str, Any] = {}
        results: Dict[str, Any] = {}
        for field_name in field_list:
            parsed = parse_field_path(field_name)
            if not parsed:
                derived = self._resolve_derived_field(
                    ts_code,
                    trade_date,
                    field_name,
                    derived_cache,
                )
                if derived is not None:
                    results[field_name] = derived
                continue
            table, column = parsed
            grouped.setdefault(table, []).append((column, field_name))

        if not grouped:
            if cache_key is not None and results:
                self._cache_store(
                    self._latest_cache,
                    cache_key,
                    deepcopy(results),
                    self.latest_cache_size,
                )
            return results

        try:
            with db_session(read_only=True) as conn:
                for table, items in grouped.items():
                    query = (
                        f"SELECT * FROM {table} "
                        "WHERE ts_code = ? AND trade_date <= ? "
                        "ORDER BY trade_date DESC LIMIT 1"
                    )
                    try:
                        row = conn.execute(query, (ts_code, trade_date)).fetchone()
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug(
                            "查询失败 table=%s fields=%s err=%s",
                            table,
                            [column for column, _field in items],
                            exc,
                            extra=LOG_EXTRA,
                        )
                        continue
                    if not row:
                        continue
                    available = row.keys()
                    for column, original in items:
                        resolved_column = self._resolve_column_in_row(table, column, available)
                        if resolved_column is None:
                            continue
                        value = row[resolved_column]
                        if value is None:
                            continue
                        try:
                            results[original] = float(value)
                        except (TypeError, ValueError):
                            results[original] = value
        except sqlite3.OperationalError as exc:
            LOGGER.debug("数据库只读连接失败：%s", exc, extra=LOG_EXTRA)
            if cache_key is not None:
                cached = self._cache_lookup(self._latest_cache, cache_key)
                if cached is not None:
                    LOGGER.debug(
                        "使用缓存结果 ts_code=%s trade_date=%s",
                        ts_code,
                        trade_date,
                        extra=LOG_EXTRA,
                    )
                    return deepcopy(cached)
        if cache_key is not None and results:
            self._cache_store(
                self._latest_cache,
                cache_key,
                deepcopy(results),
                self.latest_cache_size,
            )
        return results

    def fetch_series(
        self,
        table: str,
        column: str,
        ts_code: str,
        end_date: str,
        window: int,
    ) -> List[Tuple[str, float]]:
        """Return descending time series tuples within the specified window."""

        if window <= 0:
            return []
        window = min(window, self.MAX_WINDOW)
        resolved_field = self.resolve_field(f"{table}.{column}")
        if not resolved_field:
            LOGGER.debug(
                "时间序列字段不存在 table=%s column=%s",
                table,
                column,
                extra=LOG_EXTRA,
            )
            return []
        table, resolved = resolved_field

        cache_key: Optional[Tuple[Any, ...]] = None
        if self.enable_cache:
            cache_key = (table, resolved, ts_code, end_date, window)
            cached = self._cache_lookup(self._series_cache, cache_key)
            if cached is not None:
                return [tuple(item) for item in cached]

        query = (
            f"SELECT trade_date, {resolved} FROM {table} "
            "WHERE ts_code = ? AND trade_date <= ? "
            "ORDER BY trade_date DESC LIMIT ?"
        )
        try:
            with db_session(read_only=True) as conn:
                try:
                    rows = conn.execute(query, (ts_code, end_date, window)).fetchall()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug(
                        "时间序列查询失败 table=%s column=%s err=%s",
                        table,
                        column,
                        exc,
                        extra=LOG_EXTRA,
                    )
                    return []
        except sqlite3.OperationalError as exc:
            LOGGER.debug(
                "时间序列连接失败 table=%s column=%s err=%s",
                table,
                column,
                exc,
                extra=LOG_EXTRA,
            )
            if cache_key is not None:
                cached = self._cache_lookup(self._series_cache, cache_key)
                if cached is not None:
                    LOGGER.debug(
                        "使用缓存时间序列 table=%s column=%s ts_code=%s",
                        table,
                        resolved,
                        ts_code,
                        extra=LOG_EXTRA,
                    )
                    return [tuple(item) for item in cached]
            return []
        series: List[Tuple[str, float]] = []
        for row in rows:
            value = row[resolved]
            if value is None:
                continue
            series.append((row["trade_date"], float(value)))
        if cache_key is not None and series:
            self._cache_store(
                self._series_cache,
                cache_key,
                tuple(series),
                self.series_cache_size,
            )
        return series

    def fetch_flags(
        self,
        table: str,
        ts_code: str,
        trade_date: str,
        where_clause: str,
        params: Sequence[object],
    ) -> bool:
        """Generic helper to test if a record exists (used for limit/suspend lookups)."""

        if not _is_safe_identifier(table):
            return False
        query = (
            f"SELECT 1 FROM {table} WHERE ts_code = ? AND {where_clause} LIMIT 1"
        )
        bind_params = (ts_code, *params)
        try:
            with db_session(read_only=True) as conn:
                try:
                    row = conn.execute(query, bind_params).fetchone()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug(
                        "flag 查询失败 table=%s where=%s err=%s",
                        table,
                        where_clause,
                        exc,
                        extra=LOG_EXTRA,
                    )
                    return False
        except sqlite3.OperationalError as exc:
            LOGGER.debug(
                "flag 查询连接失败 table=%s err=%s",
                table,
                exc,
                extra=LOG_EXTRA,
            )
            return False
        return row is not None

    def fetch_table_rows(
        self,
        table: str,
        ts_code: str,
        trade_date: str,
        window: int,
    ) -> List[Dict[str, object]]:
        if window <= 0:
            return []
        window = min(window, self.MAX_WINDOW)
        columns = self._get_table_columns(table)
        if not columns:
            LOGGER.debug("表不存在或无字段 table=%s", table, extra=LOG_EXTRA)
            return []

        column_list = ", ".join(columns)
        has_trade_date = "trade_date" in columns
        if has_trade_date:
            query = (
                f"SELECT {column_list} FROM {table} "
                "WHERE ts_code = ? AND trade_date <= ? "
                "ORDER BY trade_date DESC LIMIT ?"
            )
            params: Tuple[object, ...] = (ts_code, trade_date, window)
        else:
            query = (
                f"SELECT {column_list} FROM {table} "
                "WHERE ts_code = ? ORDER BY rowid DESC LIMIT ?"
            )
            params = (ts_code, window)

        results: List[Dict[str, object]] = []
        try:
            with db_session(read_only=True) as conn:
                try:
                    rows = conn.execute(query, params).fetchall()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug(
                        "表查询失败 table=%s err=%s",
                        table,
                        exc,
                        extra=LOG_EXTRA,
                    )
                    return []
        except sqlite3.OperationalError as exc:
            LOGGER.debug(
                "表连接失败 table=%s err=%s",
                table,
                exc,
                extra=LOG_EXTRA,
            )
            return []

        for row in rows:
            record = {col: row[col] for col in columns}
            results.append(record)
        return results

    def _resolve_derived_field(
        self,
        ts_code: str,
        trade_date: str,
        field: str,
        cache: Dict[str, Any],
    ) -> Optional[Any]:
        if field in cache:
            return cache[field]

        value: Optional[Any] = None
        if field == "factors.mom_20":
            value = self._derived_price_momentum(ts_code, trade_date, 20)
        elif field == "factors.mom_60":
            value = self._derived_price_momentum(ts_code, trade_date, 60)
        elif field == "factors.volat_20":
            value = self._derived_price_volatility(ts_code, trade_date, 20)
        elif field == "factors.turn_20":
            value = self._derived_turnover_mean(ts_code, trade_date, 20)
        elif field == "news.sentiment_index":
            rows = cache.get("__news_rows__")
            if rows is None:
                rows = self._fetch_recent_news(ts_code, trade_date)
                cache["__news_rows__"] = rows
            value = self._news_sentiment_from_rows(rows)
        elif field == "news.heat_score":
            rows = cache.get("__news_rows__")
            if rows is None:
                rows = self._fetch_recent_news(ts_code, trade_date)
                cache["__news_rows__"] = rows
            value = self._news_heat_from_rows(rows)
        elif field == "macro.industry_heat":
            value = self._derived_industry_heat(ts_code, trade_date)
        elif field in {"macro.relative_strength", "index.performance_peers"}:
            value = self._derived_relative_strength(ts_code, trade_date, cache)

        cache[field] = value
        return value

    def _derived_price_momentum(
        self,
        ts_code: str,
        trade_date: str,
        window: int,
    ) -> Optional[float]:
        series = self.fetch_series("daily", "close", ts_code, trade_date, window)
        values = [value for _dt, value in series]
        if not values:
            return None
        return momentum(values, window)

    def _derived_price_volatility(
        self,
        ts_code: str,
        trade_date: str,
        window: int,
    ) -> Optional[float]:
        series = self.fetch_series("daily", "close", ts_code, trade_date, window)
        values = [value for _dt, value in series]
        if len(values) < 2:
            return None
        return volatility(values, window)

    def _derived_turnover_mean(
        self,
        ts_code: str,
        trade_date: str,
        window: int,
    ) -> Optional[float]:
        series = self.fetch_series(
            "daily_basic",
            "turnover_rate",
            ts_code,
            trade_date,
            window,
        )
        values = [value for _dt, value in series]
        if not values:
            return None
        return rolling_mean(values, window)

    def _fetch_recent_news(
        self,
        ts_code: str,
        trade_date: str,
        days: int = 3,
        limit: int = 120,
    ) -> List[Dict[str, Any]]:
        baseline = _parse_trade_date(trade_date)
        if baseline is None:
            return []
        start = _start_of_day(baseline - timedelta(days=days))
        end = _end_of_day(baseline)
        query = (
            "SELECT sentiment, heat FROM news "
            "WHERE ts_code = ? AND pub_time BETWEEN ? AND ? "
            "ORDER BY pub_time DESC LIMIT ?"
        )
        try:
            with db_session(read_only=True) as conn:
                rows = conn.execute(query, (ts_code, start, end, limit)).fetchall()
        except sqlite3.OperationalError as exc:
            LOGGER.debug(
                "新闻查询连接失败 ts_code=%s err=%s",
                ts_code,
                exc,
                extra=LOG_EXTRA,
            )
            return []
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(
                "新闻查询失败 ts_code=%s err=%s",
                ts_code,
                exc,
                extra=LOG_EXTRA,
            )
            return []
        return [dict(row) for row in rows]

    @staticmethod
    def _news_sentiment_from_rows(rows: List[Dict[str, Any]]) -> Optional[float]:
        sentiments: List[float] = []
        for row in rows:
            value = row.get("sentiment")
            if value is None:
                continue
            try:
                sentiments.append(float(value))
            except (TypeError, ValueError):
                continue
        if not sentiments:
            return None
        avg = sum(sentiments) / len(sentiments)
        return max(-1.0, min(1.0, avg))

    @staticmethod
    def _news_heat_from_rows(rows: List[Dict[str, Any]]) -> Optional[float]:
        if not rows:
            return None
        total_heat = 0.0
        for row in rows:
            value = row.get("heat")
            if value is None:
                continue
            try:
                total_heat += max(float(value), 0.0)
            except (TypeError, ValueError):
                continue
        if total_heat > 0:
            return normalize(total_heat, factor=100.0)
        return normalize(len(rows), factor=20.0)

    def _derived_industry_heat(self, ts_code: str, trade_date: str) -> Optional[float]:
        industry = self._lookup_industry(ts_code)
        if not industry:
            return None
        query = (
            "SELECT heat FROM heat_daily "
            "WHERE scope = ? AND key = ? AND trade_date <= ? "
            "ORDER BY trade_date DESC LIMIT 1"
        )
        try:
            with db_session(read_only=True) as conn:
                row = conn.execute(query, ("industry", industry, trade_date)).fetchone()
        except sqlite3.OperationalError as exc:
            LOGGER.debug(
                "行业热度查询失败 ts_code=%s err=%s",
                ts_code,
                exc,
                extra=LOG_EXTRA,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(
                "行业热度读取异常 ts_code=%s err=%s",
                ts_code,
                exc,
                extra=LOG_EXTRA,
            )
            return None
        if not row:
            return None
        heat_value = row["heat"]
        if heat_value is None:
            return None
        return normalize(heat_value, factor=100.0)

    def _lookup_industry(self, ts_code: str) -> Optional[str]:
        cache = getattr(self, "_industry_cache", None)
        if cache is None:
            cache = {}
            self._industry_cache = cache
        if ts_code in cache:
            return cache[ts_code]
        query = "SELECT industry FROM stock_basic WHERE ts_code = ?"
        try:
            with db_session(read_only=True) as conn:
                row = conn.execute(query, (ts_code,)).fetchone()
        except sqlite3.OperationalError as exc:
            LOGGER.debug(
                "行业查询连接失败 ts_code=%s err=%s",
                ts_code,
                exc,
                extra=LOG_EXTRA,
            )
            cache[ts_code] = None
            return None
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(
                "行业查询失败 ts_code=%s err=%s",
                ts_code,
                exc,
                extra=LOG_EXTRA,
            )
            cache[ts_code] = None
            return None
        industry = None
        if row:
            industry = row["industry"]
        cache[ts_code] = industry
        return industry

    def _derived_relative_strength(
        self,
        ts_code: str,
        trade_date: str,
        cache: Dict[str, Any],
    ) -> Optional[float]:
        window = 20
        series = self.fetch_series("daily", "close", ts_code, trade_date, max(window, 30))
        values = [value for _dt, value in series]
        if not values:
            return None
        stock_momentum = momentum(values, window)
        bench_key = f"__benchmark_mom_{window}"
        benchmark = cache.get(bench_key)
        if benchmark is None:
            benchmark = self._index_momentum(trade_date, window)
            cache[bench_key] = benchmark
        diff = stock_momentum if benchmark is None else stock_momentum - benchmark
        diff = max(-0.2, min(0.2, diff))
        return (diff + 0.2) / 0.4

    def _index_momentum(self, trade_date: str, window: int) -> Optional[float]:
        series = self.fetch_series(
            "index_daily",
            "close",
            self.BENCHMARK_INDEX,
            trade_date,
            window,
        )
        values = [value for _dt, value in series]
        if not values:
            return None
        return momentum(values, window)

    def resolve_field(self, field: str) -> Optional[Tuple[str, str]]:
        normalized = _safe_split(field)
        if not normalized:
            return None
        table, column = normalized
        resolved = self._resolve_column(table, column)
        if not resolved:
            LOGGER.debug(
                "字段不存在 table=%s column=%s",
                table,
                column,
                extra=LOG_EXTRA,
            )
            return None
        return table, resolved

    def _get_table_columns(self, table: str) -> Optional[List[str]]:
        if not _is_safe_identifier(table):
            return None
        cache = getattr(self, "_column_cache", None)
        if cache is None:
            cache = {}
            self._column_cache = cache
        if table in cache:
            return cache[table]
        try:
            with db_session(read_only=True) as conn:
                rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("获取表字段失败 table=%s err=%s", table, exc, extra=LOG_EXTRA)
            cache[table] = None
            return None
        if not rows:
            cache[table] = None
            return None
        columns = [row["name"] for row in rows if row["name"]]
        cache[table] = columns
        return columns

    def _cache_lookup(self, cache: OrderedDict, key: Tuple[Any, ...]) -> Optional[Any]:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def _cache_store(
        self,
        cache: OrderedDict,
        key: Tuple[Any, ...],
        value: Any,
        limit: int,
    ) -> None:
        if not self.enable_cache or limit <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _resolve_column_in_row(
        self,
        table: str,
        column: str,
        available: Sequence[str],
    ) -> Optional[str]:
        alias_map = self.FIELD_ALIASES.get(table, {})
        candidate = alias_map.get(column, column)
        if candidate in available:
            return candidate
        lowered = candidate.lower()
        for name in available:
            if name.lower() == lowered:
                return name
        return None

    def _resolve_column(self, table: str, column: str) -> Optional[str]:
        columns = self._get_table_columns(table)
        if columns is None:
            return None
        alias_map = self.FIELD_ALIASES.get(table, {})
        candidate = alias_map.get(column, column)
        if candidate in columns:
            return candidate
        # Try lower-case or fallback alias normalization
        lowered = candidate.lower()
        for name in columns:
            if name.lower() == lowered:
                return name
        return None
