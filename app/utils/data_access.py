"""Utility helpers to retrieve structured data slices for agents and departments."""
from __future__ import annotations

import re
import sqlite3
import threading
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import get_config
import types
from .db import db_session
from .logging import get_logger
from app.core.indicators import momentum, normalize, rolling_mean, volatility
from app.utils.db_query import BrokerQueryEngine

# 延迟导入，避免循环依赖
collect_data_coverage = None
ensure_data_coverage = None
initialize_database = None

# 在模块加载时尝试导入
if collect_data_coverage is None or ensure_data_coverage is None:
    try:
        from app.ingest.tushare import collect_data_coverage, ensure_data_coverage
    except ImportError:
        # 导入失败时，在实际使用时会报错
        pass

if initialize_database is None:
    try:
        from app.data.schema import initialize_database
    except ImportError:
        # 导入失败时，提供一个空实现
        def initialize_database():
            """Fallback stub used when the real initializer cannot be imported.

            Return a lightweight object with the attributes callers expect
            (executed, skipped, missing_tables) so code that calls
            `initialize_database()` can safely inspect the result.
            """
            return types.SimpleNamespace(executed=0, skipped=True, missing_tables=[])

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


@dataclass
class _RefreshCoordinator:
    """Orchestrates background refresh requests for the broker."""

    broker: "DataBroker"

    def ensure_for_latest(self, trade_date: str, fields: Iterable[str]) -> None:
        parsed_date = _parse_trade_date(trade_date)
        if not parsed_date:
            return
        normalized = parsed_date.strftime("%Y%m%d")
        tables = self._collect_tables(fields)
        if tables and self.broker.check_data_availability(normalized, tables):
            LOGGER.debug(
                "触发近端数据刷新 trade_date=%s tables=%s",
                normalized,
                sorted(tables),
                extra=LOG_EXTRA,
            )
            self.broker._trigger_background_refresh(normalized)

    def ensure_for_series(self, end_date: str, table: str) -> None:
        parsed_date = _parse_trade_date(end_date)
        if not parsed_date:
            return
        normalized = parsed_date.strftime("%Y%m%d")
        if self.broker.check_data_availability(normalized, {table}):
            LOGGER.debug(
                "触发序列刷新 trade_date=%s table=%s",
                normalized,
                table,
                extra=LOG_EXTRA,
            )
            self.broker._trigger_background_refresh(normalized)

    def _collect_tables(self, fields: Iterable[str]) -> Set[str]:
        tables: Set[str] = set()
        for field_name in fields:
            resolved = self.broker.resolve_field(field_name)
            if resolved:
                table, _ = resolved
                tables.add(table)
        return tables


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


def _coerce_date(value: object) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    parsed = _parse_trade_date(value)
    if parsed:
        return parsed.date()
    return None


@dataclass
class DataBroker:
    """Lightweight data access helper with automated data fetching capabilities."""

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
    # 自动补数配置
    AUTO_REFRESH_WINDOW: ClassVar[int] = 7  # 自动补数的时间窗口
    REFRESH_RETRY_INTERVAL: ClassVar[int] = 5  # 补数重试间隔（秒）
    MAX_REFRESH_WAIT: ClassVar[int] = 60  # 最大等待补数完成时间（秒）

    enable_cache: bool = True
    latest_cache_size: int = 256
    series_cache_size: int = 512
    _latest_cache: OrderedDict = field(init=False, repr=False)
    _series_cache: OrderedDict = field(init=False, repr=False)
    # 补数相关状态管理
    _refresh_lock: threading.RLock = field(init=False, repr=False)
    _refresh_in_progress: Dict[str, bool] = field(init=False, repr=False)
    _refresh_callbacks: Dict[str, List[Callable]] = field(init=False, repr=False)
    _coverage_cache: Dict[str, Dict] = field(init=False, repr=False)
    _refresh: _RefreshCoordinator = field(init=False, repr=False)
    _query_engine: BrokerQueryEngine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._latest_cache = OrderedDict()
        self._series_cache = OrderedDict()
        # 初始化补数相关状态
        self._refresh_lock = threading.RLock()
        self._refresh_in_progress = {}
        self._refresh_callbacks = {}
        self._coverage_cache = {}
        self._refresh = _RefreshCoordinator(self)
        self._query_engine = BrokerQueryEngine(db_session)
        if initialize_database is not None:
            initialize_database()  # 确保数据库已初始化
        else:
            LOGGER.warning("initialize_database 函数不可用，数据库可能未初始化", extra=LOG_EXTRA)

    def fetch_latest(
        self,
        ts_code: str,
        trade_date: str,
        fields: Iterable[str],
        auto_refresh: bool = True,
    ) -> Dict[str, Any]:
        """Fetch the latest value (<= trade_date) for each requested field.
        
        Args:
            ts_code: 证券代码
            trade_date: 交易日
            fields: 要查询的字段列表
            auto_refresh: 是否在数据不足时自动触发补数
        """
        field_list = [str(item) for item in fields if item]
        cache_key: Optional[Tuple[Any, ...]] = None
        if self.enable_cache and field_list:
            cache_key = (ts_code, trade_date, tuple(sorted(field_list)))
            cached = self._cache_lookup(self._latest_cache, cache_key)
            if cached is not None:
                return deepcopy(cached)

        # 检查是否需要自动补数
        if auto_refresh:
            self._refresh.ensure_for_latest(trade_date, field_list)

        grouped: Dict[str, List[str]] = {}
        field_map: Dict[Tuple[str, str], List[str]] = {}
        derived_cache: Dict[str, Any] = {}
        results: Dict[str, Any] = {}
        for field_name in field_list:
            resolved = self.resolve_field(field_name)
            if not resolved:
                derived = self._resolve_derived_field(
                    ts_code,
                    trade_date,
                    field_name,
                    derived_cache,
                )
                if derived is not None:
                    results[field_name] = derived
                continue
            table, column = resolved
            grouped.setdefault(table, [])
            if column not in grouped[table]:
                grouped[table].append(column)
            field_map.setdefault((table, column), []).append(field_name)

        if grouped:
            for table, columns in grouped.items():
                try:
                    row = self._query_engine.fetch_latest(table, ts_code, trade_date, columns)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug(
                        "查询失败 table=%s fields=%s err=%s",
                        table,
                        columns,
                        exc,
                        extra=LOG_EXTRA,
                    )
                    continue
                if not row:
                    continue
                for column in columns:
                    value = row[column]
                    if value is None:
                        continue
                    for original in field_map.get((table, column), [f"{table}.{column}"]):
                        try:
                            results[original] = float(value)
                        except (TypeError, ValueError):
                            results[original] = value

        if cache_key is not None and not results:
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
        auto_refresh: bool = True,
    ) -> List[Tuple[str, float]]:
        """Return descending time series tuples within the specified window.
        
        Args:
            table: 表名
            column: 列名
            ts_code: 证券代码
            end_date: 结束日期
            window: 时间窗口大小
            auto_refresh: 是否在数据不足时自动触发补数
        """

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

        # 检查是否需要自动补数
        if auto_refresh:
            self._refresh.ensure_for_series(end_date, table)

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
            rows = self._query_engine.fetch_series(table, resolved, ts_code, end_date, window)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(
                "时间序列查询失败 table=%s column=%s err=%s",
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
            trade_dt = row["trade_date"]
            if value is None or trade_dt is None:
                continue
            try:
                series.append((trade_dt, float(value)))
            except (TypeError, ValueError):
                continue
        if cache_key is not None and series:
            self._cache_store(
                self._series_cache,
                cache_key,
                tuple(series),
                self.series_cache_size,
            )
        return series

    def register_refresh_callback(
        self,
        start: date | str,
        end: date | str,
        callback: Callable[[], None],
    ) -> None:
        """Register a hook invoked after background refresh completes for the window."""

        if callback is None:
            return
        start_date = _coerce_date(start)
        end_date = _coerce_date(end)
        if not start_date or not end_date:
            LOGGER.debug(
                "忽略无效补数回调窗口 start=%s end=%s",
                start,
                end,
                extra=LOG_EXTRA,
            )
            return
        key = f"{start_date}_{end_date}"
        with self._refresh_lock:
            bucket = self._refresh_callbacks.setdefault(key, [])
            if callback not in bucket:
                bucket.append(callback)

    def fetch_flags(
        self,
        table: str,
        ts_code: str,
        trade_date: str,
        where_clause: str,
        params: Sequence[object],
        auto_refresh: bool = True,
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
        auto_refresh: bool = True,
    ) -> List[Dict[str, object]]:
        if window <= 0:
            return []
        window = min(window, self.MAX_WINDOW)
        
        # 检查是否需要自动补数
        if auto_refresh:
            parsed_date = _parse_trade_date(trade_date)
            if parsed_date and self.check_data_availability(trade_date, {table}):
                self._trigger_background_refresh(trade_date)
                # 短暂等待以获取最新数据
                if hasattr(time, 'sleep'):
                    time.sleep(0.5)
        
        columns = self._get_table_columns(table)
        if not columns:
            LOGGER.debug("表不存在或无字段 table=%s", table, extra=LOG_EXTRA)
            return []

        try:
            rows = self._query_engine.fetch_table(
                table,
                columns,
                ts_code,
                trade_date if "trade_date" in columns else None,
                window,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("表查询失败 table=%s err=%s", table, exc, extra=LOG_EXTRA)
            return []

        return [{col: row[col] for col in columns} for row in rows]

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
            # Certain fields are derived at runtime and intentionally
            # do not require physical columns. Suppress noisy debug logs
            # for those known derived fields so startup isn't spammy.
            derived_fields = {
                "macro.industry_heat",
                "macro.relative_strength",
                "index.performance_peers",
                "news.heat_score",
                "news.sentiment_index",
            }
            if f"{table}.{column}" in derived_fields:
                return None
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
    
    def check_data_availability(
        self,
        trade_date: str,
        tables: Set[str] = None,
        threshold: float = 0.8,
    ) -> bool:
        """检查指定交易日的数据是否可用，如不可用则返回True（需要补数）。
        
        Args:
            trade_date: 要检查的交易日
            tables: 要检查的表集合，默认检查主要行情表
            threshold: 数据覆盖率阈值，低于此值需要补数
        
        Returns:
            bool: True表示数据不足，需要补数
        """
        # 如果配置了强制刷新，则始终返回需要补数
        if get_config().force_refresh:
            return True
        
        # 如果未启用自动更新，则不进行补数
        if not get_config().auto_update_data:
            return False
        
        # 默认检查的表
        if tables is None:
            tables = {"daily", "daily_basic", "stock_basic", "trade_cal"}
        
        try:
            # 解析交易日
            parsed_date = _parse_trade_date(trade_date)
            if not parsed_date:
                LOGGER.debug("无法解析交易日: %s", trade_date, extra=LOG_EXTRA)
                return False
            
            # 计算检查窗口
            end_date = parsed_date.strftime('%Y%m%d')
            start_date = (parsed_date - timedelta(days=self.AUTO_REFRESH_WINDOW)).strftime('%Y%m%d')
            
            # 构建缓存键
            cache_key = f"{start_date}_{end_date}_{'_'.join(sorted(tables))}"
            
            # 检查缓存
            if cache_key in self._coverage_cache:
                coverage = self._coverage_cache[cache_key]
                current_time = time.time() if hasattr(time, 'time') else 0
                if coverage.get('timestamp', 0) > current_time - 300:  # 5分钟内有效
                    # 检查是否需要补数
                    for table in tables:
                        table_coverage = coverage.get(table, {})
                        if table_coverage.get('coverage', 0) < threshold:
                            return True
                    return False
            
            # 收集数据覆盖情况
            if collect_data_coverage is None:
                LOGGER.error("collect_data_coverage 函数不可用，请检查导入配置", extra=LOG_EXTRA)
                return False
            
            coverage = collect_data_coverage(
                date.fromisoformat(start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:8]),
                date.fromisoformat(end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:8])
            )
            
            # 保存到缓存
            coverage['timestamp'] = time.time() if hasattr(time, 'time') else 0
            self._coverage_cache[cache_key] = coverage
            
            # 检查是否需要补数
            for table in tables:
                table_coverage = coverage.get(table, {})
                if table_coverage.get('coverage', 0) < threshold:
                    return True
            
        except Exception as exc:
            LOGGER.exception("检查数据可用性失败: %s", exc, extra=LOG_EXTRA)
            # 出错时保守处理，不触发补数
            return False
        
        return False
    
    def _trigger_background_refresh(self, target_date: str) -> None:
        """在后台线程触发数据补数。"""
        parsed_date = _parse_trade_date(target_date)
        if not parsed_date:
            return
        
        # 构建补数日期范围
        end_date = parsed_date.date()
        start_date = end_date - timedelta(days=self.AUTO_REFRESH_WINDOW)
        refresh_key = f"{start_date}_{end_date}"
        
        # 检查是否已经在补数中
        with self._refresh_lock:
            if self._refresh_in_progress.get(refresh_key, False):
                LOGGER.debug("数据补数已经在进行中: %s", refresh_key, extra=LOG_EXTRA)
                return
            
            self._refresh_in_progress[refresh_key] = True
            self._refresh_callbacks.setdefault(refresh_key, [])
        
        def refresh_task():
            try:
                LOGGER.info("开始后台数据补数: %s 至 %s", start_date, end_date, extra=LOG_EXTRA)
                
                # 执行补数
                if ensure_data_coverage is None:
                    LOGGER.error("ensure_data_coverage 函数不可用，请检查导入配置", extra=LOG_EXTRA)
                    with self._refresh_lock:
                        self._refresh_in_progress[refresh_key] = False
                    return
                
                ensure_data_coverage(
                    start_date,
                    end_date,
                    force=False,
                    progress_hook=None
                )
                
                LOGGER.info("后台数据补数完成: %s 至 %s", start_date, end_date, extra=LOG_EXTRA)
                
                # 清除缓存，强制重新加载数据
                self._latest_cache.clear()
                self._series_cache.clear()
                self._coverage_cache.clear()
                
                # 执行回调
                with self._refresh_lock:
                    callbacks = self._refresh_callbacks.pop(refresh_key, [])
                    self._refresh_in_progress[refresh_key] = False

                if callbacks:
                    LOGGER.info(
                        "执行补数回调 count=%s key=%s",
                        len(callbacks),
                        refresh_key,
                        extra=LOG_EXTRA,
                    )
                for callback in callbacks:
                    try:
                        callback()
                    except Exception as exc:
                        LOGGER.exception("补数回调执行失败: %s", exc, extra=LOG_EXTRA)

            except Exception as exc:
                LOGGER.exception("后台数据补数失败: %s", exc, extra=LOG_EXTRA)
                with self._refresh_lock:
                    self._refresh_in_progress[refresh_key] = False
        
        # 启动后台线程
        thread = threading.Thread(target=refresh_task, daemon=True)
        thread.start()
    
    def is_refreshing(self, start_date: str = None, end_date: str = None) -> bool:
        """检查指定日期范围是否正在补数中。"""
        with self._refresh_lock:
            if not start_date and not end_date:
                # 检查是否有任何补数正在进行
                return any(self._refresh_in_progress.values())
            
            # 检查指定日期范围
            for key, in_progress in self._refresh_in_progress.items():
                if in_progress and key.startswith(start_date or '') and key.endswith(end_date or ''):
                    return True
        
        return False
    
    def wait_for_refresh_complete(
        self,
        timeout: float = None,
        start_date: str = None,
        end_date: str = None
    ) -> bool:
        """等待数据补数完成。
        
        Args:
            timeout: 超时时间（秒），默认为MAX_REFRESH_WAIT
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            bool: True表示补数已完成，False表示超时
        """
        if timeout is None:
            timeout = self.MAX_REFRESH_WAIT
        
        start_time = time.time() if hasattr(time, 'time') else 0
        current_time_func = time.time if hasattr(time, 'time') else lambda: 0
        while current_time_func() - start_time < timeout:
            if not self.is_refreshing(start_date, end_date):
                return True
            
            # 短暂休眠后再次检查
            if hasattr(time, 'sleep'):
                time.sleep(min(self.REFRESH_RETRY_INTERVAL, timeout / 10))
        
        return False
    
    def on_data_refresh(
        self,
        callback: Callable,
        start_date: str = None,
        end_date: str = None
    ) -> None:
        """注册数据补数完成的回调函数。"""
        if start_date and end_date:
            refresh_key = f"{start_date}_{end_date}"
            with self._refresh_lock:
                self._refresh_callbacks.setdefault(refresh_key, []).append(callback)
                # 如果当前没有补数在进行，则直接调用回调
                if not self._refresh_in_progress.get(refresh_key, False):
                    try:
                        callback()
                    except Exception as exc:
                        LOGGER.exception("补数回调执行失败: %s", exc, extra=LOG_EXTRA)

    def set_auto_refresh_window(self, days: int) -> None:
        """设置自动补数的时间窗口。
        
        Args:
            days: 自动补数的天数窗口
        """
        if days > 0:
            self.AUTO_REFRESH_WINDOW = days
            LOGGER.info("自动补数窗口已设置为 %d 天", days, extra=LOG_EXTRA)

    def set_refresh_retry_interval(self, seconds: int) -> None:
        """设置补数检查的重试间隔。
        
        Args:
            seconds: 重试间隔（秒）
        """
        if seconds > 0:
            self.REFRESH_RETRY_INTERVAL = seconds
            LOGGER.info("补数重试间隔已设置为 %d 秒", seconds, extra=LOG_EXTRA)

    def set_max_refresh_wait(self, seconds: int) -> None:
        """设置最大等待补数完成时间。
        
        Args:
            seconds: 最大等待时间（秒）
        """
        if seconds > 0:
            self.MAX_REFRESH_WAIT = seconds
            LOGGER.info("最大补数等待时间已设置为 %d 秒", seconds, extra=LOG_EXTRA)

    def force_refresh_data(self, start_date: str, end_date: str) -> bool:
        """强制刷新指定日期范围内的数据。
        
        Args:
            start_date: 开始日期（格式：YYYYMMDD）
            end_date: 结束日期（格式：YYYYMMDD）
        
        Returns:
            bool: 是否成功触发刷新
        """
        try:
            # 解析日期
            start = _parse_trade_date(start_date)
            end = _parse_trade_date(end_date)
            if not start or not end:
                LOGGER.error("日期格式不正确: %s, %s", start_date, end_date, extra=LOG_EXTRA)
                return False
            
            # 触发刷新
            self._trigger_background_refresh(end_date)
            return True
        except Exception as exc:
            LOGGER.exception("强制刷新数据失败: %s", exc, extra=LOG_EXTRA)
            return False

    def get_index_stocks(
        self,
        index_code: str,
        trade_date: str,
        min_weight: float = 0.0
    ) -> List[str]:
        """获取指数成分股列表。
        
        Args:
            index_code: 指数代码(如 000300.SH)
            trade_date: 交易日期
            min_weight: 最小权重筛选
            
        Returns:
            成分股代码列表
        """
        try:
            with db_session(read_only=True) as conn:
                # 获取小于等于给定日期的最新一期成分股
                rows = conn.execute(
                    """
                    SELECT DISTINCT ts_code
                    FROM index_weight
                    WHERE index_code = ?
                    AND trade_date = (
                        SELECT MAX(trade_date)
                        FROM index_weight
                        WHERE index_code = ?
                        AND trade_date <= ?
                    )
                    AND weight >= ?
                    ORDER BY weight DESC
                    """,
                    (index_code, index_code, trade_date, min_weight)
                ).fetchall()
                
                return [row["ts_code"] for row in rows if row and row["ts_code"]]
        except Exception as exc:
            LOGGER.exception(
                "获取指数成分股失败 index=%s date=%s err=%s",
                index_code,
                trade_date,
                exc,
                extra=LOG_EXTRA
            )
            return []
            
    def get_refresh_status(self) -> Dict[str, Dict[str, Any]]:
        """获取当前所有补数任务的状态。
        
        Returns:
            Dict: 包含所有补数任务状态的字典
        """
        with self._refresh_lock:
            status = {}
            for key, in_progress in self._refresh_in_progress.items():
                start, end = key.split('_')[:2] if '_' in key else (key, key)
                status[key] = {
                    'start_date': start,
                    'end_date': end,
                    'in_progress': in_progress,
                    'callback_count': len(self._refresh_callbacks.get(key, []))
                }
            return status

    def cancel_all_refresh_tasks(self) -> None:
        """取消所有正在等待的补数任务回调。
        注意：已经开始执行的补数任务无法取消，但它们的结果将被忽略。
        """
        with self._refresh_lock:
            self._refresh_callbacks.clear()
            # 保留刷新状态以避免立即重新触发
            LOGGER.info("所有补数任务回调已取消", extra=LOG_EXTRA)

    def clear_coverage_cache(self) -> None:
        """清除数据覆盖情况的缓存。"""
        self._coverage_cache.clear()
        LOGGER.info("数据覆盖缓存已清除", extra=LOG_EXTRA)

    def get_data_coverage(self, start_date: str, end_date: str) -> Dict:
        """获取指定日期范围内的数据覆盖情况。
        
        Args:
            start_date: 开始日期（格式：YYYYMMDD）
            end_date: 结束日期（格式：YYYYMMDD）
        
        Returns:
            Dict: 数据覆盖情况的详细信息
        """
        try:
            # 解析日期
            start = _parse_trade_date(start_date)
            end = _parse_trade_date(end_date)
            if not start or not end:
                LOGGER.error("日期格式不正确: %s, %s", start_date, end_date, extra=LOG_EXTRA)
                return {}
            
            # 转换日期格式
            start_d = date.fromisoformat(start.strftime('%Y-%m-%d'))
            end_d = date.fromisoformat(end.strftime('%Y-%m-%d'))
            
            # 收集数据覆盖情况
            if collect_data_coverage is None:
                LOGGER.error("collect_data_coverage 函数不可用，请检查导入配置", extra=LOG_EXTRA)
                return {}
            
            coverage = collect_data_coverage(start_d, end_d)
            return coverage
        except Exception as exc:
            LOGGER.exception("获取数据覆盖情况失败: %s", exc, extra=LOG_EXTRA)
            return {}

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

# 确保time模块可用
import sys
try:
    import time
except ImportError:
    # 创建一个简单的替代实现
    class TimeStub:
        def time(self):
            return 0
        def sleep(self, seconds):
            pass
    time = TimeStub()
    LOGGER.warning("无法导入time模块，使用替代实现", extra=LOG_EXTRA)
