"""Utility helpers to retrieve structured data slices for agents and departments."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .db import db_session
from .logging import get_logger

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


@dataclass
class DataBroker:
    """Lightweight data access helper for agent/LLM consumption."""

    FIELD_ALIASES: Dict[str, Dict[str, str]] = {
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
        },
        "stk_limit": {
            "up": "up_limit",
            "down": "down_limit",
        },
    }
    MAX_WINDOW: int = 120

    def fetch_latest(
        self,
        ts_code: str,
        trade_date: str,
        fields: Iterable[str],
    ) -> Dict[str, float]:
        """Fetch the latest value (<= trade_date) for each requested field."""

        grouped: Dict[str, List[str]] = {}
        field_map: Dict[Tuple[str, str], List[str]] = {}
        for item in fields:
            if not item:
                continue
            resolved = self.resolve_field(str(item))
            if not resolved:
                continue
            table, column = resolved
            grouped.setdefault(table, [])
            if column not in grouped[table]:
                grouped[table].append(column)
            field_map.setdefault((table, column), []).append(str(item))

        if not grouped:
            return {}

        results: Dict[str, float] = {}
        with db_session(read_only=True) as conn:
            for table, columns in grouped.items():
                joined_cols = ", ".join(columns)
                query = (
                    f"SELECT trade_date, {joined_cols} FROM {table} "
                    "WHERE ts_code = ? AND trade_date <= ? "
                    "ORDER BY trade_date DESC LIMIT 1"
                )
                try:
                    row = conn.execute(query, (ts_code, trade_date)).fetchone()
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
                        results[original] = float(value)
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
        query = (
            f"SELECT trade_date, {resolved} FROM {table} "
            "WHERE ts_code = ? AND trade_date <= ? "
            "ORDER BY trade_date DESC LIMIT ?"
        )
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
        series: List[Tuple[str, float]] = []
        for row in rows:
            value = row[resolved]
            if value is None:
                continue
            series.append((row["trade_date"], float(value)))
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
        return row is not None

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

    def _get_table_columns(self, table: str) -> Optional[set[str]]:
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
        columns = {row["name"] for row in rows if row["name"]}
        cache[table] = columns
        return columns

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
