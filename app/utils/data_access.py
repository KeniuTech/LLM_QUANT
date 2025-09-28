"""Utility helpers to retrieve structured data slices for agents and departments."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

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

    def fetch_latest(
        self,
        ts_code: str,
        trade_date: str,
        fields: Iterable[str],
    ) -> Dict[str, float]:
        """Fetch the latest value (<= trade_date) for each requested field."""

        grouped: Dict[str, List[str]] = {}
        for item in fields:
            if not item:
                continue
            normalized = _safe_split(str(item))
            if not normalized:
                continue
            table, column = normalized
            grouped.setdefault(table, [])
            if column not in grouped[table]:
                grouped[table].append(column)

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
                    key = f"{table}.{column}"
                    results[key] = float(value)
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
        if not (_is_safe_identifier(table) and _is_safe_identifier(column)):
            return []
        query = (
            f"SELECT trade_date, {column} FROM {table} "
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
            value = row[column]
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
