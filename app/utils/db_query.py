"""Shared read-only query helpers for database access."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, Optional, Sequence


@dataclass
class BrokerQueryEngine:
    """Lightweight wrapper around standard query patterns."""

    session_factory: Callable[..., object]

    def fetch_latest(
        self,
        table: str,
        ts_code: str,
        trade_date: str,
        columns: Sequence[str],
    ) -> Optional[Mapping[str, object]]:
        if not columns:
            return None
        joined_cols = ", ".join(columns)
        query = (
            f"SELECT trade_date, {joined_cols} FROM {table} "
            "WHERE ts_code = ? AND trade_date <= ? "
            "ORDER BY trade_date DESC LIMIT 1"
        )
        with self.session_factory(read_only=True) as conn:
            return conn.execute(query, (ts_code, trade_date)).fetchone()

    def fetch_series(
        self,
        table: str,
        column: str,
        ts_code: str,
        end_date: str,
        limit: int,
    ) -> List[Mapping[str, object]]:
        query = (
            f"SELECT trade_date, {column} FROM {table} "
            "WHERE ts_code = ? AND trade_date <= ? "
            "ORDER BY trade_date DESC LIMIT ?"
        )
        with self.session_factory(read_only=True) as conn:
            rows = conn.execute(query, (ts_code, end_date, limit)).fetchall()
        return list(rows)

    def fetch_table(
        self,
        table: str,
        columns: Iterable[str],
        ts_code: str,
        trade_date: Optional[str],
        limit: int,
    ) -> List[Mapping[str, object]]:
        cols = ", ".join(columns)
        if trade_date is None:
            query = (
                f"SELECT {cols} FROM {table} "
                "WHERE ts_code = ? ORDER BY rowid DESC LIMIT ?"
            )
            params: Sequence[object] = (ts_code, limit)
        else:
            query = (
                f"SELECT {cols} FROM {table} "
                "WHERE ts_code = ? AND trade_date <= ? "
                "ORDER BY trade_date DESC LIMIT ?"
            )
            params = (ts_code, trade_date, limit)
        with self.session_factory(read_only=True) as conn:
            rows = conn.execute(query, params).fetchall()
        return list(rows)
