"""Shared read-only query helpers for database access."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, Optional, Sequence


@dataclass
class BrokerQueryEngine:
    """Lightweight wrapper around standard query patterns."""

    session_factory: Callable[..., object]
    _date_cache: dict = None

    def _find_date_column(self, conn, table: str) -> str | None:
        """Return the best date column for the table or None if none found."""
        if self._date_cache is None:
            self._date_cache = {}
        if table in self._date_cache:
            return self._date_cache[table]
        try:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        except Exception:
            self._date_cache[table] = None
            return None
        cols = [row[1] if isinstance(row, tuple) else row["name"] for row in rows]
        # Prefer canonical 'trade_date'
        if "trade_date" in cols:
            self._date_cache[table] = "trade_date"
            return "trade_date"
        # Prefer any column that ends with '_date'
        for c in cols:
            if isinstance(c, str) and c.endswith("_date"):
                self._date_cache[table] = c
                return c
        # No date-like column
        self._date_cache[table] = None
        return None

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
        with self.session_factory(read_only=True) as conn:
            date_col = self._find_date_column(conn, table)
            if table == "suspend" or date_col is None:
                # For suspend table we prefer to query by ts_code only
                query = f"SELECT {joined_cols} FROM {table} WHERE ts_code = ? ORDER BY rowid DESC LIMIT 1"
                return conn.execute(query, (ts_code,)).fetchone()
            query = (
                f"SELECT {date_col}, {joined_cols} FROM {table} "
                f"WHERE ts_code = ? AND {date_col} <= ? "
                f"ORDER BY {date_col} DESC LIMIT 1"
            )
            return conn.execute(query, (ts_code, trade_date)).fetchone()

    def fetch_series(
        self,
        table: str,
        column: str,
        ts_code: str,
        end_date: str,
        limit: int,
    ) -> List[Mapping[str, object]]:
        with self.session_factory(read_only=True) as conn:
            date_col = self._find_date_column(conn, table)
            if date_col is None:
                # No date column: return most recent rows by rowid
                query = f"SELECT rowid AS trade_date, {column} FROM {table} WHERE ts_code = ? ORDER BY rowid DESC LIMIT ?"
                rows = conn.execute(query, (ts_code, limit)).fetchall()
            else:
                query = (
                    f"SELECT {date_col} AS trade_date, {column} FROM {table} "
                    f"WHERE ts_code = ? AND {date_col} <= ? "
                    f"ORDER BY {date_col} DESC LIMIT ?"
                )
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
