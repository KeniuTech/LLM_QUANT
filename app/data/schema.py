"""Database schema management for the investment assistant."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable

from app.utils.config import get_config
from app.utils.db import db_session


SCHEMA_STATEMENTS: Iterable[str] = (
    """
    CREATE TABLE IF NOT EXISTS news (
      id TEXT PRIMARY KEY,
      ts_code TEXT,
      pub_time TEXT,
      source TEXT,
      title TEXT,
      summary TEXT,
      url TEXT,
      entities TEXT,
      sentiment REAL,
      heat REAL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_news_time ON news(pub_time DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_news_code ON news(ts_code, pub_time DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS heat_daily (
      scope TEXT,
      key TEXT,
      trade_date TEXT,
      heat REAL,
      top_topics TEXT,
      PRIMARY KEY (scope, key, trade_date)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS bt_config (
      id TEXT PRIMARY KEY,
      name TEXT,
      start_date TEXT,
      end_date TEXT,
      universe TEXT,
      params TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS bt_trades (
      cfg_id TEXT,
      ts_code TEXT,
      trade_date TEXT,
      side TEXT,
      price REAL,
      qty REAL,
      reason TEXT,
      PRIMARY KEY (cfg_id, ts_code, trade_date, side)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS bt_nav (
      cfg_id TEXT,
      trade_date TEXT,
      nav REAL,
      ret REAL,
      pos_count INTEGER,
      turnover REAL,
      dd REAL,
      info TEXT,
      PRIMARY KEY (cfg_id, trade_date)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS bt_report (
      cfg_id TEXT PRIMARY KEY,
      summary TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS run_log (
      ts TEXT PRIMARY KEY,
      stage TEXT,
      level TEXT,
      msg TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_utils (
      trade_date TEXT,
      ts_code TEXT,
      agent TEXT,
      action TEXT,
      utils TEXT,
      feasible TEXT,
      weight REAL,
      PRIMARY KEY (trade_date, ts_code, agent)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS alloc_log (
      trade_date TEXT,
      ts_code TEXT,
      target_weight REAL,
      clipped_weight REAL,
      reason TEXT,
      PRIMARY KEY (trade_date, ts_code)
    );
    """
)


@dataclass
class MigrationResult:
    executed: int
    skipped: bool = False


def _schema_exists() -> bool:
    try:
        with db_session(read_only=True) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='news'"
            )
            return cursor.fetchone() is not None
    except sqlite3.OperationalError:
        return False


def initialize_database() -> MigrationResult:
    """Create tables and indexes required by the application."""

    if _schema_exists():
        return MigrationResult(executed=0, skipped=True)

    executed = 0
    with db_session() as conn:
        cursor = conn.cursor()
        for statement in SCHEMA_STATEMENTS:
            cursor.executescript(statement)
            executed += 1
    return MigrationResult(executed=executed)
