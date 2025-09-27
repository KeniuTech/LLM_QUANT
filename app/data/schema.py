"""Database schema management for the investment assistant."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, List

from app.utils.db import db_session


SCHEMA_STATEMENTS: Iterable[str] = (
    """
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
    """
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
    """
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
    """
    CREATE TABLE IF NOT EXISTS adj_factor (
      ts_code TEXT,
      trade_date TEXT,
      adj_factor REAL,
      PRIMARY KEY (ts_code, trade_date)
    );
    """,
    """
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
    """
    CREATE TABLE IF NOT EXISTS trade_calendar (
      exchange TEXT,
      cal_date TEXT,
      is_open INTEGER,
      pretrade_date TEXT,
      PRIMARY KEY (exchange, cal_date)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS stk_limit (
      ts_code TEXT,
      trade_date TEXT,
      up_limit REAL,
      down_limit REAL,
      PRIMARY KEY (ts_code, trade_date)
    );
    """,
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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

REQUIRED_TABLES = (
    "stock_basic",
    "daily",
    "daily_basic",
    "adj_factor",
    "suspend",
    "trade_calendar",
    "stk_limit",
    "index_basic",
    "index_daily",
    "fund_basic",
    "fund_nav",
    "fut_basic",
    "fut_daily",
    "fx_daily",
    "hk_daily",
    "us_daily",
    "news",
    "heat_daily",
    "bt_config",
    "bt_trades",
    "bt_nav",
    "bt_report",
    "run_log",
    "agent_utils",
    "alloc_log",
)


@dataclass
class MigrationResult:
    executed: int
    skipped: bool = False
    missing_tables: List[str] = field(default_factory=list)


def _missing_tables() -> List[str]:
    try:
        with db_session(read_only=True) as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    except sqlite3.OperationalError:
        return list(REQUIRED_TABLES)
    existing = {row["name"] for row in rows}
    return [name for name in REQUIRED_TABLES if name not in existing]


def initialize_database() -> MigrationResult:
    """Create tables and indexes required by the application."""

    missing = _missing_tables()
    if not missing:
        return MigrationResult(executed=0, skipped=True, missing_tables=[])

    executed = 0
    with db_session() as conn:
        cursor = conn.cursor()
        for statement in SCHEMA_STATEMENTS:
            cursor.executescript(statement)
            executed += 1
    return MigrationResult(executed=executed, skipped=False, missing_tables=missing)
