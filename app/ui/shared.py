"""Shared utilities and constants for Streamlit UI views."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import streamlit as st

from app.utils.db import db_session
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "ui"}


def get_query_params() -> dict[str, list[str]]:
    """Safely read URL query parameters from Streamlit."""
    try:
        return dict(st.query_params)
    except Exception:  # noqa: BLE001
        return {}


def set_query_params(**kwargs: object) -> None:
    """Update URL query parameters, ignoring failures in unsupported contexts."""
    try:
        payload = {k: v for k, v in kwargs.items() if v is not None}
        if payload:
            st.query_params.update(payload)
    except Exception:  # noqa: BLE001
        pass


def get_latest_trade_date() -> Optional[date]:
    """Fetch the most recent trade date from the database."""
    try:
        with db_session(read_only=True) as conn:
            row = conn.execute(
                "SELECT trade_date FROM daily ORDER BY trade_date DESC LIMIT 1"
            ).fetchone()
    except Exception:  # noqa: BLE001
        LOGGER.exception("查询最新交易日失败", extra=LOG_EXTRA)
        return None
    if not row:
        return None
    raw_value = row["trade_date"]
    if not raw_value:
        return None
    try:
        return datetime.strptime(str(raw_value), "%Y%m%d").date()
    except ValueError:
        try:
            return datetime.fromisoformat(str(raw_value)).date()
        except ValueError:
            LOGGER.warning("无法解析交易日：%s", raw_value, extra=LOG_EXTRA)
            return None


def default_backtest_range(window_days: int = 60) -> tuple[date, date]:
    """Return a sensible (end, start) date range for backtests."""
    latest = get_latest_trade_date() or date.today()
    start = latest - timedelta(days=window_days)
    if start > latest:
        start = latest
    return start, latest
