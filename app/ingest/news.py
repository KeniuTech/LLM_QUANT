"""Unified news ingestion orchestration with GDELT as the primary source."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Set, Tuple

from app.data.schema import initialize_database
from app.utils.logging import get_logger

from .gdelt import ingest_configured_gdelt

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "news_ingest"}

_PREPARED_WINDOWS: Set[Tuple[str, int]] = set()


def _normalize_date(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


def ingest_latest_news(
    *,
    days_back: int = 1,
    force: bool = False,
) -> int:
    """Fetch latest news primarily via GDELT within a day-level window."""

    initialize_database()
    now = datetime.utcnow()
    days = max(days_back, 1)
    start_day = (now.date() - timedelta(days=days - 1))
    start_dt = datetime.combine(start_day, datetime.min.time())
    end_dt = datetime.combine(now.date(), datetime.max.time())
    LOGGER.info(
        "触发 GDELT 新闻拉取 days=%s force=%s",
        days,
        force,
        extra=LOG_EXTRA,
    )
    inserted = ingest_configured_gdelt(
        start=start_dt,
        end=end_dt,
        incremental=not force,
    )
    LOGGER.info("新闻拉取完成 inserted=%s", inserted, extra=LOG_EXTRA)
    return inserted


def ensure_news_range(
    start: date | datetime,
    end: date | datetime,
    *,
    force: bool = False,
) -> int:
    """Ensure the news store covers the requested window."""

    initialize_database()
    start_dt = _normalize_date(start)
    end_dt = _normalize_date(end)
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt
    start_dt = datetime.combine(start_dt.date(), datetime.min.time())
    end_dt = datetime.combine(end_dt.date(), datetime.max.time())
    LOGGER.info(
        "同步 GDELT 新闻数据 start=%s end=%s force=%s",
        start_dt.isoformat(),
        end_dt.isoformat(),
        force,
        extra=LOG_EXTRA,
    )
    inserted = ingest_configured_gdelt(
        start=start_dt,
        end=end_dt,
        incremental=not force,
    )
    LOGGER.info(
        "新闻窗口同步完成 inserted=%s start=%s end=%s",
        inserted,
        start_dt.isoformat(),
        end_dt.isoformat(),
        extra=LOG_EXTRA,
    )
    return inserted


def prepare_news_for_factors(
    trade_date: date,
    *,
    lookback_days: int = 3,
    force: bool = False,
) -> int:
    """Prepare news data before sentiment factor computation."""

    key = (trade_date.strftime("%Y%m%d"), max(lookback_days, 1))
    if not force and key in _PREPARED_WINDOWS:
        LOGGER.debug(
            "新闻窗口已准备完成 trade_date=%s lookback=%s",
            key[0],
            key[1],
            extra=LOG_EXTRA,
        )
        return 0

    end_date = trade_date
    start_date = trade_date - timedelta(days=max(lookback_days - 1, 0))
    inserted = ensure_news_range(start_date, end_date, force=force)
    if not force:
        _PREPARED_WINDOWS.add(key)
    return inserted


__all__ = [
    "ensure_news_range",
    "ingest_latest_news",
    "prepare_news_for_factors",
]
