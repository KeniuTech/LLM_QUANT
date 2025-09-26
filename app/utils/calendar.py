"""Trading calendar utilities.

These helpers abstract exchange calendars and trading day lookups. Real
implementations should integrate with TuShare or cached calendars.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, List


def is_trading_day(day: date, holidays: Iterable[date] | None = None) -> bool:
    if day.weekday() >= 5:
        return False
    if holidays and day in set(holidays):
        return False
    return True


def previous_trading_day(day: date, holidays: Iterable[date] | None = None) -> date:
    current = day - timedelta(days=1)
    while not is_trading_day(current, holidays):
        current -= timedelta(days=1)
    return current


def trading_days_between(start: date, end: date, holidays: Iterable[date] | None = None) -> List[date]:
    current = start
    days: List[date] = []
    while current <= end:
        if is_trading_day(current, holidays):
            days.append(current)
        current += timedelta(days=1)
    return days
