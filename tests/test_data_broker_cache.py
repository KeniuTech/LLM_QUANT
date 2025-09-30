"""验证 DataBroker 的缓存与回退行为。"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Tuple

import pytest

from app.utils import data_access
from app.utils.data_access import DataBroker


class _FakeRow(dict):
    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        return dict.get(self, key)


class _FakeCursor:
    def __init__(self, rows: Iterable[Dict[str, Any]]):
        self._rows = list(rows)

    def fetchone(self) -> Dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> List[Dict[str, Any]]:
        return list(self._rows)


class _FakeConn:
    def __init__(self, calls: List[str], failure_flags: Dict[str, bool]) -> None:
        self._calls = calls
        self._failure_flags = failure_flags
        self._daily_series = [
            _FakeRow({"trade_date": "20250112", "close": 12.3, "open": 12.1}),
            _FakeRow({"trade_date": "20250111", "close": 11.9, "open": 11.6}),
        ]
        self._turn_series = [
            _FakeRow({"trade_date": "20250112", "turnover_rate": 2.5}),
            _FakeRow({"trade_date": "20250111", "turnover_rate": 2.2}),
        ]

    def execute(self, query: str, params: Tuple[Any, ...] | None = None):
        self._calls.append(query)
        params = params or ()
        upper = query.upper()
        if upper.startswith("PRAGMA TABLE_INFO"):
            if "DAILY_BASIC" in upper:
                rows = [_FakeRow({"name": "trade_date"}), _FakeRow({"name": "turnover_rate"})]
            else:
                rows = [_FakeRow({"name": "trade_date"}), _FakeRow({"name": "close"}), _FakeRow({"name": "open"})]
            return _FakeCursor(rows)
        if "FROM DAILY " in upper:
            if self._failure_flags.get("daily"):
                raise sqlite3.OperationalError("stub failure")
            if "LIMIT 1" in upper:
                return _FakeCursor([self._daily_series[0]])
            limit = params[-1] if params else len(self._daily_series)
            return _FakeCursor(self._daily_series[: int(limit)])
        if "FROM DAILY_BASIC" in upper:
            limit = params[-1] if params else len(self._turn_series)
            return _FakeCursor(self._turn_series[: int(limit)])
        raise AssertionError(f"Unexpected query: {query}")

    def close(self) -> None:  # pragma: no cover - compatibility
        return None


@pytest.fixture()
def patched_db(monkeypatch):
    calls: List[str] = []
    failure_flags: Dict[str, bool] = defaultdict(bool)

    @contextmanager
    def _session(read_only: bool = False):  # noqa: D401 - contextmanager stub
        conn = _FakeConn(calls, failure_flags)
        yield conn

    monkeypatch.setattr(data_access, "db_session", _session)
    yield calls, failure_flags


def test_fetch_latest_uses_cache(patched_db):
    calls, failure = patched_db
    broker = DataBroker()

    result = broker.fetch_latest("000001.SZ", "20250112", ["daily.close", "daily.open"])
    assert result["daily.close"] == pytest.approx(12.3)
    first_count = len(calls)

    result_cached = broker.fetch_latest("000001.SZ", "20250112", ["daily.open", "daily.close"])
    assert result_cached == result
    assert len(calls) == first_count

    failure["daily"] = True
    still_cached = broker.fetch_latest("000001.SZ", "20250112", ["daily.close", "daily.open"])
    assert still_cached["daily.close"] == pytest.approx(12.3)
    assert len(calls) == first_count


def test_fetch_series_cache_and_disable(patched_db):
    calls, _ = patched_db
    broker = DataBroker(series_cache_size=4)

    series = broker.fetch_series("daily", "close", "000001.SZ", "20250112", 2)
    assert len(series) == 2
    first_count = len(calls)

    series_cached = broker.fetch_series("daily", "close", "000001.SZ", "20250112", 2)
    assert series_cached == series
    assert len(calls) == first_count

    broker_no_cache = DataBroker(enable_cache=False)
    calls_before = len(calls)
    broker_no_cache.fetch_series("daily", "close", "000001.SZ", "20250112", 2)
    assert len(calls) > calls_before
