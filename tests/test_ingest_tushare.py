"""验证 TuShare 拉数流程与因子计算的集成行为。"""
from __future__ import annotations

import sys
import types
from datetime import date

import pytest

# 某些环境下 pandas 可能存在二进制依赖问题，这里提供最小桩避免导入失败
try:  # pragma: no cover - 测试运行环境中若 pandas 可用则直接复用
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - stub fallback
    pandas_stub = types.ModuleType("pandas")

    class _DummyFrame:  # pylint: disable=too-few-public-methods
        empty = True

        def __init__(self, *args, **kwargs):  # noqa: D401
            """轻量占位，避免测试期调用实际逻辑。"""

        def to_dict(self, *_args, **_kwargs):
            return {}

        def reindex(self, *_args, **_kwargs):
            return self

        def where(self, *_args, **_kwargs):
            return self

    pandas_stub.DataFrame = _DummyFrame
    pandas_stub.Series = _DummyFrame
    pandas_stub.concat = lambda *args, **kwargs: _DummyFrame()  # type: ignore[arg-type]
    pandas_stub.Timestamp = lambda *args, **kwargs: None  # type: ignore[assignment]
    pandas_stub.to_datetime = lambda value, **kwargs: value  # type: ignore[assignment]
    pandas_stub.isna = lambda value: False  # type: ignore[assignment]
    pandas_stub.notna = lambda value: True  # type: ignore[assignment]

    sys.modules.setdefault("pandas", pandas_stub)
else:  # pragma: no cover
    sys.modules.setdefault("pandas", _pd)

from app.ingest.tushare import FetchJob, run_ingestion
from app.utils import alerts


@pytest.fixture(autouse=True)
def clear_alerts():
    alerts.clear_warnings()
    yield
    alerts.clear_warnings()


def test_run_ingestion_triggers_factor_range(monkeypatch):
    job = FetchJob(
        name="daily_job",
        start=date(2025, 1, 10),
        end=date(2025, 1, 11),
        ts_codes=("000001.SZ",),
    )

    coverage_called = {}

    def fake_coverage(*args, **kwargs):
        coverage_called["args"] = (args, kwargs)

    monkeypatch.setattr("app.ingest.tushare.ensure_data_coverage", fake_coverage)

    captured: dict = {}

    def fake_compute(start, end, **kwargs):
        captured["start"] = start
        captured["end"] = end
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr("app.ingest.tushare.compute_factor_range", fake_compute)

    run_ingestion(job, include_limits=False)

    assert "args" in coverage_called
    assert captured["start"] == job.start
    assert captured["end"] == job.end
    assert captured["kwargs"] == {"ts_codes": job.ts_codes, "skip_existing": False}


def test_run_ingestion_skips_factors_for_non_daily(monkeypatch):
    job = FetchJob(
        name="weekly_job",
        start=date(2025, 1, 10),
        end=date(2025, 1, 17),
        granularity="weekly",
        ts_codes=None,
    )

    monkeypatch.setattr("app.ingest.tushare.ensure_data_coverage", lambda *_, **__: None)

    invoked = {"count": 0}

    def fake_compute(*args, **kwargs):
        invoked["count"] += 1
        return []

    monkeypatch.setattr("app.ingest.tushare.compute_factor_range", fake_compute)

    run_ingestion(job)

    assert invoked["count"] == 0
