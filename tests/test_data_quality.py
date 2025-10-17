from __future__ import annotations

from app.utils.data_access import DataBroker
from app.utils.data_quality import DataQualityResult, summarize_data_quality


def test_summarize_data_quality_produces_score():
    results = [
        DataQualityResult("check_a", "ERROR", "fatal issue"),
        DataQualityResult("check_b", "WARN", "warning issue"),
        DataQualityResult("check_c", "INFO", "info message"),
    ]

    summary = summarize_data_quality(results, window_days=7)

    assert summary.total_checks == 3
    assert summary.severity_counts["ERROR"] == 1
    assert summary.has_blockers is True
    assert 0.0 <= summary.score < 100.0


def test_data_broker_evaluate_quality_runs_checks(isolated_db):
    broker = DataBroker()
    summary = broker.evaluate_data_quality(window_days=1)

    assert 0.0 <= summary.score <= 100.0
    assert summary.window_days == 1
