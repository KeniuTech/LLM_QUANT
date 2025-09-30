"""针对 scripts/run_ingestion_job.py 的 CLI 行为测试。"""
from __future__ import annotations

import pytest

import scripts.run_ingestion_job as cli


@pytest.fixture(autouse=True)
def reset_alerts(monkeypatch):
    monkeypatch.setattr(cli.alerts, "clear_warnings", lambda *args, **kwargs: None)
    yield


def test_cli_invokes_run_ingestion_with_codes(monkeypatch):
    captured: dict = {}

    def fake_run(job, include_limits):
        captured["job"] = job
        captured["include_limits"] = include_limits

    monkeypatch.setattr(cli, "run_ingestion", fake_run)
    monkeypatch.setattr(cli.alerts, "get_warnings", lambda: [])

    exit_code = cli.run_cli(
        [
            "20250110",
            "20250112",
            "--codes",
            "000001.SZ",
            "000002.SZ",
            "--include-limits",
            "--name",
            "test_job",
        ]
    )

    assert exit_code == 0
    job = captured["job"]
    assert job.name == "test_job"
    assert job.start.isoformat() == "2025-01-10"
    assert job.end.isoformat() == "2025-01-12"
    assert job.ts_codes == ("000001.SZ", "000002.SZ")
    assert captured["include_limits"] is True


def test_cli_returns_warning_status(monkeypatch):
    monkeypatch.setattr(cli, "run_ingestion", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli.alerts,
        "get_warnings",
        lambda: [{"source": "Factors", "message": "mock warning"}],
    )

    exit_code = cli.run_cli(["20250101", "20250102"])

    assert exit_code == 2


def test_cli_validates_date_order(monkeypatch):
    with pytest.raises(SystemExit):
        cli.run_cli(["20250105", "20250101"])
