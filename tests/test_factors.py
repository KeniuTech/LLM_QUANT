"""Tests for factor computation pipeline."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.core.indicators import momentum, rolling_mean, volatility
from app.data.schema import initialize_database
from app.features.factors import (
    DEFAULT_FACTORS,
    FactorResult,
    FactorSpec,
    compute_factor_range,
    compute_factors,
)
from app.utils.config import DataPaths, get_config
from app.utils.data_access import DataBroker
from app.utils.db import db_session


@pytest.fixture()
def isolated_db(tmp_path):
    cfg = get_config()
    original_paths = cfg.data_paths
    tmp_root = tmp_path / "data"
    tmp_root.mkdir(parents=True, exist_ok=True)
    cfg.data_paths = DataPaths(root=tmp_root)
    try:
        yield
    finally:
        cfg.data_paths = original_paths


def _populate_sample_data(ts_code: str, as_of: date) -> None:
    initialize_database()
    with db_session() as conn:
        for offset in range(60):
            current_day = as_of - timedelta(days=offset)
            trade_date = current_day.strftime("%Y%m%d")
            close = 100 + (59 - offset)
            turnover = 5 + 0.1 * (59 - offset)
            conn.execute(
                """
                INSERT OR REPLACE INTO daily
                (ts_code, trade_date, open, high, low, close, pct_chg, vol, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_code,
                    trade_date,
                    close,
                    close,
                    close,
                    close,
                    0.0,
                    1000.0,
                    1_000_000.0,
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_basic
                (ts_code, trade_date, turnover_rate, turnover_rate_f, volume_ratio)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ts_code,
                    trade_date,
                    turnover,
                    turnover,
                    1.0,
                ),
            )


def test_compute_factors_persists_and_updates(isolated_db):
    ts_code = "000001.SZ"
    trade_day = date(2025, 1, 30)
    _populate_sample_data(ts_code, trade_day)

    specs = [*DEFAULT_FACTORS, FactorSpec("mom_5", 5)]
    results = compute_factors(trade_day, specs)

    assert results
    result_map = {result.ts_code: result for result in results}
    assert ts_code in result_map
    result: FactorResult = result_map[ts_code]

    close_series = [100 + (59 - offset) for offset in range(60)]
    turnover_series = [5 + 0.1 * (59 - offset) for offset in range(60)]

    expected_mom20 = momentum(close_series, 20)
    expected_mom60 = momentum(close_series, 60)
    expected_mom5 = momentum(close_series, 5)
    expected_volat20 = volatility(close_series, 20)
    expected_turn20 = rolling_mean(turnover_series, 20)

    assert result.values["mom_20"] == pytest.approx(expected_mom20)
    assert result.values["mom_60"] == pytest.approx(expected_mom60)
    assert result.values["mom_5"] == pytest.approx(expected_mom5)
    assert result.values["volat_20"] == pytest.approx(expected_volat20)
    assert result.values["turn_20"] == pytest.approx(expected_turn20)

    trade_date_str = trade_day.strftime("%Y%m%d")
    with db_session(read_only=True) as conn:
        row = conn.execute(
            """
            SELECT mom_20, mom_60, mom_5, volat_20, turn_20
            FROM factors WHERE ts_code = ? AND trade_date = ?
            """,
            (ts_code, trade_date_str),
        ).fetchone()
    assert row is not None
    assert row["mom_20"] == pytest.approx(expected_mom20)
    assert row["mom_60"] == pytest.approx(expected_mom60)
    assert row["mom_5"] == pytest.approx(expected_mom5)
    assert row["volat_20"] == pytest.approx(expected_volat20)
    assert row["turn_20"] == pytest.approx(expected_turn20)

    broker = DataBroker()
    latest = broker.fetch_latest(ts_code, trade_date_str, ["factors.mom_5", "factors.turn_20"])
    assert latest["factors.mom_5"] == pytest.approx(expected_mom5)
    assert latest["factors.turn_20"] == pytest.approx(expected_turn20)

    # Calling compute_factors again should update existing rows without error.
    second_results = compute_factors(trade_day, specs)
    assert second_results
    assert broker.fetch_latest(ts_code, trade_date_str, ["factors.mom_20"])["factors.mom_20"] == pytest.approx(
        expected_mom20
    )


def test_compute_factors_skip_existing(isolated_db):
    ts_code = "000001.SZ"
    trade_day = date(2025, 2, 10)
    _populate_sample_data(ts_code, trade_day)

    compute_factors(trade_day)
    skipped = compute_factors(trade_day, skip_existing=True)
    assert skipped == []


def test_compute_factor_range_filters_universe(isolated_db):
    code_a = "000001.SZ"
    code_b = "000002.SZ"
    end_day = date(2025, 3, 5)
    start_day = end_day - timedelta(days=1)

    _populate_sample_data(code_a, end_day)
    _populate_sample_data(code_b, end_day)

    results = compute_factor_range(start_day, end_day, ts_codes=[code_a])
    assert results
    assert {result.ts_code for result in results} == {code_a}

    with db_session(read_only=True) as conn:
        rows = conn.execute("SELECT DISTINCT ts_code FROM factors").fetchall()
    assert {row["ts_code"] for row in rows} == {code_a}

    repeated = compute_factor_range(start_day, end_day, ts_codes=[code_a])
    assert repeated == []
