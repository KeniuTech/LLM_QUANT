"""Tests for factor computation pipeline."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.core.indicators import momentum, rolling_mean, volatility
from app.features.factors import (
    DEFAULT_FACTORS,
    FactorResult,
    FactorSpec,
    compute_factor_range,
    compute_factors_incremental,
    compute_factors,
    _valuation_score,
    _volume_ratio_score,
)
from app.utils.data_access import DataBroker
from app.utils.db import db_session
from tests.factor_utils import populate_sample_data


def test_compute_factors_persists_and_updates(isolated_db):
    ts_code = "000001.SZ"
    trade_day = date(2025, 1, 30)
    populate_sample_data(ts_code, trade_day)

    specs = [
        *DEFAULT_FACTORS,
        FactorSpec("mom_5", 5),
        FactorSpec("turn_5", 5),
        FactorSpec("val_pe_score", 0),
        FactorSpec("val_pb_score", 0),
        FactorSpec("volume_ratio_score", 0),
    ]
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
    expected_turn5 = rolling_mean(turnover_series, 5)
    latest_pe = 10.0 + (0 % 5)
    latest_pb = 1.5 + (0 % 3) * 0.1
    latest_volume_ratio = 0.5 + (0 % 4) * 0.5
    expected_val_pe = _valuation_score(latest_pe, scale=12.0)
    expected_val_pb = _valuation_score(latest_pb, scale=2.5)
    expected_vol_ratio_score = _volume_ratio_score(latest_volume_ratio)

    assert result.values["mom_20"] == pytest.approx(expected_mom20)
    assert result.values["mom_60"] == pytest.approx(expected_mom60)
    assert result.values["mom_5"] == pytest.approx(expected_mom5)
    assert result.values["volat_20"] == pytest.approx(expected_volat20)
    assert result.values["turn_20"] == pytest.approx(expected_turn20)
    assert result.values["turn_5"] == pytest.approx(expected_turn5)
    assert result.values["val_pe_score"] == pytest.approx(expected_val_pe)
    assert result.values["val_pb_score"] == pytest.approx(expected_val_pb)
    assert result.values["volume_ratio_score"] == pytest.approx(expected_vol_ratio_score)

    trade_date_str = trade_day.strftime("%Y%m%d")
    with db_session(read_only=True) as conn:
        row = conn.execute(
            """
            SELECT mom_20, mom_60, mom_5, volat_20, turn_20, turn_5, val_pe_score, val_pb_score, volume_ratio_score
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
    assert row["turn_5"] == pytest.approx(expected_turn5)
    assert row["val_pe_score"] == pytest.approx(expected_val_pe)
    assert row["val_pb_score"] == pytest.approx(expected_val_pb)
    assert row["volume_ratio_score"] == pytest.approx(expected_vol_ratio_score)

    broker = DataBroker()
    latest = broker.fetch_latest(
        ts_code,
        trade_date_str,
        [
            "factors.mom_5",
            "factors.turn_20",
            "factors.turn_5",
            "factors.val_pe_score",
            "factors.val_pb_score",
            "factors.volume_ratio_score",
        ],
    )
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
    populate_sample_data(ts_code, trade_day)

    basic_specs = [
        FactorSpec("mom_5", 5),
        FactorSpec("mom_20", 20),
        FactorSpec("volat_20", 20),
        FactorSpec("turn_5", 5),
    ]
    compute_factors(trade_day, basic_specs)
    skipped = compute_factors(trade_day, basic_specs, skip_existing=True)
    assert skipped == []


def test_compute_factors_dry_run(isolated_db):
    ts_code = "000001.SZ"
    trade_day = date(2025, 2, 12)
    populate_sample_data(ts_code, trade_day)

    results = compute_factors(trade_day, persist=False)
    assert results

    trade_date_str = trade_day.strftime("%Y%m%d")
    with db_session(read_only=True) as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM factors WHERE trade_date = ?",
            (trade_date_str,),
        ).fetchone()
    assert count["cnt"] == 0


def test_compute_factors_incremental(isolated_db):
    ts_code = "000001.SZ"
    latest_day = date(2025, 2, 10)
    populate_sample_data(ts_code, latest_day, days=180)

    first_day = latest_day - timedelta(days=1)
    basic_specs = [
        FactorSpec("mom_5", 5),
        FactorSpec("mom_20", 20),
        FactorSpec("turn_20", 20),
    ]
    compute_factors(first_day, basic_specs)

    summary = compute_factors_incremental(factors=basic_specs, max_trading_days=3)
    trade_dates = summary["trade_dates"]
    assert trade_dates
    assert trade_dates[0] > first_day
    assert summary["count"] > 0

    # No new dates should return empty result
    summary_again = compute_factors_incremental(factors=basic_specs, max_trading_days=3)
    assert summary_again["count"] == 0


def test_compute_factor_range_filters_universe(isolated_db):
    code_a = "000001.SZ"
    code_b = "000002.SZ"
    end_day = date(2025, 3, 5)
    start_day = end_day - timedelta(days=1)

    populate_sample_data(code_a, end_day)
    populate_sample_data(code_b, end_day)

    basic_specs = [
        FactorSpec("mom_5", 5),
        FactorSpec("mom_20", 20),
        FactorSpec("turn_20", 20),
    ]

    results = compute_factor_range(start_day, end_day, ts_codes=[code_a], factors=basic_specs)
    assert results
    assert {result.ts_code for result in results} == {code_a}

    with db_session(read_only=True) as conn:
        rows = conn.execute("SELECT DISTINCT ts_code FROM factors").fetchall()
    assert {row["ts_code"] for row in rows} == {code_a}

    repeated = compute_factor_range(
        start_day,
        end_day,
        ts_codes=[code_a],
        factors=basic_specs,
        skip_existing=True,
    )
    assert repeated == []


def test_compute_extended_factors(isolated_db):
    """Extended factors should be persisted alongside base factors."""
    from app.features.extended_factors import EXTENDED_FACTORS

    trade_day = date(2025, 2, 28)
    ts_codes = ["000001.SZ", "000002.SZ"]
    for code in ts_codes:
        populate_sample_data(code, trade_day, days=120)

    all_factors = list(DEFAULT_FACTORS) + EXTENDED_FACTORS
    results = compute_factors(trade_day, all_factors)

    assert results
    result_map = {result.ts_code: result for result in results}
    for code in ts_codes:
        assert code in result_map
        factor_payload = result_map[code].values
        required_extended = {
            "tech_rsi_14",
            "tech_macd_signal",
            "trend_ma_cross",
            "micro_trade_imbalance",
        }
        assert required_extended.issubset(factor_payload.keys())
        for name in required_extended:
            assert factor_payload.get(name) is not None
