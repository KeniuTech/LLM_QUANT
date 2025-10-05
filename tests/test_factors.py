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
    _valuation_score,
    _volume_ratio_score,
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
            pe = 10.0 + (offset % 5)
            pb = 1.5 + (offset % 3) * 0.1
            ps = 2.0 + (offset % 4) * 0.1
            volume_ratio = 0.5 + (offset % 4) * 0.5
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_basic
                (ts_code, trade_date, turnover_rate, turnover_rate_f, volume_ratio, pe, pb, ps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_code,
                    trade_date,
                    turnover,
                    turnover,
                    volume_ratio,
                    pe,
                    pb,
                    ps,
                ),
            )


def test_compute_factors_persists_and_updates(isolated_db):
    ts_code = "000001.SZ"
    trade_day = date(2025, 1, 30)
    _populate_sample_data(ts_code, trade_day)

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


def test_compute_extended_factors(isolated_db):
    """Test computation of extended factors."""
    # Use the existing _populate_sample_data function
    from app.utils.data_access import DataBroker
    broker = DataBroker()
    
    # Sample data for 5 trading days
    dates = ["20240101", "20240102", "20240103", "20240104", "20240105"]
    ts_codes = ["000001.SZ", "000002.SZ", "600000.SH"]

    # Populate daily data
    for ts_code in ts_codes:
        for i, trade_date in enumerate(dates):
            broker.insert_or_update_daily(
                ts_code,
                trade_date,
                open_price=10.0 + i * 0.1,
                high=10.5 + i * 0.1,
                low=9.5 + i * 0.1,
                close=10.0 + i * 0.2,  # 上涨趋势
                pre_close=10.0 + (i - 1) * 0.2 if i > 0 else 10.0,
                vol=100000 + i * 10000,
                amount=1000000 + i * 100000,
            )

            broker.insert_or_update_daily_basic(
                ts_code,
                trade_date,
                close=10.0 + i * 0.2,
                turnover_rate=1.0 + i * 0.1,
                turnover_rate_f=1.0 + i * 0.1,
                volume_ratio=1.0 + (i % 3) * 0.2,  # 在0.8-1.2之间变化
                pe=15.0 + (i % 3) * 2,  # 在15-19之间变化
                pe_ttm=15.0 + (i % 3) * 2,
                pb=1.5 + (i % 3) * 0.1,  # 在1.5-1.7之间变化
                ps=3.0 + (i % 3) * 0.2,  # 在3.0-3.4之间变化
                ps_ttm=3.0 + (i % 3) * 0.2,
                dv_ratio=2.0 + (i % 3) * 0.1,  # 股息率
                total_mv=1000000 + i * 100000,
                circ_mv=800000 + i * 80000,
            )
    
    # Compute factors with extended factors
    from app.features.extended_factors import EXTENDED_FACTORS
    all_factors = list(DEFAULT_FACTORS) + EXTENDED_FACTORS
    
    trade_day = date(2024, 1, 5)
    results = compute_factors(trade_day, all_factors)
    
    # Verify that we got results
    assert results
    
    # Verify that extended factors are computed
    result_map = {result.ts_code: result for result in results}
    ts_code = "000001.SZ"
    assert ts_code in result_map
    result = result_map[ts_code]
    
    # Check that extended factors are present in the results
    extended_factor_names = [spec.name for spec in EXTENDED_FACTORS]
    for factor_name in extended_factor_names:
        assert factor_name in result.values
        # Values should not be None
        assert result.values[factor_name] is not None
