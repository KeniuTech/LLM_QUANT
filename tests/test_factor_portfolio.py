from __future__ import annotations

from datetime import date, timedelta

from app.features.evaluation import (
    FactorPerformance,
    evaluate_factor_portfolio,
    optimize_factor_weights,
)
from app.features.factors import FactorSpec, compute_factor_range
from tests.factor_utils import populate_sample_data


def _seed_factor_history(codes, end_day):
    specs = [
        FactorSpec("mom_5", 5),
        FactorSpec("mom_20", 20),
        FactorSpec("turn_20", 20),
    ]
    start_day = end_day - timedelta(days=5)
    for code in codes:
        populate_sample_data(code, end_day, days=180)
    compute_factor_range(start_day, end_day, ts_codes=codes, factors=specs)
    return specs, start_day


def test_optimize_factor_weights_returns_normalized_vector(isolated_db):
    codes = [f"0000{i:02d}.SZ" for i in range(1, 4)]
    end_day = date(2025, 2, 28)
    specs, start_day = _seed_factor_history(codes, end_day)
    factor_names = [spec.name for spec in specs]

    weights, performances = optimize_factor_weights(
        factor_names,
        start_day,
        end_day,
        universe=codes,
    )

    assert set(weights.keys()) == set(factor_names)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    for perf in performances.values():
        assert isinstance(perf, FactorPerformance)


def test_evaluate_factor_portfolio_returns_report(isolated_db):
    codes = [f"0000{i:02d}.SZ" for i in range(1, 4)]
    end_day = date(2025, 3, 10)
    specs, start_day = _seed_factor_history(codes, end_day)
    factor_names = [spec.name for spec in specs]

    report = evaluate_factor_portfolio(
        factor_names,
        start_day,
        end_day,
        universe=codes,
    )

    assert set(report.weights.keys()) == set(factor_names)
    assert isinstance(report.combined, FactorPerformance)
    assert report.combined.sample_size >= 0
    assert set(report.components.keys()) == set(factor_names)
