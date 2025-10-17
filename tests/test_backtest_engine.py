from __future__ import annotations

from datetime import date

from app.backtest.engine import BacktestEngine, BtConfig


def test_required_fields_include_precomputed_factors(isolated_db):
    cfg = BtConfig(
        id="bt-test",
        name="bt-test",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
        universe=["000001.SZ"],
        params={},
    )
    engine = BacktestEngine(cfg)
    required = set(engine.required_fields)
    expected_fields = {
        "factors.mom_5",
        "factors.turn_5",
        "factors.val_pe_score",
        "factors.val_pb_score",
        "factors.volume_ratio_score",
        "factors.val_multiscore",
        "factors.risk_penalty",
        "factors.sent_momentum",
        "factors.sent_market",
        "factors.sent_divergence",
    }
    assert expected_fields.issubset(required)
