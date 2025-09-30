"""Verify BacktestEngine consumes persisted factor fields."""
from __future__ import annotations

from datetime import date

import pytest

from app.backtest.engine import BacktestEngine, BtConfig


@pytest.fixture()
def engine(monkeypatch):
    cfg = BtConfig(
        id="test",
        name="factor",
        start_date=date(2025, 1, 10),
        end_date=date(2025, 1, 10),
        universe=["000001.SZ"],
        params={},
    )
    engine = BacktestEngine(cfg)

    def fake_fetch_latest(ts_code, trade_date, fields):  # noqa: D401
        assert "factors.mom_20" in fields
        return {
            "daily.close": 10.0,
            "daily.pct_chg": 0.02,
            "daily_basic.turnover_rate": 5.0,
            "daily_basic.volume_ratio": 15.0,
            "factors.mom_20": 0.12,
            "factors.mom_60": 0.25,
            "factors.volat_20": 0.05,
            "factors.turn_20": 3.0,
            "news.sentiment_index": 0.3,
            "news.heat_score": 0.4,
            "macro.industry_heat": 0.6,
            "macro.relative_strength": 0.7,
        }

    monkeypatch.setattr(engine.data_broker, "fetch_latest", fake_fetch_latest)
    monkeypatch.setattr(engine.data_broker, "fetch_series", lambda *args, **kwargs: [])
    monkeypatch.setattr(engine.data_broker, "fetch_flags", lambda *args, **kwargs: False)

    return engine


def test_load_market_data_prefers_factors(engine):
    data = engine.load_market_data(date(2025, 1, 10))
    record = data["000001.SZ"]
    features = record["features"]
    assert features["mom_20"] == pytest.approx(0.12)
    assert features["mom_60"] == pytest.approx(0.25)
    assert features["volat_20"] == pytest.approx(0.05)
    assert features["turn_20"] == pytest.approx(3.0)
    assert features["news_sentiment"] == pytest.approx(0.3)
    assert features["news_heat"] == pytest.approx(0.4)
    assert features["risk_penalty"] == pytest.approx(min(1.0, 0.05 * 5.0))
