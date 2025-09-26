"""Command line entry points for routine tasks."""
from __future__ import annotations

from datetime import date

from app.backtest.engine import BtConfig, run_backtest
from app.data.schema import initialize_database


def init_db() -> None:
    result = initialize_database()
    if result.skipped:
        print("Database already initialized; skipping schema creation")
    else:
        print(f"Initialized database with {result.executed} statements")


def run_sample_backtest() -> None:
    cfg = BtConfig(
        id="demo",
        name="Demo Strategy",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 3, 31),
        universe=["000001.SZ"],
        params={
            "target": 0.035,
            "stop": -0.015,
            "hold_days": 10,
        },
    )
    run_backtest(cfg)


if __name__ == "__main__":
    init_db()
