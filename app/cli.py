"""Command line entry points for routine tasks."""
from __future__ import annotations

import argparse
from datetime import date

from app.backtest.engine import BtConfig, run_backtest
from app.data.schema import initialize_database
from app.ingest.checker import run_boot_check


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


def run_boot_check_cli(days: int) -> None:
    report = run_boot_check(days=days)
    print("Boot check summary:")
    print(f"  Period: {report.start} ~ {report.end}")
    print(f"  Expected trading days: {report.expected_trading_days}")
    for name, info in report.tables.items():
        print(
            f"  {name}: min={info.get('min')}, max={info.get('max')}, "
            f"distinct={info.get('distinct_days')}, ok={info.get('meets_expectation')}"
        )
    stock = report.stock_basic
    print(
        f"  stock_basic: total={stock.get('total')}, "
        f"SSE listed={stock.get('sse_listed')}, SZSE listed={stock.get('szse_listed')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Investment assistant toolkit")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init-db", help="Initialize SQLite schema")

    boot_parser = sub.add_parser("boot-check", help="Run startup data coverage check")
    boot_parser.add_argument("--days", type=int, default=365, help="Lookback window in days")

    sub.add_parser("sample-backtest", help="Execute demo backtest run")

    args = parser.parse_args()

    if args.command is None or args.command == "init-db":
        init_db()
    elif args.command == "boot-check":
        run_boot_check_cli(days=args.days)
    elif args.command == "sample-backtest":
        run_sample_backtest()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
