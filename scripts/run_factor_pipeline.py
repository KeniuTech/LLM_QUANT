"""Command-line helper for running the factor computation pipeline."""
from __future__ import annotations

import argparse
from datetime import date, datetime
from typing import Iterable, List, Optional, Sequence

from app.features.factors import (
    DEFAULT_FACTORS,
    FactorResult,
    FactorSpec,
    compute_factor_range,
    compute_factors,
    compute_factors_incremental,
    lookup_factor_spec,
)
from app.features.factor_audit import audit_factors
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    args = _build_parser().parse_args()
    persist = not args.no_persist
    factor_specs = _resolve_factor_specs(args.factors)
    ts_codes = _normalize_codes(args.ts_codes)
    batch_size = args.batch_size or 100

    if args.mode == "single":
        if not args.trade_date:
            raise SystemExit("--trade-date is required in single mode")
        trade_day = _parse_date(args.trade_date)
        results = compute_factors(
            trade_day,
            factor_specs,
            ts_codes=ts_codes,
            skip_existing=args.skip_existing,
            batch_size=batch_size,
            persist=persist,
        )
        _print_summary_single(trade_day, results, persist)
        audit_dates = [trade_day] if args.audit else []
    elif args.mode == "range":
        if not args.start or not args.end:
            raise SystemExit("--start and --end are required in range mode")
        start = _parse_date(args.start)
        end = _parse_date(args.end)
        results = compute_factor_range(
            start,
            end,
            factors=factor_specs,
            ts_codes=ts_codes,
            skip_existing=args.skip_existing,
            persist=persist,
        )
        _print_summary_range(start, end, results, persist)
        audit_dates = sorted({result.trade_date for result in results}) if args.audit else []
    else:
        summary = compute_factors_incremental(
            factors=factor_specs,
            ts_codes=ts_codes,
            skip_existing=args.skip_existing,
            max_trading_days=args.max_days,
            persist=persist,
        )
        _print_summary_incremental(summary, persist)
        audit_dates = summary.get("trade_dates", []) if args.audit else []

    if args.audit and audit_dates:
        for audit_date in audit_dates:
            summary = audit_factors(
                audit_date,
                factors=factor_specs,
                tolerance=args.audit_tolerance,
                max_issues=args.max_audit_issues,
            )
            _print_audit_summary(summary)
    elif args.audit:
        LOGGER.info("无可审计的日期，跳过因子审计步骤")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run factor computation pipeline.")
    parser.add_argument(
        "--mode",
        choices=("single", "range", "incremental"),
        default="single",
        help="Pipeline mode (default: single).",
    )
    parser.add_argument("--trade-date", help="Trade date (YYYYMMDD) for single mode.")
    parser.add_argument("--start", help="Start date (YYYYMMDD) for range mode.")
    parser.add_argument("--end", help="End date (YYYYMMDD) for range mode.")
    parser.add_argument(
        "--max-days",
        type=int,
        default=5,
        help="Limit of trading days for incremental mode.",
    )
    parser.add_argument(
        "--ts-code",
        dest="ts_codes",
        action="append",
        help="Limit computation to specific ts_code. Can be provided multiple times.",
    )
    parser.add_argument(
        "--factor",
        dest="factors",
        action="append",
        help="Factor name to include. Defaults to the built-in set.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip securities that already have persisted values for the target date(s).",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Dry-run mode; compute factors without writing to the database.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Override default batch size when computing factors.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run formula audit after computation completes.",
    )
    parser.add_argument(
        "--audit-tolerance",
        type=float,
        default=1e-6,
        help="Allowed absolute difference when auditing factors.",
    )
    parser.add_argument(
        "--max-audit-issues",
        type=int,
        default=50,
        help="Maximum number of detailed audit issues to print.",
    )
    return parser


def _resolve_factor_specs(names: Optional[Sequence[str]]) -> List[FactorSpec]:
    if not names:
        return list(DEFAULT_FACTORS)
    resolved: List[FactorSpec] = []
    seen: set[str] = set()
    for name in names:
        spec = lookup_factor_spec(name)
        if spec is None:
            LOGGER.warning("未知因子，忽略: %s", name)
            continue
        if spec.name in seen:
            continue
        resolved.append(spec)
        seen.add(spec.name)
    return resolved or list(DEFAULT_FACTORS)


def _normalize_codes(codes: Optional[Iterable[str]]) -> List[str] | None:
    if not codes:
        return None
    normalized = []
    for code in codes:
        text = (code or "").strip().upper()
        if text:
            normalized.append(text)
    return normalized or None


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise SystemExit(f"Invalid date: {value}")


def _print_summary_single(trade_day: date, results: Sequence[FactorResult], persist: bool) -> None:
    LOGGER.info(
        "单日因子计算完成 trade_date=%s rows=%s persist=%s",
        trade_day.isoformat(),
        len(results),
        bool(persist),
    )


def _print_summary_range(start: date, end: date, results: Sequence[FactorResult], persist: bool) -> None:
    trade_dates = sorted({result.trade_date for result in results})
    LOGGER.info(
        "区间因子计算完成 start=%s end=%s days=%s rows=%s persist=%s",
        start.isoformat(),
        end.isoformat(),
        len(trade_dates),
        len(results),
        bool(persist),
    )


def _print_summary_incremental(summary: dict, persist: bool) -> None:
    trade_dates = summary.get("trade_dates") or []
    start = trade_dates[0].isoformat() if trade_dates else None
    end = trade_dates[-1].isoformat() if trade_dates else None
    LOGGER.info(
        "增量因子计算完成 start=%s end=%s days=%s rows=%s persist=%s",
        start,
        end,
        len(trade_dates),
        summary.get("count", 0),
        bool(persist),
    )


def _print_audit_summary(summary) -> None:
    LOGGER.info(
        "因子审计 trade_date=%s mismatched=%s evaluated=%s missing_persisted=%s missing_recomputed=%s issues=%s",
        summary.trade_date.isoformat(),
        summary.mismatched,
        summary.evaluated,
        summary.missing_persisted,
        summary.missing_recomputed,
        len(summary.issues),
    )
    for issue in summary.issues:
        LOGGER.warning(
            "审计异常 ts_code=%s factor=%s stored=%s recomputed=%s diff=%s",
            issue.ts_code,
            issue.factor,
            issue.stored,
            issue.recomputed,
            issue.difference,
        )


if __name__ == "__main__":
    main()
