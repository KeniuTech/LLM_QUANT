"""Command-line entrypoint for data integrity checks and remediation."""
from __future__ import annotations

import argparse
from datetime import date, timedelta

from app.utils import alerts
from app.utils.data_access import DataBroker
from app.utils.data_quality import evaluate_data_quality


def main() -> None:
    args = _build_parser().parse_args()
    summary = evaluate_data_quality(window_days=args.window, top_issues=args.top)

    _print_summary(summary)

    if summary.has_blockers:
        alerts.add_warning(
            "data_quality",
            f"检测到 {len(summary.blocking)} 项阻塞数据质量问题，得分 {summary.score:.1f}",
            detail=str(summary.as_dict()),
        )
        if args.auto_fill:
            _trigger_auto_fill(args.window)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run data integrity checks and optional remediation.")
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Number of trailing days to inspect (default: 7).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Maximum issues per severity to display (default: 5).",
    )
    parser.add_argument(
        "--auto-fill",
        action="store_true",
        help="Trigger DataBroker coverage runner when blocking issues are detected.",
    )
    return parser


def _print_summary(summary) -> None:
    print(f"窗口: {summary.window_days} 天 | 质量得分: {summary.score:.1f} | 总检查数: {summary.total_checks}")
    if summary.severity_counts:
        print("严重度统计:")
        for severity, count in summary.severity_counts.items():
            print(f"  - {severity}: {count}")
    if summary.blocking:
        print("阻塞问题:")
        for issue in summary.blocking:
            print(f"  [ERROR] {issue.check}: {issue.detail}")
    if summary.warnings:
        print("警告:")
        for issue in summary.warnings:
            print(f"  [WARN] {issue.check}: {issue.detail}")


def _trigger_auto_fill(window_days: int) -> None:
    broker = DataBroker()
    end = date.today()
    start = end - timedelta(days=window_days)
    try:
        broker.coverage_runner(start, end)
        print(f"已触发补数流程: {start} -> {end}")
    except Exception as exc:  # noqa: BLE001
        alerts.add_warning("data_quality", "自动补数失败", detail=str(exc))


if __name__ == "__main__":
    main()
