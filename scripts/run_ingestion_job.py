"""命令行脚本：按日期区间执行 TuShare 拉数并同步计算因子。"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.tushare import FetchJob, run_ingestion
from app.utils.config import get_config
from app.utils.logging import get_logger
from app.utils import alerts

LOGGER = get_logger(__name__)


def _parse_date(text: str) -> date:
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"无法解析日期：{text}") from exc


def _parse_codes(raw: Sequence[str] | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    normalized = []
    for item in raw:
        token = item.strip().upper()
        if token:
            normalized.append(token)
    return tuple(dict.fromkeys(normalized)) or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按日期区间执行 TuShare 拉数并同步更新因子表",
    )
    parser.add_argument("start", type=_parse_date, help="起始交易日（格式：YYYYMMDD）")
    parser.add_argument("end", type=_parse_date, help="结束交易日（格式：YYYYMMDD）")
    parser.add_argument(
        "--codes",
        nargs="*",
        default=None,
        help="可选的股票代码列表（如 000001.SZ），不传则处理全市场",
    )
    parser.add_argument(
        "--include-limits",
        action="store_true",
        help="是否同步涨跌停/停牌等扩展数据（默认关闭，便于快速试跑）",
    )
    parser.add_argument(
        "--name",
        default="daily_ingestion",
        help="任务名称，用于日志与告警标记",
    )
    parser.add_argument(
        "--granularity",
        default="daily",
        choices=("daily", "weekly"),
        help="任务粒度，目前仅 daily 会触发因子计算",
    )
    return parser


def run_cli(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.end < args.start:
        parser.error("结束日期不能早于起始日期")

    codes = _parse_codes(args.codes)
    job = FetchJob(
        name=str(args.name),
        start=args.start,
        end=args.end,
        granularity=str(args.granularity),
        ts_codes=codes,
    )

    LOGGER.info(
        "准备执行拉数任务 name=%s start=%s end=%s codes=%s granularity=%s",
        job.name,
        job.start,
        job.end,
        job.ts_codes,
        job.granularity,
    )

    try:
        run_ingestion(job, include_limits=bool(args.include_limits))
    except Exception:  # noqa: BLE001
        LOGGER.exception("拉数任务执行失败")
        return 1

    warnings = alerts.get_warnings()
    if warnings:
        LOGGER.warning("任务完成但存在告警：%s", warnings)
        return 2

    LOGGER.info("任务执行完成，无告警")
    return 0


def main() -> None:
    exit_code = run_cli()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
