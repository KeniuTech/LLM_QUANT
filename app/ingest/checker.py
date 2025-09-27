"""数据覆盖开机检查器。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Dict

from app.data.schema import initialize_database
from app.ingest.tushare import collect_data_coverage, ensure_data_coverage
from app.utils.config import get_config
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class CoverageReport:
    start: str
    end: str
    expected_trading_days: int
    tables: Dict[str, Dict[str, object]]
    stock_basic: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "expected_trading_days": self.expected_trading_days,
            "tables": self.tables,
            "stock_basic": self.stock_basic,
        }


def _default_window(days: int = 365) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=days)
    return start, end


def run_boot_check(
    days: int = 365,
    auto_fetch: bool = True,
    progress_hook: Callable[[str, float], None] | None = None,
    force_refresh: bool | None = None,
) -> CoverageReport:
    """执行开机自检，必要时自动补数据。"""

    initialize_database()
    start, end = _default_window(days)
    LOGGER.info("开机检查覆盖窗口：%s 至 %s", start, end)

    refresh = force_refresh
    if refresh is None:
        refresh = get_config().force_refresh

    if auto_fetch:
        ensure_data_coverage(
            start,
            end,
            force=refresh,
            progress_hook=progress_hook,
        )

    coverage = collect_data_coverage(start, end)

    report = CoverageReport(
        start=coverage["period"]["start"],
        end=coverage["period"]["end"],
        expected_trading_days=coverage["period"]["expected_trading_days"],
        tables={k: v for k, v in coverage.items() if k not in ("period", "stock_basic")},
        stock_basic=coverage["stock_basic"],
    )

    LOGGER.info(
        "数据覆盖情况：日线[%s,%s]，Distinct=%s，目标交易日=%s",
        report.tables["daily"].get("min"),
        report.tables["daily"].get("max"),
        report.tables["daily"].get("distinct_days"),
        report.expected_trading_days,
    )
    if progress_hook:
        progress_hook("数据覆盖检查完成", 1.0)

    return report
