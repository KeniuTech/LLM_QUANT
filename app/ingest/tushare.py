"""Ingestion job orchestrator wrapping TuShare utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, List, Optional, Sequence

from app.features.factors import compute_factors_incremental
from app.utils import alerts
from app.utils.logging import get_logger

from .api_client import LOG_EXTRA
from .coverage import collect_data_coverage, ensure_data_coverage
from .job_logger import JobLogger

LOGGER = get_logger(__name__)

PostTask = Callable[["FetchJob"], None]


@dataclass
class FetchJob:
    name: str
    start: date
    end: date
    granularity: str = "daily"
    ts_codes: Optional[Sequence[str]] = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "start": str(self.start),
            "end": str(self.end),
            "granularity": self.granularity,
            "codes": list(self.ts_codes or ()),
        }


def _default_post_tasks(job: FetchJob) -> List[PostTask]:
    if job.granularity != "daily":
        return []
    return [_run_factor_backfill]


def _run_factor_backfill(job: FetchJob) -> None:
    LOGGER.info("开始计算因子：%s", job.name, extra=LOG_EXTRA)
    compute_factors_incremental(
        ts_codes=job.ts_codes,
        skip_existing=True,
        persist=True,
    )
    alerts.clear_warnings("Factors")


def run_ingestion(
    job: FetchJob,
    *,
    include_limits: bool = True,
    include_extended: bool = True,
    include_news: bool = True,
    post_tasks: Optional[Iterable[PostTask]] = None,
) -> None:
    """Execute a TuShare ingestion job with optional post processing hooks."""

    with JobLogger("TuShare数据获取") as logger:
        LOGGER.info("启动 TuShare 拉取任务：%s", job.name, extra=LOG_EXTRA)
        try:
            ensure_data_coverage(
                job.start,
                job.end,
                ts_codes=job.ts_codes,
                include_limits=include_limits,
                include_extended=include_extended,
                include_news=include_news,
                force=True,
            )
            logger.update_metadata(job.as_dict())
            alerts.clear_warnings("TuShare")

            tasks = list(post_tasks) if post_tasks is not None else _default_post_tasks(job)
            for task in tasks:
                try:
                    task(job)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception(
                        "后置任务执行失败：task=%s",
                        getattr(task, "__name__", task),
                        extra=LOG_EXTRA,
                    )
                    alerts.add_warning("Factors", f"后置任务失败：{job.name}", str(exc))
                    logger.update_status("failed", f"后置任务失败：{exc}")
                    raise

        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("数据拉取失败 job=%s", job.name, extra=LOG_EXTRA)
            alerts.add_warning("TuShare", f"拉取任务失败：{job.name}", str(exc))
            raise
        LOGGER.info("任务 %s 完成", job.name, extra=LOG_EXTRA)


__all__ = [
    "FetchJob",
    "collect_data_coverage",
    "ensure_data_coverage",
    "run_ingestion",
]
