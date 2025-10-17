"""任务记录工具类。"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.utils.db import db_session
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "data_ingest"}


class JobLogger:
    """任务记录器，通过数据库记录抓取作业运行情况。"""

    def __init__(self, job_type: str) -> None:
        """初始化任务记录器。

        Args:
            job_type: 任务类型
        """
        self.job_type = job_type
        self.job_id: Optional[int] = None

    def __enter__(self) -> "JobLogger":
        """开始记录任务。"""
        with db_session() as session:
            cursor = session.execute(
                """
                INSERT INTO fetch_jobs (job_type, status, created_at, updated_at)
                VALUES (?, 'running', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (self.job_type,),
            )
            self.job_id = cursor.lastrowid
            session.commit()
        LOGGER.info(
            "抓取任务启动 job_type=%s job_id=%s",
            self.job_type,
            self.job_id,
            extra=LOG_EXTRA,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """结束任务记录。"""
        if exc_val:
            LOGGER.exception(
                "抓取任务失败 job_type=%s job_id=%s err=%s",
                self.job_type,
                self.job_id,
                exc_val,
                extra=LOG_EXTRA,
            )
            self.update_status("failed", str(exc_val))
        else:
            LOGGER.info(
                "抓取任务完成 job_type=%s job_id=%s",
                self.job_type,
                self.job_id,
                extra=LOG_EXTRA,
            )
            self.update_status("success")

    def update_status(self, status: str, error_msg: Optional[str] = None) -> None:
        """更新任务状态。

        Args:
            status: 新状态
            error_msg: 错误信息（如果有）
        """
        if not self.job_id:
            LOGGER.debug("忽略无效任务状态更新 job_type=%s status=%s", self.job_type, status, extra=LOG_EXTRA)
            return

        with db_session() as session:
            session.execute(
                """
                UPDATE fetch_jobs
                SET status = ?,
                    error_msg = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, error_msg, self.job_id),
            )
            session.commit()
        LOGGER.debug(
            "更新任务状态 job_type=%s job_id=%s status=%s error=%s",
            self.job_type,
            self.job_id,
            status,
            error_msg,
            extra=LOG_EXTRA,
        )

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """更新任务元数据。

        Args:
            metadata: 元数据字典
        """
        if not self.job_id:
            LOGGER.debug(
                "忽略元数据更新（尚未初始化） job_type=%s",
                self.job_type,
                extra=LOG_EXTRA,
            )
            return

        with db_session() as session:
            session.execute(
                """
                UPDATE fetch_jobs
                SET metadata = ?
                WHERE id = ?
                """,
                (json.dumps(metadata), self.job_id),
            )
            session.commit()
        LOGGER.debug(
            "记录任务元数据 job_type=%s job_id=%s keys=%s",
            self.job_type,
            self.job_id,
            sorted(metadata.keys()),
            extra=LOG_EXTRA,
        )
