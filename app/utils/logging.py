"""项目级日志配置模块。

提供统一的日志初始化入口，支持同时输出到终端、文件以及 SQLite
数据库中的 `run_log` 表。数据库写入便于在 UI 中或离线复盘时查看
运行轨迹。
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
from datetime import datetime
from logging import Handler, LogRecord
from pathlib import Path
from typing import Optional

from .config import get_config
from .db import db_session

_LOGGER_NAME = "app.logging"
_IS_CONFIGURED = False


class DatabaseLogHandler(Handler):
    """将日志写入 SQLite `run_log` 表的自定义 Handler。"""

    def emit(self, record: LogRecord) -> None:  # noqa: D401 - 标准 logging 接口
        try:
            message = self.format(record)
            stage = getattr(record, "stage", None)
            ts = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
            with db_session() as conn:
                conn.execute(
                    "INSERT INTO run_log (ts, stage, level, msg) VALUES (?, ?, ?, ?)",
                    (ts, stage, record.levelname, message),
                )
        except sqlite3.OperationalError as exc:
            # 表不存在时直接跳过，避免首次初始化阶段报错
            if "no such table" not in str(exc).lower():
                self.handleError(record)
        except Exception:
            self.handleError(record)


def _build_formatter() -> logging.Formatter:
    return logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")


def setup_logging(
    *,
    level: int = logging.INFO,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    db_level: Optional[int] = None,
) -> logging.Logger:
    """配置根 logger。重复调用时将复用已存在的配置。"""

    global _IS_CONFIGURED
    if _IS_CONFIGURED:
        return logging.getLogger()

    env_level = os.getenv("LLM_QUANT_LOG_LEVEL")
    if env_level is None:
        level = logging.DEBUG
    else:
        try:
            level = getattr(logging, env_level.upper())
        except AttributeError:
            logging.getLogger(_LOGGER_NAME).warning(
                "非法的日志级别 %s，回退到 DEBUG", env_level
            )
            level = logging.DEBUG

    cfg = get_config()
    log_dir: Path = cfg.data_paths.root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "app.log"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = _build_formatter()

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(console_level or level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setLevel(file_level or level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    db_handler = DatabaseLogHandler(level=db_level or level)
    db_handler.setFormatter(formatter)
    root.addHandler(db_handler)

    _IS_CONFIGURED = True
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """返回指定名称的 logger，确保全局配置已就绪。"""

    setup_logging()
    return logging.getLogger(name)


# 默认在模块导入时完成配置，适配现有调用方式。
setup_logging()
