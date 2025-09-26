"""Centralized logging configuration."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import get_config


def configure_logging(level: int = logging.INFO) -> None:
    """Setup root logger with file and console handlers."""

    cfg = get_config()
    log_dir = cfg.data_paths.root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "app.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(logfile, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


configure_logging()
