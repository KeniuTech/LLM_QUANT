"""Helpers for logging decision tuning experiments."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .db import db_session
from .logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "tuning"}


def log_tuning_result(
    *,
    experiment_id: str,
    strategy: str,
    action: Dict[str, Any],
    reward: float,
    metrics: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> None:
    """Persist a tuning result into the SQLite table."""

    try:
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO tuning_results (experiment_id, strategy, action, weights, reward, metrics)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    strategy,
                    json.dumps(action, ensure_ascii=False),
                    json.dumps(weights or {}, ensure_ascii=False),
                    float(reward),
                    json.dumps(metrics, ensure_ascii=False),
                ),
            )
    except Exception:  # noqa: BLE001
        LOGGER.exception("记录调参结果失败", extra=LOG_EXTRA)
