"""Helpers for logging decision tuning experiments."""
from __future__ import annotations

import json

from typing import Any, Dict, Mapping, Optional

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


def select_best_tuning_result(
    experiment_id: str,
    *,
    metric: str = "reward",
    descending: bool = True,
    require_weights: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return the best tuning result for the given experiment.

    ``metric`` may refer to ``reward`` (default) or any key inside the
    persisted metrics payload. When ``require_weights`` is True, rows lacking
    weight definitions are ignored.
    """

    with db_session(read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT id, action, weights, reward, metrics, created_at
            FROM tuning_results
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchall()

    if not rows:
        return None

    best_row: Optional[Mapping[str, Any]] = None
    best_metrics: Dict[str, Any] = {}
    best_action: Dict[str, float] = {}
    best_weights: Dict[str, float] = {}
    best_score: Optional[float] = None

    for row in rows:
        action = _decode_json(row["action"])
        weights = _decode_json(row["weights"])
        metrics_payload = _decode_json(row["metrics"])
        reward_value = float(row["reward"] or 0.0)

        if require_weights and not weights:
            continue

        if metric == "reward":
            score = reward_value
        else:
            score_raw = metrics_payload.get(metric)
            if score_raw is None:
                continue
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                continue

        if best_score is None:
            choose = True
        else:
            choose = score > best_score if descending else score < best_score

        if choose:
            best_score = score
            best_row = row
            best_metrics = metrics_payload
            best_action = action
            best_weights = weights

    if best_row is None:
        return None

    return {
        "id": best_row["id"],
        "reward": float(best_row["reward"] or 0.0),
        "score": best_score,
        "metric": metric,
        "action": best_action,
        "weights": best_weights,
        "metrics": best_metrics,
        "created_at": best_row["created_at"],
    }


def _decode_json(payload: Any) -> Dict[str, Any]:
    if not payload:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return {}
