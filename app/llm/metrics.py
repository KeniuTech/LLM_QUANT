"""Simple runtime metrics collector for LLM calls."""
from __future__ import annotations

import copy
import logging
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Deque, Dict, List, Optional


@dataclass
class _Metrics:
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    provider_calls: Dict[str, int] = field(default_factory=dict)
    model_calls: Dict[str, int] = field(default_factory=dict)
    decisions: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=500))
    decision_action_counts: Dict[str, int] = field(default_factory=dict)
    total_latency: float = 0.0
    latency_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    template_usage: Dict[str, Dict[str, object]] = field(default_factory=dict)


_METRICS = _Metrics()
_LOCK = Lock()
_LISTENERS: List[Callable[[Dict[str, object]], None]] = []

LOGGER = logging.getLogger(__name__)


def record_call(
    provider: str,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    *,
    duration: Optional[float] = None,
) -> None:
    """Record a single LLM API invocation."""

    normalized_provider = (provider or "unknown").lower()
    normalized_model = (model or "").strip()
    with _LOCK:
        _METRICS.total_calls += 1
        _METRICS.provider_calls[normalized_provider] = (
            _METRICS.provider_calls.get(normalized_provider, 0) + 1
        )
        if normalized_model:
            _METRICS.model_calls[normalized_model] = (
                _METRICS.model_calls.get(normalized_model, 0) + 1
            )
        if prompt_tokens:
            _METRICS.total_prompt_tokens += int(prompt_tokens)
        if completion_tokens:
            _METRICS.total_completion_tokens += int(completion_tokens)
        if duration is not None:
            duration_value = max(0.0, float(duration))
            _METRICS.total_latency += duration_value
            _METRICS.latency_samples.append(duration_value)
    _notify_listeners()


def snapshot(reset: bool = False) -> Dict[str, object]:
    """Return a snapshot of current metrics. Optionally reset counters."""

    with _LOCK:
        data = {
            "total_calls": _METRICS.total_calls,
            "total_prompt_tokens": _METRICS.total_prompt_tokens,
            "total_completion_tokens": _METRICS.total_completion_tokens,
            "provider_calls": dict(_METRICS.provider_calls),
            "model_calls": dict(_METRICS.model_calls),
            "decision_action_counts": dict(_METRICS.decision_action_counts),
            "recent_decisions": list(_METRICS.decisions),
            "average_latency": (
                _METRICS.total_latency / _METRICS.total_calls
                if _METRICS.total_calls
                else 0.0
            ),
            "latency_samples": list(_METRICS.latency_samples),
            "template_usage": copy.deepcopy(_METRICS.template_usage),
        }
        if reset:
            _METRICS.total_calls = 0
            _METRICS.total_prompt_tokens = 0
            _METRICS.total_completion_tokens = 0
            _METRICS.provider_calls.clear()
            _METRICS.model_calls.clear()
            _METRICS.decision_action_counts.clear()
            _METRICS.decisions.clear()
            _METRICS.total_latency = 0.0
            _METRICS.latency_samples.clear()
            _METRICS.template_usage.clear()
        return data


def reset() -> None:
    """Reset all collected metrics."""

    snapshot(reset=True)
    _notify_listeners()


def record_decision(
    *,
    ts_code: str,
    trade_date: str,
    action: str,
    confidence: float,
    summary: str,
    source: str,
    departments: Optional[Dict[str, object]] = None,
) -> None:
    """Record a high-level decision for later inspection."""

    record = {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "action": action,
        "confidence": confidence,
        "summary": summary,
        "source": source,
        "departments": departments or {},
    }
    with _LOCK:
        _METRICS.decisions.append(record)
        _METRICS.decision_action_counts[action] = (
            _METRICS.decision_action_counts.get(action, 0) + 1
        )
    _notify_listeners()


def record_template_usage(
    template_id: str,
    *,
    version: Optional[str],
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> None:
    """Record usage statistics for a specific prompt template."""

    if not template_id:
        return
    label = template_id.strip()
    version_label = version or "active"
    with _LOCK:
        entry = _METRICS.template_usage.setdefault(
            label,
            {"total_calls": 0, "versions": {}},
        )
        entry["total_calls"] = int(entry.get("total_calls", 0)) + 1
        versions = entry.setdefault("versions", {})
        version_entry = versions.setdefault(
            version_label,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        )
        version_entry["calls"] = int(version_entry.get("calls", 0)) + 1
        if prompt_tokens:
            version_entry["prompt_tokens"] = int(version_entry.get("prompt_tokens", 0)) + int(prompt_tokens)
        if completion_tokens:
            version_entry["completion_tokens"] = int(version_entry.get("completion_tokens", 0)) + int(completion_tokens)
    _notify_listeners()


def recent_decisions(limit: int = 50) -> List[Dict[str, object]]:
    """Return the most recent decisions up to limit."""

    with _LOCK:
        if limit <= 0:
            return []
        return list(_METRICS.decisions)[-limit:]


def register_listener(callback: Callable[[Dict[str, object]], None]) -> None:
    """Register a callback invoked whenever metrics change."""

    if not callable(callback):
        return
    with _LOCK:
        if callback in _LISTENERS:
            should_invoke = False
        else:
            _LISTENERS.append(callback)
            should_invoke = True
    if should_invoke:
        try:
            callback(snapshot())
        except Exception:  # noqa: BLE001
            LOGGER.exception("Metrics listener failed on initial callback")


def unregister_listener(callback: Callable[[Dict[str, object]], None]) -> None:
    """Remove a previously registered metrics callback."""

    with _LOCK:
        if callback in _LISTENERS:
            _LISTENERS.remove(callback)


def _notify_listeners() -> None:
    with _LOCK:
        listeners = list(_LISTENERS)
    if not listeners:
        return
    data = snapshot()
    for callback in listeners:
        try:
            callback(data)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Metrics listener execution failed")
