"""Runtime data warning registry with external dispatch support."""
from __future__ import annotations

import logging
from datetime import datetime
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence

from .alert_dispatcher import get_dispatcher

if TYPE_CHECKING:  # pragma: no cover
    from app.utils.config import AlertChannelSettings

LOGGER = logging.getLogger(__name__)

_ALERTS: List[Dict[str, Any]] = []
_SINKS: List[Callable[[Dict[str, Any]], None]] = []
_LOCK = Lock()
_MAX_ALERTS = 50


def configure_channels(channels: Mapping[str, "AlertChannelSettings"]) -> None:
    """Configure external dispatch channels."""

    try:
        get_dispatcher().configure(channels)
    except Exception:  # noqa: BLE001
        LOGGER.debug("配置外部告警通道失败", exc_info=True)


def register_sink(sink: Callable[[Dict[str, Any]], None]) -> None:
    """Attach an additional sink to receive alert payloads."""

    with _LOCK:
        if sink not in _SINKS:
            _SINKS.append(sink)


def unregister_sink(sink: Callable[[Dict[str, Any]], None]) -> None:
    """Detach a previously registered sink."""

    with _LOCK:
        _SINKS[:] = [existing for existing in _SINKS if existing is not sink]


def add_warning(
    source: str,
    message: str,
    detail: Optional[str] = None,
    *,
    level: str = "warning",
    tags: Optional[Sequence[str]] = None,
    payload: Optional[Mapping[str, Any]] = None,
) -> None:
    """Register or update a warning entry and dispatch to sinks."""

    source = (source or "").strip() or "unknown"
    message = (message or "").strip() or "发生未知异常"
    normalized_level = str(level or "warning").lower()
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    normalized_tags: List[str] = []
    if tags:
        normalized_tags = [
            str(tag).strip()
            for tag in tags
            if isinstance(tag, str) and tag.strip()
        ]

    snapshot: Dict[str, Any] = {}
    sinks: List[Callable[[Dict[str, Any]], None]] = []

    with _LOCK:
        for alert in _ALERTS:
            if alert["source"] == source and alert["message"] == message:
                alert["timestamp"] = timestamp
                alert["level"] = normalized_level
                if detail:
                    alert["detail"] = detail
                if normalized_tags:
                    alert["tags"] = list(normalized_tags)
                if payload is not None:
                    alert["payload"] = dict(payload) if isinstance(payload, Mapping) else payload
                snapshot = dict(alert)
                sinks = list(_SINKS)
                break
        else:
            entry: Dict[str, Any] = {
                "source": source,
                "message": message,
                "timestamp": timestamp,
                "level": normalized_level,
            }
            if detail:
                entry["detail"] = detail
            if normalized_tags:
                entry["tags"] = list(normalized_tags)
            if payload is not None:
                entry["payload"] = dict(payload) if isinstance(payload, Mapping) else payload
            _ALERTS.append(entry)
            if len(_ALERTS) > _MAX_ALERTS:
                del _ALERTS[:-_MAX_ALERTS]
            snapshot = dict(entry)
            sinks = list(_SINKS)

    for sink in sinks:
        try:
            sink(dict(snapshot))
        except Exception:  # noqa: BLE001
            LOGGER.debug("执行告警 sink 失败：%s", getattr(sink, "__name__", sink), exc_info=True)

    try:
        get_dispatcher().dispatch(snapshot)
    except Exception:  # noqa: BLE001
        LOGGER.debug("外部告警发送失败 source=%s", source, exc_info=True)


def get_warnings() -> List[Dict[str, Any]]:
    """Return a copy of current warning entries."""

    with _LOCK:
        return [dict(alert) for alert in _ALERTS]


def clear_warnings(source: Optional[str] = None) -> None:
    """Clear warnings entirely or for a specific source."""

    with _LOCK:
        if source is None:
            _ALERTS.clear()
            return
        source_key = source.strip()
        _ALERTS[:] = [alert for alert in _ALERTS if alert["source"] != source_key]
