"""Runtime data warning registry for surfacing ingestion issues in UI."""
from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional


_ALERTS: List[Dict[str, str]] = []
_LOCK = Lock()


def add_warning(source: str, message: str, detail: Optional[str] = None) -> None:
    """Register or update a warning entry."""

    source = source.strip() or "unknown"
    message = message.strip() or "发生未知异常"
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with _LOCK:
        for alert in _ALERTS:
            if alert["source"] == source and alert["message"] == message:
                alert["timestamp"] = timestamp
                if detail:
                    alert["detail"] = detail
                return
        entry = {
            "source": source,
            "message": message,
            "timestamp": timestamp,
        }
        if detail:
            entry["detail"] = detail
        _ALERTS.append(entry)
        if len(_ALERTS) > 50:
            del _ALERTS[:-50]


def get_warnings() -> List[Dict[str, str]]:
    """Return a copy of current warning entries."""

    with _LOCK:
        return list(_ALERTS)


def clear_warnings(source: Optional[str] = None) -> None:
    """Clear warnings entirely or for a specific source."""

    with _LOCK:
        if source is None:
            _ALERTS.clear()
            return
        source = source.strip()
        _ALERTS[:] = [alert for alert in _ALERTS if alert["source"] != source]

