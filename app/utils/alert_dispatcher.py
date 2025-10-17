"""Dispatch structured alerts to external channels."""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import requests

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from app.utils.config import AlertChannelSettings
else:
    AlertChannelSettings = Any  # type: ignore[assignment]

_LEVEL_RANK: Dict[str, int] = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
}


class _Channel:
    """Runtime wrapper around a configured alert channel."""

    __slots__ = (
        "settings",
        "_lock",
        "_last_signature",
        "_last_sent",
    )

    def __init__(self, settings: AlertChannelSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._last_signature: Optional[str] = None
        self._last_sent: float = 0.0

    @property
    def name(self) -> str:
        return getattr(self.settings, "key", "channel")

    def send(self, entry: Mapping[str, Any]) -> None:
        if not self._should_send(entry):
            return
        payload = self._build_payload(entry)
        self._deliver(payload)

    def _should_send(self, entry: Mapping[str, Any]) -> bool:
        level = str(entry.get("level", "warning") or "warning").lower()
        level_rank = _LEVEL_RANK.get(level, _LEVEL_RANK["warning"])

        threshold = str(getattr(self.settings, "level", "warning") or "warning").lower()
        threshold_rank = _LEVEL_RANK.get(threshold, _LEVEL_RANK["warning"])
        if level_rank < threshold_rank:
            return False

        channel_tags: Sequence[str] = getattr(self.settings, "tags", []) or []
        if channel_tags:
            event_tags = entry.get("tags") or []
            if not isinstance(event_tags, Iterable):
                event_tags = []
            if not set(str(tag) for tag in event_tags).intersection(str(tag) for tag in channel_tags):
                return False

        cooldown = float(getattr(self.settings, "cooldown_seconds", 0.0) or 0.0)
        signature = f"{entry.get('source')}|{entry.get('message')}|{level}"
        if cooldown > 0:
            now = time.monotonic()
            with self._lock:
                if self._last_signature == signature and (now - self._last_sent) < cooldown:
                    return False
                self._last_signature = signature
                self._last_sent = now
        return True

    def _build_payload(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source": entry.get("source"),
            "message": entry.get("message"),
            "detail": entry.get("detail"),
            "timestamp": entry.get("timestamp"),
            "level": entry.get("level"),
        }
        tags = entry.get("tags")
        if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
            payload["tags"] = list(tags)
        if "payload" in entry and entry["payload"] is not None:
            payload["payload"] = entry["payload"]
        extra_params = getattr(self.settings, "extra_params", None)
        if isinstance(extra_params, Mapping):
            for key, value in extra_params.items():
                payload.setdefault(key, value)
        return payload

    def _deliver(self, payload: Mapping[str, Any]) -> None:
        url = getattr(self.settings, "url", "")
        if not url:
            return
        method = str(getattr(self.settings, "method", "POST") or "POST").upper()
        timeout = float(getattr(self.settings, "timeout", 3.0) or 3.0)
        headers: MutableMapping[str, str] = {}
        raw_headers = getattr(self.settings, "headers", None)
        if isinstance(raw_headers, Mapping):
            headers = {str(k): str(v) for k, v in raw_headers.items()}

        headers.setdefault("Content-Type", "application/json")
        body = json.dumps(payload, ensure_ascii=False)

        signing_secret = getattr(self.settings, "signing_secret", None)
        if signing_secret:
            import hashlib
            import hmac

            digest = hmac.new(
                str(signing_secret).encode("utf-8"),
                body.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers.setdefault("X-Signature", digest)

        try:
            requests.request(
                method=method,
                url=url,
                data=body,
                headers=headers,
                timeout=timeout,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("发送告警失败: channel=%s", self.name)


class AlertDispatcher:
    """Singleton-style dispatcher coordinating channel delivery."""

    def __init__(self) -> None:
        self._channels: Dict[str, _Channel] = {}
        self._lock = threading.Lock()

    def configure(self, configs: Mapping[str, AlertChannelSettings]) -> None:
        active: Dict[str, _Channel] = {}
        for key, cfg in configs.items():
            if not cfg or not getattr(cfg, "enabled", True):
                continue
            if not getattr(cfg, "url", ""):
                continue
            channel = _Channel(cfg)
            active[key] = channel
        with self._lock:
            self._channels = active

    def dispatch(self, entry: Mapping[str, Any]) -> None:
        if not self._channels:
            return
        for channel in list(self._channels.values()):
            channel.send(entry)


_DISPATCHER = AlertDispatcher()


def get_dispatcher() -> AlertDispatcher:
    return _DISPATCHER
