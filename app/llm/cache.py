"""In-memory response cache for LLM calls."""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from copy import deepcopy
from threading import Lock
from typing import Any, Callable, Mapping, Optional, Sequence

from time import monotonic

DEFAULT_CACHE_MAX_SIZE = int(os.getenv("LLM_CACHE_MAX_SIZE", "512") or 0)
DEFAULT_CACHE_TTL = float(os.getenv("LLM_CACHE_DEFAULT_TTL", "180") or 0.0)
_GLOBAL_CACHE: "LLMResponseCache" | None = None


def _normalize(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(key): _normalize(value) for key, value in sorted(obj.items(), key=lambda item: str(item[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class LLMResponseCache:
    """Simple thread-safe LRU cache with TTL support."""

    def __init__(
        self,
        max_size: int = DEFAULT_CACHE_MAX_SIZE,
        default_ttl: float = DEFAULT_CACHE_TTL,
        *,
        time_func: Callable[[], float] = monotonic,
    ) -> None:
        self._max_size = max(0, int(max_size))
        self._default_ttl = max(0.0, float(default_ttl))
        self._time = time_func
        self._lock = Lock()
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._max_size > 0 and self._default_ttl > 0

    def get(self, key: str) -> Optional[Any]:
        if not key or not self.enabled:
            return None
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            expires_at, value = entry
            if expires_at <= self._time():
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return deepcopy(value)

    def set(self, key: str, value: Any, *, ttl: Optional[float] = None) -> None:
        if not key or not self.enabled:
            return
        ttl_value = self._default_ttl if ttl is None else float(ttl)
        if ttl_value <= 0:
            return
        expires_at = self._time() + ttl_value
        with self._lock:
            self._store[key] = (expires_at, deepcopy(value))
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def llm_cache() -> LLMResponseCache:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = LLMResponseCache()
    return _GLOBAL_CACHE


def build_cache_key(
    provider_key: str,
    resolved_endpoint: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    tools: Optional[Sequence[Mapping[str, Any]]],
    tool_choice: Any,
) -> str:
    payload = {
        "provider": provider_key,
        "model": resolved_endpoint.get("model"),
        "base_url": resolved_endpoint.get("base_url"),
        "temperature": resolved_endpoint.get("temperature"),
        "mode": resolved_endpoint.get("mode"),
        "messages": _normalize(messages),
        "tools": _normalize(tools) if tools else None,
        "tool_choice": _normalize(tool_choice),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_cacheable(
    resolved_endpoint: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    tools: Optional[Sequence[Mapping[str, Any]]],
) -> bool:
    if tools:
        return False
    if not messages:
        return False
    temperature = resolved_endpoint.get("temperature", 0.0)
    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError):
        temperature_value = 0.0
    return temperature_value <= 0.3
