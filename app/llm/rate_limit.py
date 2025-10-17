"""Simple token-bucket rate limiter for LLM calls."""
from __future__ import annotations

from threading import Lock
from time import monotonic
from typing import Callable, Dict


class RateLimiter:
    """Token bucket rate limiter that returns required wait time."""

    def __init__(self, monotonic_func: Callable[[], float] | None = None) -> None:
        self._now = monotonic_func or monotonic
        self._lock = Lock()
        self._buckets: Dict[str, dict[str, float]] = {}

    def acquire(self, key: str, rate_per_minute: int, burst: int) -> float:
        """Attempt to consume a token; return wait time if throttled."""

        if rate_per_minute <= 0:
            return 0.0
        capacity = float(max(1, burst if burst > 0 else rate_per_minute))
        rate = float(rate_per_minute)
        now = self._now()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = {"tokens": capacity, "capacity": capacity, "last": now, "rate": rate}
                self._buckets[key] = bucket
            else:
                bucket["capacity"] = capacity
                bucket["rate"] = rate
            tokens = bucket["tokens"]
            elapsed = max(0.0, now - bucket["last"])
            tokens = min(capacity, tokens + elapsed * rate / 60.0)
            if tokens >= 1.0:
                bucket["tokens"] = tokens - 1.0
                bucket["last"] = now
                return 0.0
            bucket["tokens"] = tokens
            bucket["last"] = now
            deficit = 1.0 - tokens
            wait_time = deficit * 60.0 / rate
            return max(wait_time, 0.0)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
