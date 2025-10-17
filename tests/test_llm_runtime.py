"""Tests for LLM runtime helpers such as rate limiting and caching."""
from __future__ import annotations

import pytest

from app.llm.cache import LLMResponseCache
from app.llm.rate_limit import RateLimiter


def test_rate_limiter_returns_wait_time() -> None:
    """Ensure limiter enforces configured throughput."""

    current = [0.0]

    def fake_time() -> float:
        return current[0]

    limiter = RateLimiter(monotonic_func=fake_time)

    assert limiter.acquire("openai", rate_per_minute=2, burst=1) == pytest.approx(0.0)
    delay = limiter.acquire("openai", rate_per_minute=2, burst=1)
    assert delay == pytest.approx(30.0, rel=1e-3)
    current[0] += 30.0
    assert limiter.acquire("openai", rate_per_minute=2, burst=1) == pytest.approx(0.0)


def test_llm_response_cache_ttl_and_lru() -> None:
    """Validate cache expiration and eviction semantics."""

    current = [0.0]

    def fake_time() -> float:
        return current[0]

    cache = LLMResponseCache(max_size=2, default_ttl=10, time_func=fake_time)

    cache.set("key1", {"value": 1})
    assert cache.get("key1") == {"value": 1}

    current[0] += 11
    assert cache.get("key1") is None

    cache.set("key1", {"value": 1})
    cache.set("key2", {"value": 2})
    assert cache.get("key1") == {"value": 1}
    cache.set("key3", {"value": 3})
    assert cache.get("key2") is None
    assert cache.get("key1") == {"value": 1}
    assert cache.get("key3") == {"value": 3}
