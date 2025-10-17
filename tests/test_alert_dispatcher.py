"""Tests for alert dispatcher configuration and delivery."""
from __future__ import annotations

import json

from app.utils import alerts
from app.utils.config import AlertChannelSettings


def test_alert_dispatcher_posts_payload(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_request(*, method, url, data=None, headers=None, timeout=None):
        calls.append(
            {
                "method": method,
                "url": url,
                "data": data,
                "headers": headers,
                "timeout": timeout,
            }
        )

        class _Resp:
            status_code = 200

        return _Resp()

    monkeypatch.setattr("app.utils.alert_dispatcher.requests.request", fake_request)

    alerts.clear_warnings()
    alerts.configure_channels(
        {
            "ops": AlertChannelSettings(
                key="ops",
                kind="webhook",
                url="https://example.com/webhook",
                enabled=True,
                level="info",
                headers={"X-Test": "1"},
                extra_params={"channel": "risk"},
            )
        }
    )

    alerts.add_warning(
        "risk_system",
        "阻断测试",
        detail="blocked",
        level="error",
        tags=["risk", "blocked"],
        payload={"reason": "blocked"},
    )

    assert calls, "expected dispatcher to send webhook call"
    call = calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://example.com/webhook"
    assert call["headers"]["X-Test"] == "1"
    payload = json.loads(call["data"])
    assert payload["message"] == "阻断测试"
    assert payload["channel"] == "risk"
    assert payload["payload"]["reason"] == "blocked"

    warnings = alerts.get_warnings()
    assert warnings
    assert warnings[0]["level"] == "error"
    assert "blocked" in warnings[0].get("tags", [])

    alerts.configure_channels({})
