"""Sidebar dashboard for the Streamlit UI."""
from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from app.llm.metrics import (
    recent_decisions as llm_recent_decisions,
    register_listener as register_llm_metrics_listener,
    snapshot as snapshot_llm_metrics,
)
from app.utils import alerts

from app.ui.shared import LOGGER, LOG_EXTRA

_DASHBOARD_CONTAINERS: Optional[tuple[object, object]] = None
_DASHBOARD_ELEMENTS: Optional[Dict[str, object]] = None
_SIDEBAR_LISTENER_ATTACHED = False
_WARNINGS_PLACEHOLDER = None


def _ensure_dashboard_elements(metrics_container: object, decisions_container: object) -> Dict[str, object]:
    # Create dedicated placeholders so successive updates rewrite the same slots
    metrics_calls = metrics_container.empty()
    metrics_prompt = metrics_container.empty()
    metrics_completion = metrics_container.empty()

    elements = {
        "metrics_calls": metrics_calls,
        "metrics_prompt": metrics_prompt,
        "metrics_completion": metrics_completion,
        "provider_distribution": metrics_container.empty(),
        "model_distribution": metrics_container.empty(),
        "decisions_list": decisions_container.empty(),
    }
    return elements


def update_dashboard_sidebar(metrics: Optional[Dict[str, object]] = None) -> None:
    """Refresh sidebar metrics and warnings."""
    global _DASHBOARD_CONTAINERS
    global _DASHBOARD_ELEMENTS
    global _WARNINGS_PLACEHOLDER

    containers = _DASHBOARD_CONTAINERS
    if not containers:
        return
    metrics_container, decisions_container = containers
    elements = _DASHBOARD_ELEMENTS
    if elements is None:
        elements = _ensure_dashboard_elements(metrics_container, decisions_container)
        _DASHBOARD_ELEMENTS = elements

    if metrics is None:
        metrics = snapshot_llm_metrics()

    elements["metrics_calls"].metric("LLM 调用", metrics.get("total_calls", 0))
    elements["metrics_prompt"].metric("Prompt Tokens", metrics.get("total_prompt_tokens", 0))
    elements["metrics_completion"].metric("Completion Tokens", metrics.get("total_completion_tokens", 0))

    provider_calls = metrics.get("provider_calls", {})
    model_calls = metrics.get("model_calls", {})

    provider_placeholder = elements["provider_distribution"]
    provider_placeholder.empty()
    if provider_calls:
        provider_placeholder.json(provider_calls)
    else:
        provider_placeholder.info("暂无 Provider 分布数据。")

    model_placeholder = elements["model_distribution"]
    model_placeholder.empty()
    if model_calls:
        model_placeholder.json(model_calls)
    else:
        model_placeholder.info("暂无模型分布数据。")

    decisions = metrics.get("recent_decisions") or llm_recent_decisions(10)
    decisions_placeholder = elements["decisions_list"]
    decisions_placeholder.empty()
    if decisions:
        lines = []
        for record in reversed(decisions[-10:]):
            ts_code = record.get("ts_code")
            trade_date = record.get("trade_date")
            action = record.get("action")
            confidence = record.get("confidence", 0.0)
            summary = record.get("summary")
            line = f"**{trade_date} {ts_code}** → {action} (置信度 {confidence:.2f})"
            if summary:
                line += f"\n<small>{summary}</small>"
            lines.append(line)
        decisions_placeholder.markdown("\n\n".join(lines), unsafe_allow_html=True)
    else:
        decisions_placeholder.info("暂无决策记录。执行回测或实时评估后可在此查看。")

    if _WARNINGS_PLACEHOLDER is not None:
        _WARNINGS_PLACEHOLDER.empty()
        with _WARNINGS_PLACEHOLDER.container():
            st.subheader("数据告警")
            warn_list = alerts.get_warnings()
            if warn_list:
                lines = []
                for warning in warn_list[-10:]:
                    detail = warning.get("detail")
                    source = warning.get("source")
                    ts = warning.get("ts")
                    label = warning.get("label")
                    line = f"- **{source or '未知来源'}** {label or ''}"
                    if detail:
                        line += f"：{detail}"
                    if ts:
                        line += f"（{ts}）"
                    lines.append(line)
                st.markdown("\n".join(lines))
            else:
                st.caption("暂无数据告警。")


def _sidebar_metrics_listener(metrics: Dict[str, object]) -> None:
    try:
        update_dashboard_sidebar(metrics)
    except Exception:  # noqa: BLE001
        LOGGER.debug("侧边栏监听器刷新失败", exc_info=True, extra=LOG_EXTRA)


def render_global_dashboard() -> None:
    """Render a persistent sidebar with realtime LLM stats and recent decisions."""

    global _DASHBOARD_CONTAINERS
    global _DASHBOARD_ELEMENTS
    global _SIDEBAR_LISTENER_ATTACHED
    global _WARNINGS_PLACEHOLDER

    warnings = alerts.get_warnings()
    badge = f" ({len(warnings)})" if warnings else ""
    st.sidebar.header(f"系统监控{badge}")

    metrics_container = st.sidebar.container()
    decisions_container = st.sidebar.container()
    st.sidebar.container()  # legacy placeholder for layout spacing
    warn_placeholder = st.sidebar.empty()

    _DASHBOARD_CONTAINERS = (metrics_container, decisions_container)
    _DASHBOARD_ELEMENTS = _ensure_dashboard_elements(metrics_container, decisions_container)
    _WARNINGS_PLACEHOLDER = warn_placeholder

    if not _SIDEBAR_LISTENER_ATTACHED:
        register_llm_metrics_listener(_sidebar_metrics_listener)
        _SIDEBAR_LISTENER_ATTACHED = True
    update_dashboard_sidebar()
