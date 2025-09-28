"""Streamlit UI scaffold for the investment assistant."""
from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from requests.exceptions import RequestException
import streamlit as st

from app.agents.base import AgentContext
from app.agents.game import Decision
from app.backtest.engine import BtConfig, run_backtest
from app.data.schema import initialize_database
from app.ingest.checker import run_boot_check
from app.ingest.tushare import FetchJob, run_ingestion
from app.llm.client import llm_config_snapshot, run_llm
from app.llm.metrics import (
    reset as reset_llm_metrics,
    snapshot as snapshot_llm_metrics,
    recent_decisions as llm_recent_decisions,
)
from app.utils.config import (
    ALLOWED_LLM_STRATEGIES,
    DEFAULT_LLM_BASE_URLS,
    DEFAULT_LLM_MODEL_OPTIONS,
    DEFAULT_LLM_MODELS,
    DepartmentSettings,
    LLMEndpoint,
    LLMProvider,
    get_config,
    save_config,
)
from app.utils.db import db_session
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "ui"}


def render_global_dashboard() -> None:
    """Render a persistent sidebar with realtime LLM stats and recent decisions."""

    metrics_container = st.sidebar.container()
    decisions_container = st.sidebar.container()
    st.session_state["dashboard_placeholders"] = (metrics_container, decisions_container)
    _update_dashboard_sidebar()


def _update_dashboard_sidebar(metrics: Optional[Dict[str, object]] = None) -> None:
    placeholders = st.session_state.get("dashboard_placeholders")
    if not placeholders:
        return
    metrics_container, decisions_container = placeholders
    metrics = metrics or snapshot_llm_metrics()

    metrics_container.empty()
    with metrics_container.container():
        st.header("系统监控")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("LLM 调用", metrics.get("total_calls", 0))
        col_b.metric("Prompt Tokens", metrics.get("total_prompt_tokens", 0))
        col_c.metric("Completion Tokens", metrics.get("total_completion_tokens", 0))

        provider_calls = metrics.get("provider_calls", {})
        model_calls = metrics.get("model_calls", {})
        if provider_calls or model_calls:
            with st.expander("调用分布", expanded=False):
                if provider_calls:
                    st.write("按 Provider：")
                    st.json(provider_calls)
                if model_calls:
                    st.write("按模型：")
                    st.json(model_calls)

    decisions_container.empty()
    with decisions_container.container():
        st.subheader("最新决策")
        decisions = metrics.get("recent_decisions") or llm_recent_decisions(10)
        if decisions:
            for record in reversed(decisions[-10:]):
                ts_code = record.get("ts_code")
                trade_date = record.get("trade_date")
                action = record.get("action")
                confidence = record.get("confidence", 0.0)
                summary = record.get("summary")
                st.markdown(
                    f"**{trade_date} {ts_code}** → {action} (置信度 {confidence:.2f})"
                )
                if summary:
                    st.caption(summary)
        else:
            st.caption("暂无决策记录。执行回测或实时评估后可在此查看。")

def _discover_provider_models(provider: LLMProvider, base_override: str = "", api_override: Optional[str] = None) -> tuple[list[str], Optional[str]]:
    """Attempt to query provider API and return available model ids."""

    base_url = (base_override or provider.base_url or DEFAULT_LLM_BASE_URLS.get(provider.key, "")).strip()
    if not base_url:
        return [], "请先填写 Base URL"
    timeout = float(provider.default_timeout or 30.0)
    mode = provider.mode or ("ollama" if provider.key == "ollama" else "openai")

    try:
        if mode == "ollama":
            url = base_url.rstrip('/') + "/api/tags"
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            models = []
            for item in data.get("models", []) or data.get("data", []):
                name = item.get("name") or item.get("model") or item.get("tag")
                if name:
                    models.append(str(name).strip())
            return sorted(set(models)), None

        api_key = (api_override or provider.api_key or "").strip()
        if not api_key:
            return [], "缺少 API Key"
        url = base_url.rstrip('/') + "/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        models = [
            str(item.get("id")).strip()
            for item in payload.get("data", [])
            if item.get("id")
        ]
        return sorted(set(models)), None
    except RequestException as exc:  # noqa: BLE001
        return [], f"HTTP 错误：{exc}"
    except Exception as exc:  # noqa: BLE001
        return [], f"解析失败：{exc}"

def _load_stock_options(limit: int = 500) -> list[str]:
    try:
        with db_session(read_only=True) as conn:
            rows = conn.execute(
                "SELECT ts_code, name FROM stock_basic WHERE list_status = 'L' ORDER BY ts_code"
            ).fetchall()
    except Exception:
        LOGGER.exception("加载股票列表失败", extra=LOG_EXTRA)
        return []
    options: list[str] = []
    for row in rows[:limit]:
        code = row["ts_code"]
        name = row["name"] or ""
        label = f"{code} | {name}" if name else code
        options.append(label)
    LOGGER.info("加载股票选项完成，数量=%s", len(options), extra=LOG_EXTRA)
    return options


def _parse_ts_code(selection: str) -> str:
    return selection.split(' | ')[0].strip().upper()


def _load_daily_frame(ts_code: str, start: date, end: date) -> pd.DataFrame:
    LOGGER.info(
        "加载行情数据：ts_code=%s start=%s end=%s",
        ts_code,
        start,
        end,
        extra=LOG_EXTRA,
    )
    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')
    range_query = (
        "SELECT trade_date, open, high, low, close, vol, amount "
        "FROM daily WHERE ts_code = ? AND trade_date BETWEEN ? AND ? ORDER BY trade_date"
    )
    fallback_query = (
        "SELECT trade_date, open, high, low, close, vol, amount "
        "FROM daily WHERE ts_code = ? ORDER BY trade_date DESC LIMIT 200"
    )
    with db_session(read_only=True) as conn:
        df = pd.read_sql_query(range_query, conn, params=(ts_code, start_str, end_str))
        if df.empty:
            df = pd.read_sql_query(fallback_query, conn, params=(ts_code,))
            if df.empty:
                LOGGER.warning(
                    "行情数据为空：ts_code=%s start=%s end=%s",
                    ts_code,
                    start,
                    end,
                    extra=LOG_EXTRA,
                )
                return df
            df = df.sort_values('trade_date')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    LOGGER.info("行情数据加载完成：条数=%s", len(df), extra=LOG_EXTRA)
    return df


def render_today_plan() -> None:
    LOGGER.info("渲染今日计划页面", extra=LOG_EXTRA)
    st.header("今日计划")
    st.caption("统计与决策概览现已移至左侧“系统监控”侧栏。")
    try:
        with db_session(read_only=True) as conn:
            date_rows = conn.execute(
                """
                SELECT DISTINCT trade_date
                FROM agent_utils
                ORDER BY trade_date DESC
                LIMIT 30
                """
            ).fetchall()
    except Exception:  # noqa: BLE001
        LOGGER.exception("加载 agent_utils 失败", extra=LOG_EXTRA)
        st.warning("暂未写入部门/代理决策，请先运行回测或策略评估流程。")
        return

    trade_dates = [row["trade_date"] for row in date_rows]
    if not trade_dates:
        st.info("暂无决策记录，完成一次回测后即可在此查看部门意见与投票结果。")
        return

    trade_date = st.selectbox("交易日", trade_dates, index=0)

    with db_session(read_only=True) as conn:
        code_rows = conn.execute(
            """
            SELECT DISTINCT ts_code
            FROM agent_utils
            WHERE trade_date = ?
            ORDER BY ts_code
            """,
            (trade_date,),
        ).fetchall()
    symbols = [row["ts_code"] for row in code_rows]
    if not symbols:
        st.info("所选交易日暂无 agent_utils 记录。")
        return

    ts_code = st.selectbox("标的", symbols, index=0)

    with db_session(read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT agent, action, utils, feasible, weight
            FROM agent_utils
            WHERE trade_date = ? AND ts_code = ?
            ORDER BY CASE WHEN agent = 'global' THEN 1 ELSE 0 END, agent
            """,
            (trade_date, ts_code),
        ).fetchall()

    if not rows:
        st.info("未查询到详细决策记录，稍后再试。")
        return

    try:
        feasible_actions = json.loads(rows[0]["feasible"] or "[]")
    except (KeyError, TypeError, json.JSONDecodeError):
        feasible_actions = []

    global_info = None
    dept_records: List[Dict[str, object]] = []
    dept_details: Dict[str, Dict[str, object]] = {}
    agent_records: List[Dict[str, object]] = []

    for item in rows:
        agent_name = item["agent"]
        action = item["action"]
        weight = float(item["weight"] or 0.0)
        try:
            utils = json.loads(item["utils"] or "{}")
        except json.JSONDecodeError:
            utils = {}

        if agent_name == "global":
            global_info = {
                "action": action,
                "confidence": float(utils.get("_confidence", 0.0)),
                "target_weight": float(utils.get("_target_weight", 0.0)),
                "department_votes": utils.get("_department_votes", {}),
                "requires_review": bool(utils.get("_requires_review", False)),
                "scope_values": utils.get("_scope_values", {}),
                "close_series": utils.get("_close_series", []),
                "turnover_series": utils.get("_turnover_series", []),
                "department_supplements": utils.get("_department_supplements", {}),
                "department_dialogue": utils.get("_department_dialogue", {}),
            }
            continue

        if agent_name.startswith("dept_"):
            code = agent_name.split("dept_", 1)[-1]
            signals = utils.get("_signals", [])
            risks = utils.get("_risks", [])
            supplements = utils.get("_supplements", [])
            dialogue = utils.get("_dialogue", [])
            dept_records.append(
                {
                    "部门": code,
                    "行动": action,
                    "信心": float(utils.get("_confidence", 0.0)),
                    "权重": weight,
                    "摘要": utils.get("_summary", ""),
                    "核心信号": "；".join(signals) if isinstance(signals, list) else signals,
                    "风险提示": "；".join(risks) if isinstance(risks, list) else risks,
                    "补充次数": len(supplements) if isinstance(supplements, list) else 0,
                }
            )
            dept_details[code] = {
                "supplements": supplements if isinstance(supplements, list) else [],
                "dialogue": dialogue if isinstance(dialogue, list) else [],
                "summary": utils.get("_summary", ""),
                "signals": signals,
                "risks": risks,
            }
        else:
            score_map = {
                key: float(val)
                for key, val in utils.items()
                if not str(key).startswith("_")
            }
            agent_records.append(
                {
                    "代理": agent_name,
                    "建议动作": action,
                    "权重": weight,
                    "SELL": score_map.get("SELL", 0.0),
                    "HOLD": score_map.get("HOLD", 0.0),
                    "BUY_S": score_map.get("BUY_S", 0.0),
                    "BUY_M": score_map.get("BUY_M", 0.0),
                    "BUY_L": score_map.get("BUY_L", 0.0),
                }
            )

    if feasible_actions:
        st.caption(f"可行操作集合：{', '.join(feasible_actions)}")

    st.subheader("全局策略")
    if global_info:
        col1, col2, col3 = st.columns(3)
        col1.metric("最终行动", global_info["action"])
        col2.metric("信心", f"{global_info['confidence']:.2f}")
        col3.metric("目标权重", f"{global_info['target_weight']:+.2%}")
        if global_info["department_votes"]:
            st.json(global_info["department_votes"])
        if global_info["requires_review"]:
            st.warning("部门分歧较大，已标记为需人工复核。")
        with st.expander("基础上下文数据", expanded=False):
            if global_info.get("scope_values"):
                st.write("最新字段：")
                st.json(global_info["scope_values"])
            if global_info.get("close_series"):
                st.write("收盘价时间序列 (最近窗口)：")
                st.json(global_info["close_series"])
            if global_info.get("turnover_series"):
                st.write("换手率时间序列 (最近窗口)：")
                st.json(global_info["turnover_series"])
        dept_sup = global_info.get("department_supplements") or {}
        dept_dialogue = global_info.get("department_dialogue") or {}
        if dept_sup or dept_dialogue:
            with st.expander("部门补数与对话记录", expanded=False):
                if dept_sup:
                    st.write("补充数据：")
                    st.json(dept_sup)
                if dept_dialogue:
                    st.write("对话片段：")
                    st.json(dept_dialogue)
    else:
        st.info("暂未写入全局策略摘要。")

    st.subheader("部门意见")
    if dept_records:
        dept_df = pd.DataFrame(dept_records)
        st.dataframe(dept_df, width='stretch', hide_index=True)
        for code, details in dept_details.items():
            with st.expander(f"{code} 补充详情", expanded=False):
                supplements = details.get("supplements", [])
                dialogue = details.get("dialogue", [])
                if supplements:
                    st.write("补充数据：")
                    st.json(supplements)
                else:
                    st.caption("无补充数据请求。")
                if dialogue:
                    st.write("对话记录：")
                    for idx, line in enumerate(dialogue, start=1):
                        st.markdown(f"**回合 {idx}:** {line}")
                else:
                    st.caption("无额外对话。")
    else:
        st.info("暂无部门记录。")

    st.subheader("代理评分")
    if agent_records:
        agent_df = pd.DataFrame(agent_records)
        st.dataframe(agent_df, width='stretch', hide_index=True)
    else:
        st.info("暂无基础代理评分。")

    st.caption("以上内容来源于 agent_utils 表，可通过回测或实时评估自动更新。")


def render_backtest() -> None:
    LOGGER.info("渲染回测页面", extra=LOG_EXTRA)
    st.header("回测与复盘")
    st.write("在此运行回测、展示净值曲线与代理贡献。")

    default_start = date(2020, 1, 1)
    default_end = date(2020, 3, 31)
    LOGGER.debug(
        "回测默认参数：start=%s end=%s universe=%s target=%s stop=%s hold_days=%s",
        default_start,
        default_end,
        "000001.SZ",
        0.035,
        -0.015,
        10,
        extra=LOG_EXTRA,
    )

    col1, col2 = st.columns(2)
    start_date = col1.date_input("开始日期", value=default_start)
    end_date = col2.date_input("结束日期", value=default_end)
    universe_text = st.text_input("股票列表（逗号分隔）", value="000001.SZ")
    target = st.number_input("目标收益（例：0.035 表示 3.5%）", value=0.035, step=0.005, format="%.3f")
    stop = st.number_input("止损收益（例：-0.015 表示 -1.5%）", value=-0.015, step=0.005, format="%.3f")
    hold_days = st.number_input("持有期（交易日）", value=10, step=1)
    LOGGER.debug(
        "当前回测表单输入：start=%s end=%s universe_text=%s target=%.3f stop=%.3f hold_days=%s",
        start_date,
        end_date,
        universe_text,
        target,
        stop,
        hold_days,
        extra=LOG_EXTRA,
    )

    if st.button("运行回测"):
        LOGGER.info("用户点击运行回测按钮", extra=LOG_EXTRA)
        decision_log_container = st.container()
        status_box = st.status("准备执行回测...", expanded=True)
        llm_stats_placeholder = st.empty()
        decision_entries: List[str] = []

        def _decision_callback(ts_code: str, trade_dt: date, ctx: AgentContext, decision: Decision) -> None:
            ts_label = trade_dt.isoformat()
            summary = ""
            for dept_decision in decision.department_decisions.values():
                if getattr(dept_decision, "summary", ""):
                    summary = str(dept_decision.summary)
                    break
            entry_lines = [
                f"**{ts_label} {ts_code}** → {decision.action.value} (信心 {decision.confidence:.2f})",
            ]
            if summary:
                entry_lines.append(f"摘要：{summary}")
            dep_highlights = []
            for dept_code, dept_decision in decision.department_decisions.items():
                dep_highlights.append(
                    f"{dept_code}:{dept_decision.action.value}({dept_decision.confidence:.2f})"
                )
            if dep_highlights:
                entry_lines.append("部门意见：" + "；".join(dep_highlights))
            decision_entries.append("  \n".join(entry_lines))
            decision_log_container.markdown("\n\n".join(decision_entries[-200:]))
            status_box.write(f"{ts_label} {ts_code} → {decision.action.value} (信心 {decision.confidence:.2f})")
            stats = snapshot_llm_metrics()
            llm_stats_placeholder.json(
                {
                    "LLM 调用次数": stats.get("total_calls", 0),
                    "Prompt Tokens": stats.get("total_prompt_tokens", 0),
                    "Completion Tokens": stats.get("total_completion_tokens", 0),
                    "按 Provider": stats.get("provider_calls", {}),
                    "按模型": stats.get("model_calls", {}),
                }
            )
            _update_dashboard_sidebar(stats)

        reset_llm_metrics()
        status_box.update(label="执行回测中...", state="running")
        try:
            universe = [code.strip() for code in universe_text.split(',') if code.strip()]
            LOGGER.info(
                "回测参数：start=%s end=%s universe=%s target=%s stop=%s hold_days=%s",
                start_date,
                end_date,
                universe,
                target,
                stop,
                hold_days,
                extra=LOG_EXTRA,
            )
            cfg = BtConfig(
                id="streamlit_demo",
                name="Streamlit Demo Strategy",
                start_date=start_date,
                end_date=end_date,
                universe=universe,
                params={
                    "target": target,
                    "stop": stop,
                    "hold_days": int(hold_days),
                },
            )
            result = run_backtest(cfg, decision_callback=_decision_callback)
            LOGGER.info(
                "回测完成：nav_records=%s trades=%s",
                len(result.nav_series),
                len(result.trades),
                extra=LOG_EXTRA,
            )
            status_box.update(label="回测执行完成", state="complete")
            st.success("回测执行完成，详见下方结果与统计。")
            metrics = snapshot_llm_metrics()
            llm_stats_placeholder.json(
                {
                    "LLM 调用次数": metrics.get("total_calls", 0),
                    "Prompt Tokens": metrics.get("total_prompt_tokens", 0),
                    "Completion Tokens": metrics.get("total_completion_tokens", 0),
                    "按 Provider": metrics.get("provider_calls", {}),
                    "按模型": metrics.get("model_calls", {}),
                }
            )
            _update_dashboard_sidebar(metrics)
            st.json({"nav_records": result.nav_series, "trades": result.trades})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("回测执行失败", extra=LOG_EXTRA)
            status_box.update(label="回测执行失败", state="error")
            st.error(f"回测执行失败：{exc}")


def render_settings() -> None:
    LOGGER.info("渲染设置页面", extra=LOG_EXTRA)
    st.header("数据与设置")
    cfg = get_config()
    LOGGER.debug("当前 TuShare Token 是否已配置=%s", bool(cfg.tushare_token), extra=LOG_EXTRA)
    token = st.text_input("TuShare Token", value=cfg.tushare_token or "", type="password")

    if st.button("保存设置"):
        LOGGER.info("保存设置按钮被点击", extra=LOG_EXTRA)
        cfg.tushare_token = token.strip() or None
        LOGGER.info("TuShare Token 更新，是否为空=%s", cfg.tushare_token is None, extra=LOG_EXTRA)
        save_config()
        st.success("设置已保存，仅在当前会话生效。")

    st.write("新闻源开关与数据库备份将在此配置。")

    st.divider()
    st.subheader("LLM 设置")
    providers = cfg.llm_providers
    provider_keys = sorted(providers.keys())
    st.caption("先在 Provider 中维护基础连接（URL、Key、模型），再为全局与各部门设置个性化参数。")

    # Provider management -------------------------------------------------
    provider_select_col, provider_manage_col = st.columns([3, 1])
    if provider_keys:
        try:
            default_provider = cfg.llm.primary.provider or provider_keys[0]
            provider_index = provider_keys.index(default_provider)
        except ValueError:
            provider_index = 0
        selected_provider = provider_select_col.selectbox(
            "选择 Provider",
            provider_keys,
            index=provider_index,
            key="llm_provider_select",
        )
    else:
        selected_provider = None
        provider_select_col.info("尚未配置 Provider，请先创建。")

    new_provider_name = provider_manage_col.text_input("新增 Provider", key="new_provider_name")
    if provider_manage_col.button("创建 Provider", key="create_provider_btn"):
        key = (new_provider_name or "").strip().lower()
        if not key:
            st.warning("请输入有效的 Provider 名称。")
        elif key in providers:
            st.warning(f"Provider {key} 已存在。")
        else:
            providers[key] = LLMProvider(key=key)
            cfg.llm_providers = providers
            save_config()
            st.success(f"已创建 Provider {key}。")
            st.rerun()

    if selected_provider:
        provider_cfg = providers.get(selected_provider, LLMProvider(key=selected_provider))
        title_key = f"provider_title_{selected_provider}"
        base_key = f"provider_base_{selected_provider}"
        api_key_key = f"provider_api_{selected_provider}"
        default_model_key = f"provider_default_model_{selected_provider}"
        mode_key = f"provider_mode_{selected_provider}"
        temp_key = f"provider_temp_{selected_provider}"
        timeout_key = f"provider_timeout_{selected_provider}"
        prompt_key = f"provider_prompt_{selected_provider}"
        enabled_key = f"provider_enabled_{selected_provider}"

        title_val = st.text_input("备注名称", value=provider_cfg.title or "", key=title_key)
        base_val = st.text_input("Base URL", value=provider_cfg.base_url or "", key=base_key, help="调用地址，例如：https://api.openai.com")
        api_val = st.text_input("API Key", value=provider_cfg.api_key or "", key=api_key_key, type="password")
        st.markdown("可用模型：")
        if provider_cfg.models:
            st.code("\n".join(provider_cfg.models), language="text")
        else:
            st.info("尚未获取模型列表，可点击下方按钮自动拉取。")

        model_choice_key = f"{default_model_key}_choice"
        if provider_cfg.models:
            options = provider_cfg.models + ["自定义"]
            default_choice = provider_cfg.default_model if provider_cfg.default_model in provider_cfg.models else "自定义"
            model_choice = st.selectbox("默认模型", options, index=options.index(default_choice), key=model_choice_key)
            if model_choice == "自定义":
                default_model_val = st.text_input("自定义默认模型", value=provider_cfg.default_model or "", key=default_model_key).strip() or None
            else:
                default_model_val = model_choice
        else:
            default_model_val = st.text_input("默认模型", value=provider_cfg.default_model or "", key=default_model_key).strip() or None
        mode_val = st.selectbox("调用模式", ["openai", "ollama"], index=0 if provider_cfg.mode == "openai" else 1, key=mode_key)
        temp_val = st.slider("默认温度", min_value=0.0, max_value=2.0, value=float(provider_cfg.default_temperature), step=0.05, key=temp_key)
        timeout_val = st.number_input("默认超时(秒)", min_value=5, max_value=300, value=int(provider_cfg.default_timeout or 30), step=5, key=timeout_key)
        prompt_template_val = st.text_area("默认 Prompt 模板（可选，使用 {prompt} 占位）", value=provider_cfg.prompt_template or "", key=prompt_key, height=120)
        enabled_val = st.checkbox("启用", value=provider_cfg.enabled, key=enabled_key)

        fetch_key = f"fetch_models_{selected_provider}"
        if st.button("获取模型列表", key=fetch_key):
            with st.spinner("正在获取模型列表..."):
                models, error = _discover_provider_models(provider_cfg, base_val, api_val)
            if error:
                st.error(error)
            else:
                provider_cfg.models = models
                if models and (not provider_cfg.default_model or provider_cfg.default_model not in models):
                    provider_cfg.default_model = models[0]
                providers[selected_provider] = provider_cfg
                cfg.llm_providers = providers
                cfg.sync_runtime_llm()
                save_config()
                st.success(f"共获取 {len(models)} 个模型。")
                st.rerun()

        if st.button("保存 Provider", key=f"save_provider_{selected_provider}"):
            provider_cfg.title = title_val.strip()
            provider_cfg.base_url = base_val.strip()
            provider_cfg.api_key = api_val.strip() or None
            if provider_cfg.models and default_model_val in provider_cfg.models:
                provider_cfg.default_model = default_model_val
            else:
                provider_cfg.default_model = default_model_val
            provider_cfg.default_temperature = float(temp_val)
            provider_cfg.default_timeout = float(timeout_val)
            provider_cfg.prompt_template = prompt_template_val.strip()
            provider_cfg.enabled = enabled_val
            provider_cfg.mode = mode_val
            providers[selected_provider] = provider_cfg
            cfg.llm_providers = providers
            cfg.sync_runtime_llm()
            save_config()
            st.success("Provider 已保存。")
            st.session_state[title_key] = provider_cfg.title or ""
            st.session_state[default_model_key] = provider_cfg.default_model or ""

        provider_in_use = (cfg.llm.primary.provider == selected_provider) or any(
            ep.provider == selected_provider for ep in cfg.llm.ensemble
        )
        if not provider_in_use:
            for dept in cfg.departments.values():
                if dept.llm.primary.provider == selected_provider or any(ep.provider == selected_provider for ep in dept.llm.ensemble):
                    provider_in_use = True
                    break
        if st.button(
            "删除 Provider",
            key=f"delete_provider_{selected_provider}",
            disabled=provider_in_use or len(providers) <= 1,
        ):
            providers.pop(selected_provider, None)
            cfg.llm_providers = providers
            cfg.sync_runtime_llm()
            save_config()
            st.success("Provider 已删除。")
            st.rerun()

    st.markdown("##### 全局推理配置")
    if not provider_keys:
        st.warning("请先配置至少一个 Provider。")
    else:
        global_cfg = cfg.llm
        primary = global_cfg.primary
        try:
            provider_index = provider_keys.index(primary.provider or provider_keys[0])
        except ValueError:
            provider_index = 0
        selected_global_provider = st.selectbox(
            "主模型 Provider",
            provider_keys,
            index=provider_index,
            key="global_provider_select",
        )
        provider_cfg = providers.get(selected_global_provider)
        available_models = provider_cfg.models if provider_cfg else []
        default_model = primary.model or (provider_cfg.default_model if provider_cfg else None)
        if available_models:
            options = available_models + ["自定义"]
            try:
                model_index = available_models.index(default_model)
                model_choice = st.selectbox("主模型", options, index=model_index, key="global_model_choice")
            except ValueError:
                model_choice = st.selectbox("主模型", options, index=len(options) - 1, key="global_model_choice")
            if model_choice == "自定义":
                model_val = st.text_input("自定义模型", value=default_model or "", key="global_model_custom").strip()
            else:
                model_val = model_choice
        else:
            model_val = st.text_input("主模型", value=default_model or "", key="global_model_custom").strip()

        temp_default = primary.temperature if primary.temperature is not None else (provider_cfg.default_temperature if provider_cfg else 0.2)
        temp_val = st.slider("主模型温度", min_value=0.0, max_value=2.0, value=float(temp_default), step=0.05, key="global_temp")
        timeout_default = primary.timeout if primary.timeout is not None else (provider_cfg.default_timeout if provider_cfg else 30.0)
        timeout_val = st.number_input("主模型超时(秒)", min_value=5, max_value=300, value=int(timeout_default), step=5, key="global_timeout")
        prompt_template_val = st.text_area(
            "主模型 Prompt 模板（可选）",
            value=primary.prompt_template or provider_cfg.prompt_template if provider_cfg else "",
            height=120,
            key="global_prompt_template",
        )

        strategy_val = st.selectbox("推理策略", sorted(ALLOWED_LLM_STRATEGIES), index=sorted(ALLOWED_LLM_STRATEGIES).index(global_cfg.strategy) if global_cfg.strategy in ALLOWED_LLM_STRATEGIES else 0, key="global_strategy")
        show_ensemble = strategy_val != "single"
        majority_threshold_val = st.number_input(
            "多数投票门槛",
            min_value=1,
            max_value=10,
            value=int(global_cfg.majority_threshold),
            step=1,
            key="global_majority",
            disabled=not show_ensemble,
        )
        if not show_ensemble:
            majority_threshold_val = 1

        ensemble_rows: List[Dict[str, str]] = []
        if show_ensemble:
            ensemble_rows = [
                {
                    "provider": ep.provider,
                    "model": ep.model or "",
                    "temperature": "" if ep.temperature is None else f"{ep.temperature:.3f}",
                    "timeout": "" if ep.timeout is None else str(int(ep.timeout)),
                    "prompt_template": ep.prompt_template or "",
                }
                for ep in global_cfg.ensemble
            ] or [{"provider": primary.provider or selected_global_provider, "model": "", "temperature": "", "timeout": "", "prompt_template": ""}]

            ensemble_editor = st.data_editor(
                ensemble_rows,
                num_rows="dynamic",
                key="global_ensemble_editor",
                width='stretch',
                hide_index=True,
                column_config={
                    "provider": st.column_config.SelectboxColumn("Provider", options=provider_keys),
                    "model": st.column_config.TextColumn("模型"),
                    "temperature": st.column_config.TextColumn("温度"),
                    "timeout": st.column_config.TextColumn("超时(秒)"),
                    "prompt_template": st.column_config.TextColumn("Prompt 模板"),
                },
            )
            if hasattr(ensemble_editor, "to_dict"):
                ensemble_rows = ensemble_editor.to_dict("records")
            else:
                ensemble_rows = ensemble_editor
        else:
            st.info("当前策略为单模型，未启用协作模型。")

        if st.button("保存全局配置", key="save_global_llm"):
            primary.provider = selected_global_provider
            primary.model = model_val or None
            primary.temperature = float(temp_val)
            primary.timeout = float(timeout_val)
            primary.prompt_template = prompt_template_val.strip() or None
            primary.base_url = None
            primary.api_key = None

            new_ensemble: List[LLMEndpoint] = []
            if show_ensemble:
                for row in ensemble_rows:
                    provider_val = (row.get("provider") or "").strip().lower()
                    if not provider_val:
                        continue
                    model_raw = (row.get("model") or "").strip() or None
                    temp_raw = (row.get("temperature") or "").strip()
                    timeout_raw = (row.get("timeout") or "").strip()
                    prompt_raw = (row.get("prompt_template") or "").strip()
                    new_ensemble.append(
                        LLMEndpoint(
                            provider=provider_val,
                            model=model_raw,
                            temperature=float(temp_raw) if temp_raw else None,
                            timeout=float(timeout_raw) if timeout_raw else None,
                            prompt_template=prompt_raw or None,
                        )
                    )
            cfg.llm.ensemble = new_ensemble
            cfg.llm.strategy = strategy_val
            cfg.llm.majority_threshold = int(majority_threshold_val)
            cfg.sync_runtime_llm()
            save_config()
            st.success("全局 LLM 配置已保存。")
            st.json(llm_config_snapshot())

    # Department configuration -------------------------------------------
    st.markdown("##### 部门配置")
    dept_settings = cfg.departments or {}
    dept_rows = [
        {
            "code": code,
            "title": dept.title,
            "description": dept.description,
            "weight": float(dept.weight),
            "strategy": dept.llm.strategy,
            "majority_threshold": dept.llm.majority_threshold,
            "provider": dept.llm.primary.provider or (provider_keys[0] if provider_keys else ""),
            "model": dept.llm.primary.model or "",
            "temperature": "" if dept.llm.primary.temperature is None else f"{dept.llm.primary.temperature:.3f}",
            "timeout": "" if dept.llm.primary.timeout is None else str(int(dept.llm.primary.timeout)),
            "prompt_template": dept.llm.primary.prompt_template or "",
        }
        for code, dept in sorted(dept_settings.items())
    ]

    if not dept_rows:
        st.info("当前未配置部门，可在 config.json 中添加。")
        dept_rows = []

    dept_editor = st.data_editor(
        dept_rows,
        num_rows="fixed",
        key="department_editor",
        width='stretch',
        hide_index=True,
        column_config={
            "code": st.column_config.TextColumn("编码", disabled=True),
            "title": st.column_config.TextColumn("名称"),
            "description": st.column_config.TextColumn("说明"),
            "weight": st.column_config.NumberColumn("权重", min_value=0.0, max_value=10.0, step=0.1),
            "strategy": st.column_config.SelectboxColumn("策略", options=sorted(ALLOWED_LLM_STRATEGIES)),
            "majority_threshold": st.column_config.NumberColumn("投票阈值", min_value=1, max_value=10, step=1),
            "provider": st.column_config.SelectboxColumn("Provider", options=provider_keys or [""]),
            "model": st.column_config.TextColumn("模型"),
            "temperature": st.column_config.TextColumn("温度"),
            "timeout": st.column_config.TextColumn("超时(秒)"),
            "prompt_template": st.column_config.TextColumn("Prompt 模板"),
        },
    )

    if hasattr(dept_editor, "to_dict"):
        dept_rows = dept_editor.to_dict("records")
    else:
        dept_rows = dept_editor

    col_reset, col_save = st.columns([1, 1])

    if col_save.button("保存部门配置"):
        updated_departments: Dict[str, DepartmentSettings] = {}
        for row in dept_rows:
            code = row.get("code")
            if not code:
                continue
            existing = dept_settings.get(code) or DepartmentSettings(code=code, title=code)
            existing.title = row.get("title") or existing.title
            existing.description = row.get("description") or ""
            try:
                existing.weight = max(0.0, float(row.get("weight", existing.weight)))
            except (TypeError, ValueError):
                pass

            strategy_val = (row.get("strategy") or existing.llm.strategy).lower()
            if strategy_val in ALLOWED_LLM_STRATEGIES:
                existing.llm.strategy = strategy_val
            if existing.llm.strategy == "single":
                existing.llm.majority_threshold = 1
                existing.llm.ensemble = []
            else:
                majority_raw = row.get("majority_threshold")
                try:
                    majority_val = int(majority_raw)
                    if majority_val > 0:
                        existing.llm.majority_threshold = majority_val
                except (TypeError, ValueError):
                    pass

            provider_val = (row.get("provider") or existing.llm.primary.provider or (provider_keys[0] if provider_keys else "ollama")).strip().lower()
            model_val = (row.get("model") or "").strip() or None
            temp_raw = (row.get("temperature") or "").strip()
            timeout_raw = (row.get("timeout") or "").strip()
            prompt_raw = (row.get("prompt_template") or "").strip()

            endpoint = existing.llm.primary or LLMEndpoint()
            endpoint.provider = provider_val
            endpoint.model = model_val
            endpoint.temperature = float(temp_raw) if temp_raw else None
            endpoint.timeout = float(timeout_raw) if timeout_raw else None
            endpoint.prompt_template = prompt_raw or None
            endpoint.base_url = None
            endpoint.api_key = None
            existing.llm.primary = endpoint
            if existing.llm.strategy != "single":
                existing.llm.ensemble = []

            updated_departments[code] = existing

        if updated_departments:
            cfg.departments = updated_departments
            cfg.sync_runtime_llm()
            save_config()
            st.success("部门配置已更新。")
        else:
            st.warning("未能解析部门配置输入。")

    if col_reset.button("恢复默认部门"):
        from app.utils.config import _default_departments

        cfg.departments = _default_departments()
        cfg.sync_runtime_llm()
        save_config()
        st.success("已恢复默认部门配置。")
        st.rerun()

    st.caption("部门配置存储为独立 LLM 参数，执行时会自动套用对应 Provider 的连接信息。")


def render_tests() -> None:
    LOGGER.info("渲染自检页面", extra=LOG_EXTRA)
    st.header("自检测试")
    st.write("用于快速检查数据库与数据拉取是否正常工作。")

    if st.button("测试数据库初始化"):
        LOGGER.info("点击测试数据库初始化按钮", extra=LOG_EXTRA)
        with st.spinner("正在检查数据库..."):
            result = initialize_database()
            if result.skipped:
                LOGGER.info("数据库已存在，无需初始化", extra=LOG_EXTRA)
                st.success("数据库已存在，检查通过。")
            else:
                LOGGER.info("数据库初始化完成，执行语句数=%s", result.executed, extra=LOG_EXTRA)
                st.success(f"数据库初始化完成，共执行 {result.executed} 条语句。")

    st.divider()

    if st.button("测试 TuShare 拉取（示例 2024-01-01 至 2024-01-03）"):
        LOGGER.info("点击示例 TuShare 拉取按钮", extra=LOG_EXTRA)
        with st.spinner("正在调用 TuShare 接口..."):
            try:
                run_ingestion(
                    FetchJob(
                        name="streamlit_self_test",
                        start=date(2024, 1, 1),
                        end=date(2024, 1, 3),
                        ts_codes=("000001.SZ",),
                    ),
                    include_limits=False,
                )
                LOGGER.info("示例 TuShare 拉取成功", extra=LOG_EXTRA)
                st.success("TuShare 示例拉取完成，数据已写入数据库。")
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("示例 TuShare 拉取失败", extra=LOG_EXTRA)
                st.error(f"拉取失败：{exc}")

    st.info("注意：TuShare 拉取依赖网络与 Token，若环境未配置将出现错误提示。")

    st.divider()
    days = int(st.number_input("检查窗口（天数）", min_value=30, max_value=1095, value=365, step=30))
    LOGGER.debug("检查窗口天数=%s", days, extra=LOG_EXTRA)
    cfg = get_config()
    force_refresh = st.checkbox(
        "强制刷新数据（关闭增量跳过）",
        value=cfg.force_refresh,
        help="勾选后将重新拉取所选区间全部数据",
    )
    if force_refresh != cfg.force_refresh:
        cfg.force_refresh = force_refresh
        LOGGER.info("更新 force_refresh=%s", force_refresh, extra=LOG_EXTRA)
        save_config()

    if st.button("执行开机检查"):
        LOGGER.info("点击执行开机检查按钮", extra=LOG_EXTRA)
        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()
        log_placeholder = st.empty()
        messages: list[str] = []

        def hook(message: str, value: float) -> None:
            progress_bar.progress(min(max(value, 0.0), 1.0))
            status_placeholder.write(message)
            messages.append(message)
            LOGGER.debug("开机检查进度：%s -> %.2f", message, value, extra=LOG_EXTRA)

        with st.spinner("正在执行开机检查..."):
            try:
                report = run_boot_check(
                    days=days,
                    progress_hook=hook,
                    force_refresh=force_refresh,
                )
                LOGGER.info("开机检查成功", extra=LOG_EXTRA)
                st.success("开机检查完成，以下为数据覆盖摘要。")
                st.json(report.to_dict())
                if messages:
                    log_placeholder.markdown("\n".join(f"- {msg}" for msg in messages))
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("开机检查失败", extra=LOG_EXTRA)
                st.error(f"开机检查失败：{exc}")
                if messages:
                    log_placeholder.markdown("\n".join(f"- {msg}" for msg in messages))
            finally:
                progress_bar.progress(1.0)

    st.divider()
    st.subheader("股票行情可视化")
    options = _load_stock_options()
    default_code = options[0] if options else "000001.SZ"

    if options:
        selection = st.selectbox("选择股票", options, index=0)
        ts_code = _parse_ts_code(selection)
        LOGGER.debug("选择股票：%s", ts_code, extra=LOG_EXTRA)
    else:
        ts_code = st.text_input("输入股票代码（如 000001.SZ）", value=default_code).strip().upper()
        LOGGER.debug("输入股票：%s", ts_code, extra=LOG_EXTRA)

    viz_col1, viz_col2 = st.columns(2)
    default_start = date.today() - timedelta(days=180)
    start_date = viz_col1.date_input("开始日期", value=default_start, key="viz_start")
    end_date = viz_col2.date_input("结束日期", value=date.today(), key="viz_end")
    LOGGER.debug("行情可视化日期范围：%s-%s", start_date, end_date, extra=LOG_EXTRA)

    if start_date > end_date:
        LOGGER.warning("无效日期范围：%s>%s", start_date, end_date, extra=LOG_EXTRA)
        st.error("开始日期不能晚于结束日期")
        return

    with st.spinner("正在加载行情数据..."):
        try:
            df = _load_daily_frame(ts_code, start_date, end_date)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("加载行情数据失败", extra=LOG_EXTRA)
            st.error(f"读取数据失败：{exc}")
            return

    if df.empty:
        LOGGER.warning("指定区间无行情数据：%s %s-%s", ts_code, start_date, end_date, extra=LOG_EXTRA)
        st.warning("未查询到该区间的交易数据，请确认数据库已拉取对应日线。")
        return

    price_df = df[["close"]].rename(columns={"close": "收盘价"})
    volume_df = df[["vol"]].rename(columns={"vol": "成交量(手)"})

    if price_df.shape[0] > 180:
        sampled = price_df.resample('3D').last().dropna()
    else:
        sampled = price_df

    if volume_df.shape[0] > 180:
        volume_sampled = volume_df.resample('3D').mean().dropna()
    else:
        volume_sampled = volume_df

    first_close = sampled.iloc[0, 0]
    last_close = sampled.iloc[-1, 0]
    delta_abs = last_close - first_close
    delta_pct = (delta_abs / first_close * 100) if first_close else 0.0

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("最新收盘价", f"{last_close:.2f}", delta=f"{delta_abs:+.2f}")
    metric_col2.metric("区间涨跌幅", f"{delta_pct:+.2f}%")
    metric_col3.metric("平均成交量", f"{volume_sampled['成交量(手)'].mean():.0f}")

    df_reset = df.reset_index().rename(columns={
        "trade_date": "交易日",
        "open": "开盘价",
        "high": "最高价",
        "low": "最低价",
        "close": "收盘价",
        "vol": "成交量(手)",
        "amount": "成交额(千元)",
    })
    df_reset["成交额(千元)"] = df_reset["成交额(千元)"] / 1000

    candle_fig = go.Figure(
        data=[
            go.Candlestick(
                x=df_reset["交易日"],
                open=df_reset["开盘价"],
                high=df_reset["最高价"],
                low=df_reset["最低价"],
                close=df_reset["收盘价"],
                name="K线",
            )
        ]
    )
    candle_fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(candle_fig, width='stretch')

    vol_fig = px.bar(
        df_reset,
        x="交易日",
        y="成交量(手)",
        labels={"成交量(手)": "成交量(手)"},
        title="成交量",
    )
    vol_fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(vol_fig, width='stretch')

    amt_fig = px.bar(
        df_reset,
        x="交易日",
        y="成交额(千元)",
        labels={"成交额(千元)": "成交额(千元)"},
        title="成交额",
    )
    amt_fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(amt_fig, width='stretch')

    df_reset["月份"] = df_reset["交易日"].dt.to_period("M").astype(str)
    box_fig = px.box(
        df_reset,
        x="月份",
        y="收盘价",
        points="outliers",
        title="月度收盘价分布",
    )
    box_fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(box_fig, width='stretch')

    st.caption("提示：成交量单位为手，成交额以千元显示。箱线图按月展示收盘价分布。")
    st.dataframe(df_reset.tail(20), width='stretch')
    LOGGER.info("行情可视化完成，展示行数=%s", len(df_reset), extra=LOG_EXTRA)

    st.divider()
    st.subheader("LLM 接口测试")
    st.json(llm_config_snapshot())
    llm_prompt = st.text_area("测试 Prompt", value="请概述今天的市场重点。", height=160)
    system_prompt = st.text_area(
        "System Prompt (可选)",
        value="你是一名量化策略研究助手，用简洁中文回答。",
        height=100,
    )
    if st.button("执行 LLM 测试"):
        with st.spinner("正在调用 LLM..."):
            try:
                response = run_llm(llm_prompt, system=system_prompt or None)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("LLM 测试失败", extra=LOG_EXTRA)
                st.error(f"LLM 调用失败：{exc}")
            else:
                LOGGER.info("LLM 测试成功", extra=LOG_EXTRA)
                st.success("LLM 调用成功，以下为返回内容：")
                st.write(response)


def main() -> None:
    LOGGER.info("初始化 Streamlit UI", extra=LOG_EXTRA)
    st.set_page_config(page_title="多智能体投资助理", layout="wide")
    render_global_dashboard()
    tabs = st.tabs(["今日计划", "回测与复盘", "数据与设置", "自检测试"])
    LOGGER.debug("Tabs 初始化完成：%s", ["今日计划", "回测与复盘", "数据与设置", "自检测试"], extra=LOG_EXTRA)
    with tabs[0]:
        render_today_plan()
    with tabs[1]:
        render_backtest()
    with tabs[2]:
        render_settings()
    with tabs[3]:
        render_tests()


if __name__ == "__main__":
    main()
