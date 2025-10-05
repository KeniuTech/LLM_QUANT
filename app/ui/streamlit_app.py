"""Streamlit UI scaffold for the investment assistant."""
from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from datetime import datetime
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from requests.exceptions import RequestException
import streamlit as st

from app.agents.base import AgentContext
from app.agents.game import Decision
from app.backtest.engine import BtConfig, run_backtest
from app.ui.portfolio_config import render_portfolio_config
from app.backtest.decision_env import DecisionEnv, ParameterSpec
from app.data.schema import initialize_database
from app.ingest.checker import run_boot_check
from app.ingest.tushare import FetchJob, run_ingestion
from app.llm.client import llm_config_snapshot, run_llm
from app.llm.metrics import (
    recent_decisions as llm_recent_decisions,
    register_listener as register_llm_metrics_listener,
    reset as reset_llm_metrics,
    snapshot as snapshot_llm_metrics,
)
from app.utils import alerts
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
from app.utils.portfolio import (
    get_latest_snapshot,
    list_investment_pool,
    list_positions,
    list_recent_trades,
)
from app.agents.registry import default_agents
from app.utils.tuning import log_tuning_result
from app.backtest.engine import BacktestEngine, PortfolioState


LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "ui"}
_DECISION_ENV_SINGLE_RESULT_KEY = "decision_env_single_result"
_DECISION_ENV_BATCH_RESULTS_KEY = "decision_env_batch_results"
_DASHBOARD_CONTAINERS: Optional[tuple[object, object]] = None
_DASHBOARD_ELEMENTS: Optional[Dict[str, object]] = None
_SIDEBAR_LISTENER_ATTACHED = False
# ADD: simple in-memory cache for provider model discovery
_MODEL_CACHE: Dict[str, Dict[str, object]] = {}
_CACHE_TTL_SECONDS = 300
_WARNINGS_CONTAINER = None
_WARNINGS_PLACEHOLDER = None

# ADD: query param helpers
def _get_query_params() -> Dict[str, List[str]]:
    try:
        return dict(st.query_params)
    except Exception:
        return {}

def _set_query_params(**kwargs: object) -> None:
    try:
        payload = {k: v for k, v in kwargs.items() if v is not None}
        if payload:
            st.query_params.update(payload)
    except Exception:
        pass


def _sidebar_metrics_listener(metrics: Dict[str, object]) -> None:
    try:
        _update_dashboard_sidebar(metrics)
    except Exception:  # noqa: BLE001
        LOGGER.debug("侧边栏监听器刷新失败", exc_info=True, extra=LOG_EXTRA)


def render_global_dashboard() -> None:
    """Render a persistent sidebar with realtime LLM stats and recent decisions."""

    global _DASHBOARD_CONTAINERS
    global _DASHBOARD_ELEMENTS
    global _SIDEBAR_LISTENER_ATTACHED
    global _WARNINGS_CONTAINER
    global _WARNINGS_PLACEHOLDER

    # ADD: warning badge on top
    warnings = alerts.get_warnings()
    badge = f" ({len(warnings)})" if warnings else ""
    st.sidebar.header(f"系统监控{badge}")

    metrics_container = st.sidebar.container()
    decisions_container = st.sidebar.container()
    _WARNINGS_CONTAINER = st.sidebar.container()
    _WARNINGS_PLACEHOLDER = st.sidebar.empty()
    _DASHBOARD_CONTAINERS = (metrics_container, decisions_container)
    _DASHBOARD_ELEMENTS = _ensure_dashboard_elements(metrics_container, decisions_container)
    if not _SIDEBAR_LISTENER_ATTACHED:
        register_llm_metrics_listener(_sidebar_metrics_listener)
        _SIDEBAR_LISTENER_ATTACHED = True
    _update_dashboard_sidebar()


def _update_dashboard_sidebar(
    metrics: Optional[Dict[str, object]] = None,
) -> None:
    global _DASHBOARD_CONTAINERS
    global _DASHBOARD_ELEMENTS
    global _WARNINGS_CONTAINER
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
    elements["metrics_completion"].metric(
        "Completion Tokens", metrics.get("total_completion_tokens", 0)
    )

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
        decisions_placeholder = elements["decisions_list"]
        decisions_placeholder.empty()
        decisions_placeholder.markdown("\n\n".join(lines), unsafe_allow_html=True)
    else:
        decisions_placeholder = elements["decisions_list"]
        decisions_placeholder.empty()
        decisions_placeholder.info("暂无决策记录。执行回测或实时评估后可在此查看。")
    # Render warnings section in-place (clear then write)
    if _WARNINGS_PLACEHOLDER is not None:
        _WARNINGS_PLACEHOLDER.empty()
        with _WARNINGS_PLACEHOLDER.container():
            st.subheader("数据告警")
            warn_list = alerts.get_warnings()
            if warn_list:
                lines = []
                for warning in warn_list[-10:]:
                    detail = warning.get("detail")
                    appendix = f" {detail}" if detail else ""
                    lines.append(
                        f"- **{warning['source']}** {warning['message']}{appendix}\n<small>{warning['timestamp']}</small>"
                    )
                st.markdown("\n".join(lines), unsafe_allow_html=True)
                btn_cols = st.columns([1,1])
                if btn_cols[0].button("清除数据告警", key="clear_data_alerts_sibling"):
                    alerts.clear_warnings()
                    _update_dashboard_sidebar()
                try:
                    st.download_button(
                        "导出告警(JSON)",
                        data=json.dumps(warn_list, ensure_ascii=False, indent=2),
                        file_name="data_warnings.json",
                        mime="application/json",
                        key="dl_warnings_json_sibling",
                    )
                except Exception:
                    pass
            else:
                st.info("暂无数据告警。")


def _ensure_dashboard_elements(metrics_container, decisions_container) -> Dict[str, object]:
    metrics_container.header("系统监控")
    col_a, col_b, col_c = metrics_container.columns(3)
    metrics_calls = col_a.empty()
    metrics_prompt = col_b.empty()
    metrics_completion = col_c.empty()
    distribution_expander = metrics_container.expander("调用分布", expanded=False)
    provider_distribution = distribution_expander.empty()
    model_distribution = distribution_expander.empty()

    decisions_container.subheader("最新决策")
    decisions_list = decisions_container.empty()

    elements = {
        "metrics_calls": metrics_calls,
        "metrics_prompt": metrics_prompt,
        "metrics_completion": metrics_completion,
        "provider_distribution": provider_distribution,
        "model_distribution": model_distribution,
        "decisions_list": decisions_list,
    }
    return elements

def _discover_provider_models(provider: LLMProvider, base_override: str = "", api_override: Optional[str] = None) -> tuple[list[str], Optional[str]]:
    """Attempt to query provider API and return available model ids."""

    base_url = (base_override or provider.base_url or DEFAULT_LLM_BASE_URLS.get(provider.key, "")).strip()
    if not base_url:
        return [], "请先填写 Base URL"
    timeout = float(provider.default_timeout or 30.0)
    mode = provider.mode or ("ollama" if provider.key == "ollama" else "openai")

    # ADD: simple cache by provider+base URL
    cache_key = f"{provider.key}|{base_url}"
    now = datetime.now()
    cached = _MODEL_CACHE.get(cache_key)
    if cached:
        ts = cached.get("ts")
        if isinstance(ts, float) and (now.timestamp() - ts) < _CACHE_TTL_SECONDS:
            models = list(cached.get("models") or [])
            return models, None

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
            _MODEL_CACHE[cache_key] = {"ts": now.timestamp(), "models": sorted(set(models))}
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
        _MODEL_CACHE[cache_key] = {"ts": now.timestamp(), "models": sorted(set(models))}
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


def _get_latest_trade_date() -> Optional[date]:
    try:
        with db_session(read_only=True) as conn:
            row = conn.execute(
                "SELECT trade_date FROM daily ORDER BY trade_date DESC LIMIT 1"
            ).fetchone()
    except Exception:  # noqa: BLE001
        LOGGER.exception("查询最新交易日失败", extra=LOG_EXTRA)
        return None
    if not row:
        return None
    raw_value = row["trade_date"]
    if not raw_value:
        return None
    try:
        return datetime.strptime(str(raw_value), "%Y%m%d").date()
    except ValueError:
        try:
            return datetime.fromisoformat(str(raw_value)).date()
        except ValueError:
            LOGGER.warning("无法解析交易日：%s", raw_value, extra=LOG_EXTRA)
            return None


def _default_backtest_range(window_days: int = 60) -> tuple[date, date]:
    latest = _get_latest_trade_date() or date.today()
    start = latest - timedelta(days=window_days)
    if start > latest:
        start = latest
    return start, latest


def render_today_plan() -> None:
    LOGGER.info("渲染今日计划页面", extra=LOG_EXTRA)
    st.header("今日计划")
    latest_trade_date = _get_latest_trade_date()
    if latest_trade_date:
        st.caption(f"最新交易日：{latest_trade_date.isoformat()}（统计数据请见左侧系统监控）")
    else:
        st.caption("统计与决策概览现已移至左侧'系统监控'侧栏。")
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

    # ADD: read default selection from URL
    q = _get_query_params()
    default_trade_date = q.get("date", [trade_dates[0]])[0]
    try:
        default_idx = trade_dates.index(default_trade_date)
    except ValueError:
        default_idx = 0
    trade_date = st.selectbox("交易日", trade_dates, index=default_idx)

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

    default_ts = q.get("code", [symbols[0]])[0]
    try:
        default_ts_idx = symbols.index(default_ts)
    except ValueError:
        default_ts_idx = 0
    ts_code = st.selectbox("标的", symbols, index=default_ts_idx)
    # ADD: batch selection for re-evaluation
    batch_symbols = st.multiselect("批量重评估（可多选）", symbols, default=[])
    
    # 一键重评估所有标的按钮
    if st.button("一键重评估所有标的", type="primary", width='stretch'):
        with st.spinner("正在对所有标的进行重评估，请稍候..."):
            try:
                # 解析交易日
                trade_date_obj = None
                try:
                    trade_date_obj = date.fromisoformat(str(trade_date))
                except Exception:
                    try:
                        trade_date_obj = datetime.strptime(str(trade_date), "%Y%m%d").date()
                    except Exception:
                        pass
                if trade_date_obj is None:
                    raise ValueError(f"无法解析交易日：{trade_date}")
                
                progress = st.progress(0.0)
                changes_all = []
                success_count = 0
                error_count = 0
                
                # 遍历所有标的
                for idx, code in enumerate(symbols, start=1):
                    try:
                        # 保存重评估前的状态
                        with db_session(read_only=True) as conn:
                            before_rows = conn.execute(
                                "SELECT agent, action FROM agent_utils WHERE trade_date = ? AND ts_code = ?",
                                (trade_date, code),
                            ).fetchall()
                        before_map = {row["agent"]: row["action"] for row in before_rows}
                        
                        # 执行重评估
                        cfg = BtConfig(
                            id="reeval_ui_all",
                            name="UI All Re-eval",
                            start_date=trade_date_obj,
                            end_date=trade_date_obj,
                            universe=[code],
                            params={},
                        )
                        engine = BacktestEngine(cfg)
                        state = PortfolioState()
                        _ = engine.simulate_day(trade_date_obj, state)
                        
                        # 检查变化
                        with db_session(read_only=True) as conn:
                            after_rows = conn.execute(
                                "SELECT agent, action FROM agent_utils WHERE trade_date = ? AND ts_code = ?",
                                (trade_date, code),
                            ).fetchall()
                        for row in after_rows:
                            agent = row["agent"]
                            new_action = row["action"]
                            old_action = before_map.get(agent)
                            if new_action != old_action:
                                changes_all.append({"代码": code, "代理": agent, "原动作": old_action, "新动作": new_action})
                        success_count += 1
                    except Exception as e:
                        LOGGER.exception(f"重评估 {code} 失败", extra=LOG_EXTRA)
                        error_count += 1
                    
                    # 更新进度
                    progress.progress(idx / len(symbols))
                
                # 显示结果
                if error_count > 0:
                    st.error(f"一键重评估完成：成功 {success_count} 个，失败 {error_count} 个")
                else:
                    st.success(f"一键重评估完成：所有 {success_count} 个标的重评估成功")
                
                # 显示变更记录
                if changes_all:
                    st.write("检测到以下动作变更：")
                    st.dataframe(pd.DataFrame(changes_all), hide_index=True, width='stretch')
                
                # 刷新页面数据
                st.rerun()
            except Exception as exc:
                LOGGER.exception("一键重评估失败", extra=LOG_EXTRA)
                st.error(f"一键重评估执行过程中发生错误：{exc}")

    # sync URL params
    _set_query_params(date=str(trade_date), code=str(ts_code))

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
                "department_telemetry": utils.get("_department_telemetry", {}),
            }
            continue

        if agent_name.startswith("dept_"):
            code = agent_name.split("dept_", 1)[-1]
            signals = utils.get("_signals", [])
            risks = utils.get("_risks", [])
            supplements = utils.get("_supplements", [])
            dialogue = utils.get("_dialogue", [])
            telemetry = utils.get("_telemetry", {})
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
                "telemetry": telemetry if isinstance(telemetry, dict) else {},
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
            # ADD: export buttons
            scope = global_info.get("scope_values") or {}
            close_series = global_info.get("close_series") or []
            turnover_series = global_info.get("turnover_series") or []
            st.write("最新字段：")
            if scope:
                st.json(scope)
                st.download_button(
                    "下载字段(JSON)",
                    data=json.dumps(scope, ensure_ascii=False, indent=2),
                    file_name=f"{ts_code}_{trade_date}_scope.json",
                    mime="application/json",
                    key="dl_scope_json",
                )
            if close_series:
                st.write("收盘价时间序列 (最近窗口)：")
                st.json(close_series)
                try:
                    import io, csv
                    buf = io.StringIO()
                    writer = csv.writer(buf)
                    writer.writerow(["trade_date", "close"])
                    for dt, val in close_series:
                        writer.writerow([dt, val])
                    st.download_button(
                        "下载收盘价(CSV)",
                        data=buf.getvalue(),
                        file_name=f"{ts_code}_{trade_date}_close_series.csv",
                        mime="text/csv",
                        key="dl_close_csv",
                    )
                except Exception:
                    pass
            if turnover_series:
                st.write("换手率时间序列 (最近窗口)：")
                st.json(turnover_series)
        dept_sup = global_info.get("department_supplements") or {}
        dept_dialogue = global_info.get("department_dialogue") or {}
        dept_telemetry = global_info.get("department_telemetry") or {}
        if dept_sup or dept_dialogue:
            with st.expander("部门补数与对话记录", expanded=False):
                if dept_sup:
                    st.write("补充数据：")
                    st.json(dept_sup)
                if dept_dialogue:
                    st.write("对话片段：")
                    st.json(dept_dialogue)
        if dept_telemetry:
            with st.expander("部门 LLM 元数据", expanded=False):
                st.json(dept_telemetry)
    else:
        st.info("暂未写入全局策略摘要。")

    st.subheader("部门意见")
    if dept_records:
        # ADD: keyword filter for department summaries
        keyword = st.text_input("筛选摘要/信号关键词", value="")
        filtered = dept_records
        if keyword.strip():
            kw = keyword.strip()
            filtered = [
                item for item in dept_records
                if kw in str(item.get("摘要", "")) or kw in str(item.get("核心信号", ""))
            ]
        # ADD: confidence filter and sort
        min_conf = st.slider("最低信心过滤", 0.0, 1.0, 0.0, 0.05)
        sort_col = st.selectbox("排序列", ["信心", "权重"], index=0)
        filtered = [row for row in filtered if float(row.get("信心", 0.0)) >= min_conf]
        filtered = sorted(filtered, key=lambda r: float(r.get(sort_col, 0.0)), reverse=True)
        dept_df = pd.DataFrame(filtered)
        st.dataframe(dept_df, width='stretch', hide_index=True)
        try:
            st.download_button(
                "下载部门意见(CSV)",
                data=dept_df.to_csv(index=False),
                file_name=f"{trade_date}_{ts_code}_departments.csv",
                mime="text/csv",
                key="dl_dept_csv",
            )
        except Exception:
            pass
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
                telemetry = details.get("telemetry") or {}
                if telemetry:
                    st.write("LLM 元数据：")
                    st.json(telemetry)
    else:
        st.info("暂无部门记录。")

    st.subheader("代理评分")
    if agent_records:
        # ADD: sorting and CSV export for agents
        sort_agent_by = st.selectbox(
            "代理排序",
            ["权重", "SELL", "HOLD", "BUY_S", "BUY_M", "BUY_L"],
            index=1,
        )
        agent_df = pd.DataFrame(agent_records)
        if sort_agent_by in agent_df.columns:
            agent_df = agent_df.sort_values(sort_agent_by, ascending=False)
        st.dataframe(agent_df, width='stretch', hide_index=True)
        try:
            st.download_button(
                "下载代理评分(CSV)",
                data=agent_df.to_csv(index=False),
                file_name=f"{trade_date}_{ts_code}_agents.csv",
                mime="text/csv",
                key="dl_agent_csv",
            )
        except Exception:
            pass
    else:
        st.info("暂无基础代理评分。")

    # 添加相关新闻展示部分
    st.divider()
    st.subheader("相关新闻")
    # 获取与当前标的相关的最新新闻
    try:
        with db_session(read_only=True) as conn:
            # 解析当前trade_date为datetime对象
            try:
                trade_date_obj = date.fromisoformat(str(trade_date))
            except:
                try:
                    trade_date_obj = datetime.strptime(str(trade_date), "%Y%m%d").date()
                except:
                    # 如果解析失败，使用当前日期向前推7天
                    trade_date_obj = date.today() - timedelta(days=7)
            
            # 查询近7天内与当前标的相关的新闻，按发布时间降序排列
            news_query = """
                SELECT id, title, source, pub_time, sentiment, heat, entities
                FROM news
                WHERE ts_code = ? AND pub_time >= ?
                ORDER BY pub_time DESC
                LIMIT 10
            """
            # 计算7天前的日期字符串
            seven_days_ago = (trade_date_obj - timedelta(days=7)).strftime("%Y-%m-%d")
            news_rows = conn.execute(news_query, (ts_code, seven_days_ago)).fetchall()
        
        if news_rows:
            news_data = []
            for row in news_rows:
                # 解析entities字段获取更多信息
                entities_info = {}
                try:
                    if row["entities"]:
                        entities_info = json.loads(row["entities"])
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # 准备新闻数据
                news_item = {
                    "标题": row["title"],
                    "来源": row["source"],
                    "发布时间": row["pub_time"],
                    "情感指数": f"{row['sentiment']:.2f}" if row["sentiment"] is not None else "-",
                    "热度评分": f"{row['heat']:.2f}" if row["heat"] is not None else "-"
                }
                
                # 如果有行业信息，添加到展示数据中
                industries = entities_info.get("industries", [])
                if industries:
                    news_item["相关行业"] = "、".join(industries[:3])  # 只显示前3个行业
                
                news_data.append(news_item)
            
            # 显示新闻表格
            news_df = pd.DataFrame(news_data)
            # 确保所有列都是字符串类型，避免PyArrow序列化错误
            for col in news_df.columns:
                news_df[col] = news_df[col].astype(str)
            st.dataframe(news_df, width='stretch', hide_index=True)
            
            # 添加新闻详情展开视图
            st.write("详细新闻内容：")
            for idx, row in enumerate(news_rows):
                with st.expander(f"{idx+1}. {row['title']}", expanded=False):
                    st.write(f"**来源：** {row['source']}")
                    st.write(f"**发布时间：** {row['pub_time']}")
                    
                    # 解析entities获取更多详细信息
                    entities_info = {}
                    try:
                        if row["entities"]:
                            entities_info = json.loads(row["entities"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                    
                    # 显示情感和热度信息
                    sentiment_display = f"{row['sentiment']:.2f}" if row["sentiment"] is not None else "-"
                    heat_display = f"{row['heat']:.2f}" if row["heat"] is not None else "-"
                    st.write(f"**情感指数：** {sentiment_display} | **热度评分：** {heat_display}")
                    
                    # 显示行业信息
                    industries = entities_info.get("industries", [])
                    if industries:
                        st.write(f"**相关行业：** {'、'.join(industries)}")
                    
                    # 显示重要关键词
                    important_keywords = entities_info.get("important_keywords", [])
                    if important_keywords:
                        st.write(f"**重要关键词：** {'、'.join(important_keywords)}")
                    
                    # 显示URL链接（如果有）
                    url = entities_info.get("source_url", "")
                    if url:
                        st.markdown(f"[查看原文]({url})", unsafe_allow_html=True)
        else:
            st.info(f"近7天内暂无关于 {ts_code} 的新闻。")
    except Exception as e:
        LOGGER.exception("获取新闻数据失败", extra=LOG_EXTRA)
        st.error(f"获取新闻数据时发生错误：{e}")

    st.divider()
    st.subheader("投资池与仓位概览")

    snapshot = get_latest_snapshot()
    if snapshot:
        col_a, col_b, col_c = st.columns(3)
        if snapshot.total_value is not None:
            col_a.metric("组合净值", f"{snapshot.total_value:,.2f}")
        if snapshot.cash is not None:
            col_b.metric("现金余额", f"{snapshot.cash:,.2f}")
        if snapshot.invested_value is not None:
            col_c.metric("持仓市值", f"{snapshot.invested_value:,.2f}")
        detail_cols = st.columns(4)
        if snapshot.unrealized_pnl is not None:
            detail_cols[0].metric("浮盈", f"{snapshot.unrealized_pnl:,.2f}")
        if snapshot.realized_pnl is not None:
            detail_cols[1].metric("已实现盈亏", f"{snapshot.realized_pnl:,.2f}")
        if snapshot.net_flow is not None:
            detail_cols[2].metric("净流入", f"{snapshot.net_flow:,.2f}")
        if snapshot.exposure is not None:
            detail_cols[3].metric("风险敞口", f"{snapshot.exposure:.2%}")
        if snapshot.notes:
            st.caption(f"备注：{snapshot.notes}")
    else:
        st.info("暂无组合快照，请在执行回测或实盘同步后写入 portfolio_snapshots。")

    candidates = list_investment_pool(trade_date=trade_date)
    if candidates:
        candidate_df = pd.DataFrame(
            [
                {
                    "交易日": item.trade_date,
                    "代码": item.ts_code,
                    "评分": item.score,
                    "状态": item.status,
                    "标签": "、".join(item.tags) if item.tags else "-",
                    "理由": item.rationale or "",
                }
                for item in candidates
            ]
        )
        st.write("候选投资池：")
        st.dataframe(candidate_df, width='stretch', hide_index=True)
    else:
        st.caption("候选投资池暂无数据。")

    positions = list_positions(active_only=False)
    if positions:
        position_df = pd.DataFrame(
            [
                {
                    "ID": pos.id,
                    "代码": pos.ts_code,
                    "开仓日": pos.opened_date,
                    "平仓日": pos.closed_date or "-",
                    "状态": pos.status,
                    "数量": pos.quantity,
                    "成本": pos.cost_price,
                    "现价": pos.market_price,
                    "市值": pos.market_value,
                    "浮盈": pos.unrealized_pnl,
                    "已实现": pos.realized_pnl,
                    "目标权重": pos.target_weight,
                }
                for pos in positions
            ]
        )
        st.write("组合持仓：")
        st.dataframe(position_df, width='stretch', hide_index=True)
    else:
        st.caption("组合持仓暂无记录。")

    trades = list_recent_trades(limit=20)
    if trades:
        trades_df = pd.DataFrame(trades)
        st.write("近期成交：")
        st.dataframe(trades_df, width='stretch', hide_index=True)
    else:
        st.caption("近期成交暂无记录。")

    st.caption("数据来源：agent_utils、investment_pool、portfolio_positions、portfolio_trades、portfolio_snapshots。")

    st.divider()
    st.subheader("策略重评估")
    st.caption("对当前选中的交易日与标的，立即触发一次策略评估并回写 agent_utils。")
    cols_re = st.columns([1,1])
    if cols_re[0].button("对该标的重评估", key="reevaluate_current_symbol"):
        with st.spinner("正在重评估..."):
            try:
                trade_date_obj = None
                try:
                    trade_date_obj = date.fromisoformat(str(trade_date))
                except Exception:
                    try:
                        trade_date_obj = datetime.strptime(str(trade_date), "%Y%m%d").date()
                    except Exception:
                        pass
                if trade_date_obj is None:
                    raise ValueError(f"无法解析交易日：{trade_date}")
                # snapshot before
                with db_session(read_only=True) as conn:
                    before_rows = conn.execute(
                        """
                        SELECT agent, action, utils FROM agent_utils
                        WHERE trade_date = ? AND ts_code = ?
                        """,
                        (trade_date, ts_code),
                    ).fetchall()
                before_map = {row["agent"]: (row["action"], row["utils"]) for row in before_rows}
                cfg = BtConfig(
                    id="reeval_ui",
                    name="UI Re-evaluation",
                    start_date=trade_date_obj,
                    end_date=trade_date_obj,
                    universe=[ts_code],
                    params={},
                )
                engine = BacktestEngine(cfg)
                state = PortfolioState()
                _ = engine.simulate_day(trade_date_obj, state)
                # compare after
                with db_session(read_only=True) as conn:
                    after_rows = conn.execute(
                        """
                        SELECT agent, action, utils FROM agent_utils
                        WHERE trade_date = ? AND ts_code = ?
                        """,
                        (trade_date, ts_code),
                    ).fetchall()
                changes = []
                for row in after_rows:
                    agent = row["agent"]
                    new_action = row["action"]
                    old_action, _old_utils = before_map.get(agent, (None, None))
                    if new_action != old_action:
                        changes.append({"代理": agent, "原动作": old_action, "新动作": new_action})
                if changes:
                    st.success("重评估完成，检测到动作变更：")
                    st.dataframe(pd.DataFrame(changes), hide_index=True, width='stretch')
                else:
                    st.success("重评估完成，无动作变更。")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("重评估失败", extra=LOG_EXTRA)
                st.error(f"重评估失败：{exc}")
    if cols_re[1].button("批量重评估（所选）", key="reevaluate_batch", disabled=not batch_symbols):
        with st.spinner("批量重评估执行中..."):
            try:
                trade_date_obj = None
                try:
                    trade_date_obj = date.fromisoformat(str(trade_date))
                except Exception:
                    try:
                        trade_date_obj = datetime.strptime(str(trade_date), "%Y%m%d").date()
                    except Exception:
                        pass
                if trade_date_obj is None:
                    raise ValueError(f"无法解析交易日：{trade_date}")
                progress = st.progress(0.0)
                changes_all: List[Dict[str, object]] = []
                for idx, code in enumerate(batch_symbols, start=1):
                    with db_session(read_only=True) as conn:
                        before_rows = conn.execute(
                            "SELECT agent, action FROM agent_utils WHERE trade_date = ? AND ts_code = ?",
                            (trade_date, code),
                        ).fetchall()
                    before_map = {row["agent"]: row["action"] for row in before_rows}
                    cfg = BtConfig(
                        id="reeval_ui_batch",
                        name="UI Batch Re-eval",
                        start_date=trade_date_obj,
                        end_date=trade_date_obj,
                        universe=[code],
                        params={},
                    )
                    engine = BacktestEngine(cfg)
                    state = PortfolioState()
                    _ = engine.simulate_day(trade_date_obj, state)
                    with db_session(read_only=True) as conn:
                        after_rows = conn.execute(
                            "SELECT agent, action FROM agent_utils WHERE trade_date = ? AND ts_code = ?",
                            (trade_date, code),
                        ).fetchall()
                    for row in after_rows:
                        agent = row["agent"]
                        new_action = row["action"]
                        old_action = before_map.get(agent)
                        if new_action != old_action:
                            changes_all.append({"代码": code, "代理": agent, "原动作": old_action, "新动作": new_action})
                    progress.progress(idx / max(1, len(batch_symbols)))
                st.success("批量重评估完成。")
                if changes_all:
                    st.dataframe(pd.DataFrame(changes_all), hide_index=True, width='stretch')
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("批量重评估失败", extra=LOG_EXTRA)
                st.error(f"批量重评估失败：{exc}")


def render_log_viewer() -> None:
    """渲染日志钻取与历史对比视图页面。"""
    LOGGER.info("渲染日志视图页面", extra=LOG_EXTRA)
    st.header("日志钻取与历史对比")
    st.write("查看系统运行日志，支持时间范围筛选、关键词搜索和历史对比功能。")
    
    # 日志时间范围选择
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=date.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("结束日期", value=date.today())
    
    # 日志级别筛选
    log_levels = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    selected_level = st.selectbox("日志级别", log_levels, index=1)
    
    # 关键词搜索
    search_query = st.text_input("搜索关键词")
    
    # 阶段筛选
    with db_session(read_only=True) as conn:
        stages = [row["stage"] for row in conn.execute("SELECT DISTINCT stage FROM run_log").fetchall()]
    stages = [s for s in stages if s]  # 过滤空值
    stages.insert(0, "ALL")
    selected_stage = st.selectbox("执行阶段", stages)
    
    # 查询日志
    with st.spinner("加载日志数据中..."):
        try:
            with db_session(read_only=True) as conn:
                query_parts = ["SELECT ts, stage, level, msg FROM run_log WHERE 1=1"]
                params = []
                
                # 添加日期过滤
                start_ts = f"{start_date.isoformat()}T00:00:00Z"
                end_ts = f"{end_date.isoformat()}T23:59:59Z"
                query_parts.append("AND ts BETWEEN ? AND ?")
                params.extend([start_ts, end_ts])
                
                # 添加级别过滤
                if selected_level != "ALL":
                    query_parts.append("AND level = ?")
                    params.append(selected_level)
                
                # 添加关键词过滤
                if search_query:
                    query_parts.append("AND msg LIKE ?")
                    params.append(f"%{search_query}%")
                
                # 添加阶段过滤
                if selected_stage != "ALL":
                    query_parts.append("AND stage = ?")
                    params.append(selected_stage)
                
                # 添加排序
                query_parts.append("ORDER BY ts DESC")
                
                # 执行查询
                query = " ".join(query_parts)
                rows = conn.execute(query, params).fetchall()
                
                # 转换为DataFrame
                if rows:
                    # 将sqlite3.Row对象转换为字典列表
                    rows_dict = [{key: row[key] for key in row.keys()} for row in rows]
                    log_df = pd.DataFrame(rows_dict)
                    # 格式化时间戳并确保数据类型一致
                    log_df["ts"] = pd.to_datetime(log_df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    # 确保所有列都是字符串类型，避免PyArrow序列化错误
                    for col in log_df.columns:
                        log_df[col] = log_df[col].astype(str)
                else:
                    log_df = pd.DataFrame(columns=["ts", "stage", "level", "msg"])
                
                # 显示日志表格
                st.dataframe(
                    log_df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "ts": st.column_config.TextColumn("时间"),
                        "stage": st.column_config.TextColumn("执行阶段"),
                        "level": st.column_config.TextColumn("日志级别"),
                        "msg": st.column_config.TextColumn("日志消息", width="large")
                    }
                )
                
                # 下载功能
                if not log_df.empty:
                    csv_data = log_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="下载日志CSV",
                        data=csv_data,
                        file_name=f"logs_{start_date}_{end_date}.csv",
                        mime="text/csv",
                        key="download_logs"
                    )
                    
                    # JSON下载
                    json_data = log_df.to_json(orient='records', force_ascii=False, indent=2)
                    st.download_button(
                        label="下载日志JSON",
                        data=json_data,
                        file_name=f"logs_{start_date}_{end_date}.json",
                        mime="application/json",
                        key="download_logs_json"
                    )
        except Exception as e:
            LOGGER.exception("加载日志失败", extra=LOG_EXTRA)
            st.error(f"加载日志数据失败：{e}")
    
    # 历史对比功能
    st.subheader("历史对比")
    st.write("选择两个时间点的日志进行对比分析。")
    
    # 第一个时间点选择
    col3, col4 = st.columns(2)
    with col3:
        compare_date1 = st.date_input("对比日期1", value=date.today() - timedelta(days=1))
    with col4:
        compare_date2 = st.date_input("对比日期2", value=date.today())
    
    if st.button("执行对比", type="secondary"):
        with st.spinner("执行日志对比分析中..."):
            try:
                with db_session(read_only=True) as conn:
                    # 获取两个日期的日志统计
                    query_date1 = f"{compare_date1.isoformat()}T00:00:00Z"
                    query_date2 = f"{compare_date1.isoformat()}T23:59:59Z"
                    logs1 = conn.execute(
                        "SELECT level, COUNT(*) as count FROM run_log WHERE ts BETWEEN ? AND ? GROUP BY level",
                        (query_date1, query_date2)
                    ).fetchall()
                    
                    query_date3 = f"{compare_date2.isoformat()}T00:00:00Z"
                    query_date4 = f"{compare_date2.isoformat()}T23:59:59Z"
                    logs2 = conn.execute(
                        "SELECT level, COUNT(*) as count FROM run_log WHERE ts BETWEEN ? AND ? GROUP BY level",
                        (query_date3, query_date4)
                    ).fetchall()
                    
                    # 转换为DataFrame并可视化
                    df1 = pd.DataFrame(logs1, columns=["level", "count"])
                    df1["date"] = compare_date1.strftime("%Y-%m-%d")
                    df2 = pd.DataFrame(logs2, columns=["level", "count"])
                    df2["date"] = compare_date2.strftime("%Y-%m-%d")
                    
                    # 确保所有列的数据类型一致，避免PyArrow序列化错误
                    for df in [df1, df2]:
                        for col in df.columns:
                            if col != "level":  # level列保持字符串类型
                                df[col] = df[col].astype(object)
                    
                    compare_df = pd.concat([df1, df2])
                    
                    # 绘制对比图表
                    fig = px.bar(
                        compare_df, 
                        x="level", 
                        y="count", 
                        color="date",
                        barmode="group",
                        title=f"日志级别分布对比 ({compare_date1} vs {compare_date2})"
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    # 显示详细对比表格
                    st.write("日志统计对比：")
                    # 使用不含连字符的日期格式作为列名后缀，避免Arrow类型转换错误
                    date1_str = compare_date1.strftime("%Y%m%d")
                    date2_str = compare_date2.strftime("%Y%m%d")
                    merged_df = df1.merge(df2, on="level", suffixes=(f"_{date1_str}", f"_{date2_str}"), how="outer").fillna(0)
                    st.dataframe(merged_df, hide_index=True, width="stretch")
            except Exception as e:
                LOGGER.exception("日志对比失败", extra=LOG_EXTRA)
                st.error(f"日志对比分析失败：{e}")

    cfg = get_config()
    default_start, default_end = _default_backtest_range(window_days=60)
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

    with st.expander("离线调参实验 (DecisionEnv)", expanded=False):
        st.caption(
            "使用 DecisionEnv 对代理权重做离线调参。请选择需要优化的代理并设定权重范围，"
            "系统会运行一次回测并返回收益、回撤等指标。若 LLM 网络不可用，将返回失败标记。"
        )

        disable_departments = st.checkbox(
            "禁用部门 LLM（仅规则代理，适合离线快速评估）",
            value=True,
            help="关闭部门调用后不依赖外部 LLM 网络，仅根据规则代理权重模拟。",
        )

        default_experiment_id = f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_id = st.text_input(
            "实验 ID",
            value=default_experiment_id,
            help="用于在 tuning_results 表中区分不同实验。",
        )
        strategy_label = st.text_input(
            "策略说明",
            value="DecisionEnv",
            help="可选：为本次调参记录一个策略名称或备注。",
        )

        agent_objects = default_agents()
        agent_names = [agent.name for agent in agent_objects]
        if not agent_names:
            st.info("暂无可调整的代理。")
        else:
            selected_agents = st.multiselect(
                "选择调参的代理权重",
                agent_names,
                default=agent_names[:2],
                key="decision_env_agents",
            )

            specs: List[ParameterSpec] = []
            action_values: List[float] = []
            range_valid = True
            for idx, agent_name in enumerate(selected_agents):
                col_min, col_max, col_action = st.columns([1, 1, 2])
                min_key = f"decision_env_min_{agent_name}"
                max_key = f"decision_env_max_{agent_name}"
                action_key = f"decision_env_action_{agent_name}"
                default_min = 0.0
                default_max = 1.0
                min_val = col_min.number_input(
                    f"{agent_name} 最小权重",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_min,
                    step=0.05,
                    key=min_key,
                )
                max_val = col_max.number_input(
                    f"{agent_name} 最大权重",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_max,
                    step=0.05,
                    key=max_key,
                )
                if max_val <= min_val:
                    range_valid = False
                action_val = col_action.slider(
                    f"{agent_name} 动作 (0-1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    key=action_key,
                )
                specs.append(
                    ParameterSpec(
                        name=f"weight_{agent_name}",
                        target=f"agent_weights.{agent_name}",
                        minimum=min_val,
                        maximum=max_val,
                    )
                )
                action_values.append(action_val)

            run_decision_env = st.button("执行单次调参", key="run_decision_env_button")
            just_finished_single = False
            if run_decision_env:
                if not selected_agents:
                    st.warning("请至少选择一个代理进行调参。")
                elif not range_valid:
                    st.error("请确保所有代理的最大权重大于最小权重。")
                else:
                    LOGGER.info(
                        "离线调参（单次）按钮点击，已选择代理=%s 动作=%s disable_departments=%s",
                        selected_agents,
                        action_values,
                        disable_departments,
                        extra=LOG_EXTRA,
                    )
                    baseline_weights = cfg.agent_weights.as_dict()
                    for agent in agent_objects:
                        baseline_weights.setdefault(agent.name, 1.0)

                    universe_env = [code.strip() for code in universe_text.split(',') if code.strip()]
                    if not universe_env:
                        st.error("请先指定至少一个股票代码。")
                    else:
                        bt_cfg_env = BtConfig(
                            id="decision_env_streamlit",
                            name="DecisionEnv Streamlit",
                            start_date=start_date,
                            end_date=end_date,
                            universe=universe_env,
                            params={
                                "target": target,
                                "stop": stop,
                                "hold_days": int(hold_days),
                            },
                            method=cfg.decision_method,
                        )
                        env = DecisionEnv(
                            bt_config=bt_cfg_env,
                            parameter_specs=specs,
                            baseline_weights=baseline_weights,
                            disable_departments=disable_departments,
                        )
                        env.reset()
                        LOGGER.debug(
                            "离线调参（单次）启动 DecisionEnv：cfg=%s 参数维度=%s",
                            bt_cfg_env,
                            len(specs),
                            extra=LOG_EXTRA,
                        )
                        with st.spinner("正在执行离线调参……"):
                            try:
                                observation, reward, done, info = env.step(action_values)
                                LOGGER.info(
                                    "离线调参（单次）完成，obs=%s reward=%.4f done=%s",
                                    observation,
                                    reward,
                                    done,
                                    extra=LOG_EXTRA,
                                )
                            except Exception as exc:  # noqa: BLE001
                                LOGGER.exception("DecisionEnv 调用失败", extra=LOG_EXTRA)
                                st.error(f"离线调参失败：{exc}")
                                st.session_state.pop(_DECISION_ENV_SINGLE_RESULT_KEY, None)
                            else:
                                if observation.get("failure"):
                                    st.error("调参失败：回测执行未完成，可能是 LLM 网络不可用或参数异常。")
                                    st.json(observation)
                                    st.session_state.pop(_DECISION_ENV_SINGLE_RESULT_KEY, None)
                                else:
                                    resolved_experiment_id = experiment_id or str(uuid.uuid4())
                                    resolved_strategy = strategy_label or "DecisionEnv"
                                    action_payload = {
                                        name: value
                                        for name, value in zip(selected_agents, action_values)
                                    }
                                    metrics_payload = dict(observation)
                                    metrics_payload["reward"] = reward
                                    log_success = False
                                    try:
                                        log_tuning_result(
                                            experiment_id=resolved_experiment_id,
                                            strategy=resolved_strategy,
                                            action=action_payload,
                                            reward=reward,
                                            metrics=metrics_payload,
                                            weights=info.get("weights", {}),
                                        )
                                    except Exception:  # noqa: BLE001
                                        LOGGER.exception("记录调参结果失败", extra=LOG_EXTRA)
                                    else:
                                        log_success = True
                                        LOGGER.info(
                                            "离线调参（单次）日志写入成功：experiment=%s strategy=%s",
                                            resolved_experiment_id,
                                            resolved_strategy,
                                            extra=LOG_EXTRA,
                                        )
                                    st.session_state[_DECISION_ENV_SINGLE_RESULT_KEY] = {
                                        "observation": dict(observation),
                                        "reward": float(reward),
                                        "weights": info.get("weights", {}),
                                        "nav_series": info.get("nav_series"),
                                        "trades": info.get("trades"),
                                        "portfolio_snapshots": info.get("portfolio_snapshots"),
                                        "portfolio_trades": info.get("portfolio_trades"),
                                        "risk_breakdown": info.get("risk_breakdown"),
                                        "selected_agents": list(selected_agents),
                                        "action_values": list(action_values),
                                        "experiment_id": resolved_experiment_id,
                                        "strategy_label": resolved_strategy,
                                        "logged": log_success,
                                    }
                                    just_finished_single = True
            single_result = st.session_state.get(_DECISION_ENV_SINGLE_RESULT_KEY)
            if single_result:
                if just_finished_single:
                    st.success("离线调参完成")
                else:
                    st.success("离线调参结果（最近一次运行）")
                st.caption(
                    f"实验 ID：{single_result.get('experiment_id', '-') } | 策略：{single_result.get('strategy_label', 'DecisionEnv')}"
                )
                observation = single_result.get("observation", {})
                reward = float(single_result.get("reward", 0.0))
                col_metrics = st.columns(4)
                col_metrics[0].metric("总收益", f"{observation.get('total_return', 0.0):+.2%}")
                col_metrics[1].metric("最大回撤", f"{observation.get('max_drawdown', 0.0):+.2%}")
                col_metrics[2].metric("波动率", f"{observation.get('volatility', 0.0):+.2%}")
                col_metrics[3].metric("奖励", f"{reward:+.4f}")

                turnover_ratio = float(observation.get("turnover", 0.0) or 0.0)
                turnover_value = float(observation.get("turnover_value", 0.0) or 0.0)
                risk_count = float(observation.get("risk_count", 0.0) or 0.0)
                col_metrics_extra = st.columns(3)
                col_metrics_extra[0].metric("平均换手率", f"{turnover_ratio:.2%}")
                col_metrics_extra[1].metric("成交额", f"{turnover_value:,.0f}")
                col_metrics_extra[2].metric("风险事件数", f"{int(risk_count)}")

                weights_dict = single_result.get("weights") or {}
                if weights_dict:
                    st.write("调参后权重：")
                    st.json(weights_dict)
                    if st.button("保存这些权重为默认配置", key="save_decision_env_weights_single"):
                        try:
                            cfg.agent_weights.update_from_dict(weights_dict)
                            save_config(cfg)
                        except Exception as exc:  # noqa: BLE001
                            LOGGER.exception("保存权重失败", extra={**LOG_EXTRA, "error": str(exc)})
                            st.error(f"写入配置失败：{exc}")
                        else:
                            st.success("代理权重已写入 config.json")

                if single_result.get("logged"):
                    st.caption("调参结果已写入 tuning_results 表。")

                nav_series = single_result.get("nav_series") or []
                if nav_series:
                    try:
                        nav_df = pd.DataFrame(nav_series)
                        if {"trade_date", "nav"}.issubset(nav_df.columns):
                            nav_df = nav_df.sort_values("trade_date")
                            nav_df["trade_date"] = pd.to_datetime(nav_df["trade_date"])
                            st.line_chart(nav_df.set_index("trade_date")["nav"], height=220)
                    except Exception:  # noqa: BLE001
                        LOGGER.debug("导航曲线绘制失败", extra=LOG_EXTRA)

                trades = single_result.get("trades") or []
                if trades:
                    st.write("成交记录：")
                    st.dataframe(pd.DataFrame(trades), hide_index=True, width='stretch')

                snapshots = single_result.get("portfolio_snapshots") or []
                if snapshots:
                    with st.expander("投资组合快照", expanded=False):
                        st.dataframe(pd.DataFrame(snapshots), hide_index=True, width='stretch')

                portfolio_trades = single_result.get("portfolio_trades") or []
                if portfolio_trades:
                    with st.expander("组合成交明细", expanded=False):
                        st.dataframe(pd.DataFrame(portfolio_trades), hide_index=True, width='stretch')

                risk_breakdown = single_result.get("risk_breakdown") or {}
                if risk_breakdown:
                    with st.expander("风险事件统计", expanded=False):
                        st.json(risk_breakdown)

                if st.button("清除单次调参结果", key="clear_decision_env_single"):
                    st.session_state.pop(_DECISION_ENV_SINGLE_RESULT_KEY, None)
                    st.success("已清除单次调参结果缓存。")

            st.divider()
            st.caption("批量调参：在下方输入多组动作，每行表示一组 0-1 之间的值，用逗号分隔。")
            default_grid = "\n".join(
                [
                    ",".join(["0.2" for _ in specs]),
                    ",".join(["0.5" for _ in specs]),
                    ",".join(["0.8" for _ in specs]),
                ]
            ) if specs else ""
            action_grid_raw = st.text_area(
                "动作列表",
                value=default_grid,
                height=120,
                key="decision_env_batch_actions",
            )
            run_batch = st.button("批量执行调参", key="run_decision_env_batch")
            batch_just_ran = False
            if run_batch:
                if not selected_agents:
                    st.warning("请先选择调参代理。")
                elif not range_valid:
                    st.error("请确保所有代理的最大权重大于最小权重。")
                else:
                    LOGGER.info(
                        "离线调参（批量）按钮点击，已选择代理=%s disable_departments=%s",
                        selected_agents,
                        disable_departments,
                        extra=LOG_EXTRA,
                    )
                    lines = [line.strip() for line in action_grid_raw.splitlines() if line.strip()]
                    if not lines:
                        st.warning("请在文本框中输入至少一组动作。")
                    else:
                        LOGGER.debug(
                            "离线调参（批量）原始输入=%s",
                            lines,
                            extra=LOG_EXTRA,
                        )
                        parsed_actions: List[List[float]] = []
                        for line in lines:
                            try:
                                values = [float(val.strip()) for val in line.split(',') if val.strip()]
                            except ValueError:
                                st.error(f"无法解析动作行：{line}")
                                parsed_actions = []
                                break
                            if len(values) != len(specs):
                                st.error(f"动作维度不匹配（期望 {len(specs)} 个值）：{line}")
                                parsed_actions = []
                                break
                            parsed_actions.append(values)
                        if parsed_actions:
                            LOGGER.info(
                                "离线调参（批量）解析动作成功，数量=%s",
                                len(parsed_actions),
                                extra=LOG_EXTRA,
                            )
                            baseline_weights = cfg.agent_weights.as_dict()
                            for agent in agent_objects:
                                baseline_weights.setdefault(agent.name, 1.0)

                            universe_env = [code.strip() for code in universe_text.split(',') if code.strip()]
                            if not universe_env:
                                st.error("请先指定至少一个股票代码。")
                            else:
                                bt_cfg_env = BtConfig(
                                    id="decision_env_streamlit_batch",
                                    name="DecisionEnv Batch",
                                    start_date=start_date,
                                    end_date=end_date,
                                    universe=universe_env,
                                    params={
                                        "target": target,
                                        "stop": stop,
                                        "hold_days": int(hold_days),
                                    },
                                    method=cfg.decision_method,
                                )
                                env = DecisionEnv(
                                    bt_config=bt_cfg_env,
                                    parameter_specs=specs,
                                    baseline_weights=baseline_weights,
                                    disable_departments=disable_departments,
                                )
                                results: List[Dict[str, object]] = []
                                resolved_experiment_id = experiment_id or str(uuid.uuid4())
                                resolved_strategy = strategy_label or "DecisionEnv"
                                LOGGER.debug(
                                    "离线调参（批量）启动 DecisionEnv：cfg=%s 动作组=%s",
                                    bt_cfg_env,
                                    len(parsed_actions),
                                    extra=LOG_EXTRA,
                                )
                                with st.spinner("正在批量执行调参……"):
                                    for idx, action_vals in enumerate(parsed_actions, start=1):
                                        env.reset()
                                        try:
                                            observation, reward, done, info = env.step(action_vals)
                                        except Exception as exc:  # noqa: BLE001
                                            LOGGER.exception("批量调参失败", extra=LOG_EXTRA)
                                            results.append(
                                                {
                                                    "序号": idx,
                                                    "动作": action_vals,
                                                    "状态": "error",
                                                    "错误": str(exc),
                                                }
                                            )
                                            continue
                                        if observation.get("failure"):
                                            results.append(
                                                {
                                                    "序号": idx,
                                                    "动作": action_vals,
                                                    "状态": "failure",
                                                    "奖励": -1.0,
                                                }
                                            )
                                        else:
                                            LOGGER.info(
                                                "离线调参（批量）第 %s 组完成，reward=%.4f obs=%s",
                                                idx,
                                                reward,
                                                observation,
                                                extra=LOG_EXTRA,
                                            )
                                            action_payload = {
                                                name: value
                                                for name, value in zip(selected_agents, action_vals)
                                            }
                                            metrics_payload = dict(observation)
                                            metrics_payload["reward"] = reward
                                            weights_payload = info.get("weights", {})
                                            try:
                                                log_tuning_result(
                                                    experiment_id=resolved_experiment_id,
                                                    strategy=resolved_strategy,
                                                    action=action_payload,
                                                    reward=reward,
                                                    metrics=metrics_payload,
                                                    weights=weights_payload,
                                                )
                                            except Exception:  # noqa: BLE001
                                                LOGGER.exception("记录调参结果失败", extra=LOG_EXTRA)
                                            results.append(
                                                {
                                                    "序号": idx,
                                                    "动作": action_vals,
                                                    "状态": "ok",
                                                    "总收益": observation.get("total_return", 0.0),
                                                    "最大回撤": observation.get("max_drawdown", 0.0),
                                                    "波动率": observation.get("volatility", 0.0),
                                                    "奖励": reward,
                                                    "权重": weights_payload,
                                                }
                                            )
                                st.session_state[_DECISION_ENV_BATCH_RESULTS_KEY] = {
                                    "results": results,
                                    "selectable": [
                                        row
                                        for row in results
                                        if row.get("状态") == "ok" and row.get("权重")
                                    ],
                                    "experiment_id": resolved_experiment_id,
                                    "strategy_label": resolved_strategy,
                                }
                                batch_just_ran = True
                                LOGGER.info(
                                    "离线调参（批量）执行结束，总结果条数=%s",
                                    len(results),
                                    extra=LOG_EXTRA,
                                )
            batch_state = st.session_state.get(_DECISION_ENV_BATCH_RESULTS_KEY)
            if batch_state:
                results = batch_state.get("results") or []
                if results:
                    if batch_just_ran:
                        st.success("批量调参完成")
                    else:
                        st.success("批量调参结果（最近一次运行）")
                    st.caption(
                        f"实验 ID：{batch_state.get('experiment_id', '-') } | 策略：{batch_state.get('strategy_label', 'DecisionEnv')}"
                    )
                    results_df = pd.DataFrame(results)
                    st.write("批量调参结果：")
                    st.dataframe(results_df, hide_index=True, width='stretch')
                    selectable = batch_state.get("selectable") or []
                    if selectable:
                        option_labels = [
                            f"序号 {row['序号']} | 奖励 {row.get('奖励', 0.0):+.4f}"
                            for row in selectable
                        ]
                        selected_label = st.selectbox(
                            "选择要保存的记录",
                            option_labels,
                            key="decision_env_batch_select",
                        )
                        selected_row = None
                        for label, row in zip(option_labels, selectable):
                            if label == selected_label:
                                selected_row = row
                                break
                        if selected_row and st.button(
                            "保存所选权重为默认配置",
                            key="save_decision_env_weights_batch",
                        ):
                            try:
                                cfg.agent_weights.update_from_dict(selected_row.get("权重", {}))
                                save_config(cfg)
                            except Exception as exc:  # noqa: BLE001
                                LOGGER.exception("批量保存权重失败", extra={**LOG_EXTRA, "error": str(exc)})
                                st.error(f"写入配置失败：{exc}")
                            else:
                                st.success(
                                    f"已将序号 {selected_row['序号']} 的权重写入 config.json"
                                )
                    else:
                        st.caption("暂无成功的结果可供保存。")
                else:
                    st.caption("批量调参在最近一次执行中未产生结果。")
                if st.button("清除批量调参结果", key="clear_decision_env_batch"):
                    st.session_state.pop(_DECISION_ENV_BATCH_RESULTS_KEY, None)
                    st.session_state.pop("decision_env_batch_select", None)
                    st.success("已清除批量调参结果缓存。")

    # ADD: Comparison view for multiple backtest configurations
    with st.expander("回测结果对比", expanded=False):
        st.caption("从历史回测配置中选择多个进行净值曲线与指标对比。")
        normalize_to_one = st.checkbox("归一化到 1 起点", value=True)
        use_log_y = st.checkbox("对数坐标", value=False)
        metric_options = ["总收益", "最大回撤", "交易数", "平均换手", "风险事件"]
        selected_metrics = st.multiselect("显示指标列", metric_options, default=metric_options)
        try:
            with db_session(read_only=True) as conn:
                cfg_rows = conn.execute(
                    "SELECT id, name FROM bt_config ORDER BY rowid DESC LIMIT 50"
                ).fetchall()
        except Exception:  # noqa: BLE001
            LOGGER.exception("读取 bt_config 失败", extra=LOG_EXTRA)
            cfg_rows = []
        cfg_options = [f"{row['id']} | {row['name']}" for row in cfg_rows]
        selected_labels = st.multiselect("选择配置", cfg_options, default=cfg_options[:2])
        selected_ids = [label.split(" | ")[0].strip() for label in selected_labels]
        nav_df = pd.DataFrame()
        rpt_df = pd.DataFrame()
        if selected_ids:
            try:
                with db_session(read_only=True) as conn:
                    nav_df = pd.read_sql_query(
                        "SELECT cfg_id, trade_date, nav FROM bt_nav WHERE cfg_id IN (%s)" % (",".join(["?"]*len(selected_ids))),
                        conn,
                        params=tuple(selected_ids),
                    )
                    rpt_df = pd.read_sql_query(
                        "SELECT cfg_id, summary FROM bt_report WHERE cfg_id IN (%s)" % (",".join(["?"]*len(selected_ids))),
                        conn,
                        params=tuple(selected_ids),
                    )
            except Exception:  # noqa: BLE001
                LOGGER.exception("读取回测结果失败", extra=LOG_EXTRA)
                st.error("读取回测结果失败")
                nav_df = pd.DataFrame()
                rpt_df = pd.DataFrame()
            if not nav_df.empty:
                try:
                    nav_df["trade_date"] = pd.to_datetime(nav_df["trade_date"], errors="coerce")
                    # ADD: date window filter
                    overall_min = pd.to_datetime(nav_df["trade_date"].min()).date()
                    overall_max = pd.to_datetime(nav_df["trade_date"].max()).date()
                    col_d1, col_d2 = st.columns(2)
                    start_filter = col_d1.date_input("起始日期", value=overall_min)
                    end_filter = col_d2.date_input("结束日期", value=overall_max)
                    if start_filter > end_filter:
                        start_filter, end_filter = end_filter, start_filter
                    mask = (nav_df["trade_date"].dt.date >= start_filter) & (nav_df["trade_date"].dt.date <= end_filter)
                    nav_df = nav_df.loc[mask]
                    pivot = nav_df.pivot_table(index="trade_date", columns="cfg_id", values="nav")
                    if normalize_to_one:
                        pivot = pivot.apply(lambda s: s / s.dropna().iloc[0] if s.dropna().size else s)
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    for col in pivot.columns:
                        fig.add_trace(go.Scatter(x=pivot.index, y=pivot[col], mode="lines", name=str(col)))
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
                    if use_log_y:
                        fig.update_yaxes(type="log")
                    st.plotly_chart(fig, width='stretch')
                    # ADD: export pivot
                    try:
                        csv_buf = pivot.reset_index()
                        csv_buf.columns = ["trade_date"] + [str(c) for c in pivot.columns]
                        st.download_button(
                            "下载曲线(CSV)",
                            data=csv_buf.to_csv(index=False),
                            file_name="bt_nav_compare.csv",
                            mime="text/csv",
                            key="dl_nav_compare",
                        )
                    except Exception:
                        pass
                except Exception:  # noqa: BLE001
                    LOGGER.debug("绘制对比曲线失败", extra=LOG_EXTRA)
            if not rpt_df.empty:
                try:
                    metrics_rows: List[Dict[str, object]] = []
                    for _, row in rpt_df.iterrows():
                        cfg_id = row["cfg_id"]
                        try:
                            summary = json.loads(row["summary"]) if isinstance(row["summary"], str) else (row["summary"] or {})
                        except json.JSONDecodeError:
                            summary = {}
                        record = {
                            "cfg_id": cfg_id,
                            "总收益": summary.get("total_return"),
                            "最大回撤": summary.get("max_drawdown"),
                            "交易数": summary.get("trade_count"),
                            "平均换手": summary.get("avg_turnover"),
                            "风险事件": summary.get("risk_events"),
                        }
                        metrics_rows.append({k: v for k, v in record.items() if (k == "cfg_id" or k in selected_metrics)})
                    if metrics_rows:
                        dfm = pd.DataFrame(metrics_rows)
                        st.dataframe(dfm, hide_index=True, width='stretch')
                        try:
                            st.download_button(
                                "下载指标(CSV)",
                                data=dfm.to_csv(index=False),
                                file_name="bt_metrics_compare.csv",
                                mime="text/csv",
                                key="dl_metrics_compare",
                            )
                        except Exception:
                            pass
                except Exception:  # noqa: BLE001
                    LOGGER.debug("渲染指标表失败", extra=LOG_EXTRA)
        else:
            st.info("请选择至少一个配置进行对比。")


def render_settings() -> None:
    LOGGER.info("渲染设置页面", extra=LOG_EXTRA)
    st.header("数据与设置")
    cfg = get_config()
    LOGGER.debug("当前 TuShare Token 是否已配置=%s", bool(cfg.tushare_token), extra=LOG_EXTRA)
    
    # 基础配置
    col1, col2 = st.columns([2, 1])
    with col1:
        token = st.text_input("TuShare Token", value=cfg.tushare_token or "", type="password")
    with col2:
        auto_update = st.checkbox(
            "自动更新数据", 
            value=cfg.auto_update_data, 
            help="勾选后，每次启动程序将自动执行Tushare和RSS数据拉取"
        )

    if st.button("保存设置"):
        LOGGER.info("保存设置按钮被点击", extra=LOG_EXTRA)
        cfg.tushare_token = token.strip() or None
        cfg.auto_update_data = auto_update
        LOGGER.info("TuShare Token 更新，是否为空=%s", cfg.tushare_token is None, extra=LOG_EXTRA)
        LOGGER.info("自动更新数据设置=%s", cfg.auto_update_data, extra=LOG_EXTRA)
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
        mode_key = f"provider_mode_{selected_provider}"
        enabled_key = f"provider_enabled_{selected_provider}"

        title_val = st.text_input("备注名称", value=provider_cfg.title or "", key=title_key)
        base_val = st.text_input("Base URL", value=provider_cfg.base_url or "", key=base_key, help="调用地址，例如：https://api.openai.com")
        api_val = st.text_input("API Key", value=provider_cfg.api_key or "", key=api_key_key, type="password")
        
        enabled_val = st.checkbox("启用", value=provider_cfg.enabled, key=enabled_key)
        mode_val = st.selectbox("模式", options=["openai", "ollama"], index=0 if provider_cfg.mode == "openai" else 1, key=mode_key)
        st.markdown("可用模型：")
        if provider_cfg.models:
            st.code("\n".join(provider_cfg.models), language="text")
        else:
            st.info("尚未获取模型列表，可点击下方按钮自动拉取。")
        # ADD: show cache last updated if available
        try:
            cache_key = f"{selected_provider}|{(base_val or '').strip()}"
            entry = _MODEL_CACHE.get(cache_key)
            if entry and isinstance(entry.get("ts"), float):
                ts = datetime.fromtimestamp(entry["ts"]).strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"最近拉取时间：{ts}")
        except Exception:
            pass

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
            provider_cfg.enabled = enabled_val
            provider_cfg.mode = mode_val
            providers[selected_provider] = provider_cfg
            cfg.llm_providers = providers
            cfg.sync_runtime_llm()
            save_config()
            st.success("Provider 已保存。")
            st.session_state[title_key] = provider_cfg.title or ""

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
                alerts.add_warning("TuShare", "示例拉取失败", str(exc))
                _update_dashboard_sidebar()

    st.info("注意：TuShare 拉取依赖网络与 Token，若环境未配置将出现错误提示。")

    st.divider()

    st.subheader("RSS 数据测试")
    st.write("用于验证 RSS 配置是否能够正常抓取新闻并写入数据库。")
    rss_url = st.text_input(
        "测试 RSS 地址",
        value="https://rsshub.app/cls/depth/1000",
        help="留空则使用默认配置的全部 RSS 来源。",
    ).strip()
    rss_hours = int(
        st.number_input(
            "回溯窗口（小时）",
            min_value=1,
            max_value=168,
            value=24,
            step=6,
            help="仅抓取最近指定小时内的新闻。",
        )
    )
    rss_limit = int(
        st.number_input(
            "单源抓取条数",
            min_value=1,
            max_value=200,
            value=50,
            step=10,
        )
    )
    if st.button("运行 RSS 测试"):
        from app.ingest import rss as rss_ingest

        LOGGER.info(
            "点击 RSS 测试按钮 rss_url=%s hours=%s limit=%s",
            rss_url,
            rss_hours,
            rss_limit,
            extra=LOG_EXTRA,
        )
        with st.spinner("正在抓取 RSS 新闻..."):
            try:
                if rss_url:
                    items = rss_ingest.fetch_rss_feed(
                        rss_url,
                        hours_back=rss_hours,
                        max_items=rss_limit,
                    )
                    count = rss_ingest.save_news_items(items)
                else:
                    count = rss_ingest.ingest_configured_rss(
                        hours_back=rss_hours,
                        max_items_per_feed=rss_limit,
                    )
                st.success(f"RSS 测试完成，新增 {count} 条新闻记录。")
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("RSS 测试失败", extra=LOG_EXTRA)
                st.error(f"RSS 测试失败：{exc}")
                alerts.add_warning("RSS", "RSS 测试执行失败", str(exc))
                _update_dashboard_sidebar()

    st.divider()
    days = int(
        st.number_input(
            "检查窗口（天数）",
            min_value=30,
            max_value=10950,
            value=365,
            step=30,
        )
    )
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

    if st.button("执行手动数据同步"):
        LOGGER.info("点击执行手动数据同步按钮", extra=LOG_EXTRA)
        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()
        log_placeholder = st.empty()
        messages: list[str] = []

        def hook(message: str, value: float) -> None:
            progress_bar.progress(min(max(value, 0.0), 1.0))
            status_placeholder.write(message)
            messages.append(message)
            LOGGER.debug("手动数据同步进度：%s -> %.2f", message, value, extra=LOG_EXTRA)

        with st.spinner("正在执行手动数据同步..."):
            try:
                report = run_boot_check(
                    days=days,
                    progress_hook=hook,
                    force_refresh=force_refresh,
                )
                LOGGER.info("手动数据同步成功", extra=LOG_EXTRA)
                st.success("手动数据同步完成，以下为数据覆盖摘要。")
                st.json(report.to_dict())
                if messages:
                    log_placeholder.markdown("\n".join(f"- {msg}" for msg in messages))
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("手动数据同步失败", extra=LOG_EXTRA)
                st.error(f"手动数据同步失败：{exc}")
                alerts.add_warning("数据同步", "手动数据同步失败", str(exc))
                _update_dashboard_sidebar()
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
    
    # 确保所有列的数据类型正确，避免PyArrow序列化错误
    numeric_columns = ["开盘价", "最高价", "最低价", "收盘价", "成交量(手)", "成交额(千元)"]
    for col in numeric_columns:
        if col in df_reset.columns:
            df_reset[col] = pd.to_numeric(df_reset[col], errors='coerce')
    
    # 确保日期列是datetime类型
    df_reset["交易日"] = pd.to_datetime(df_reset["交易日"])

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
    # 确保收盘价列是数值类型
    df_reset["收盘价"] = pd.to_numeric(df_reset["收盘价"], errors='coerce')
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


def render_data_settings() -> None:
    """渲染数据源配置界面."""
    st.subheader("Tushare 数据源")
    cfg = get_config()
    
    col1, col2 = st.columns(2)
    with col1:
        tushare_token = st.text_input(
            "Tushare Token",
            value=cfg.tushare_token or "",
            type="password",
            help="从 tushare.pro 获取的 API token"
        )
        
    with col2:
        auto_update = st.checkbox(
            "启动时自动更新数据",
            value=cfg.auto_update_data,
            help="启动应用时自动检查并更新数据"
        )
        
    update_interval = st.slider(
        "数据更新间隔(天)",
        min_value=1,
        max_value=30,
        value=cfg.data_update_interval,
        help="自动更新时检查的数据时间范围"
    )
    
    if st.button("保存数据源配置"):
        cfg.tushare_token = tushare_token
        cfg.auto_update_data = auto_update
        cfg.data_update_interval = update_interval
        save_config(cfg)
        st.success("数据源配置已更新！")
        
    st.divider()
    st.subheader("数据更新记录")
    
    with db_session() as session:
        df = pd.read_sql_query(
            """
            SELECT job_type, status, created_at, updated_at, error_msg
            FROM fetch_jobs 
            ORDER BY created_at DESC 
            LIMIT 50
            """,
            session
        )
        
    if not df.empty:
        df["duration"] = (df["updated_at"] - df["created_at"]).dt.total_seconds().round(2)
        df = df.drop(columns=["updated_at"])
        df = df.rename(columns={
            "job_type": "数据类型",
            "status": "状态",
            "created_at": "开始时间",
            "error_msg": "错误信息",
            "duration": "耗时(秒)"
        })
        st.dataframe(df, width='stretch')
    else:
        st.info("暂无数据更新记录")


def main() -> None:
    LOGGER.info("初始化 Streamlit UI", extra=LOG_EXTRA)
    st.set_page_config(page_title="多智能体个人投资助理", layout="wide")
    
    # 确保数据库表已创建
    from app.data.schema import initialize_database
    initialize_database()
    
    # 检查是否需要自动更新数据
    cfg = get_config()
    if cfg.auto_update_data:
        LOGGER.info("检测到自动更新数据选项已启用，开始执行数据拉取", extra=LOG_EXTRA)
        try:
            # 初始化数据库
            from app.data.schema import initialize_database
            initialize_database()
            
            # 执行开机检查（包含数据拉取）
            from app.ingest.checker import run_boot_check
            with st.spinner("正在自动更新数据..."):
                def progress_hook(message: str, progress: float) -> None:
                    st.write(f"📊 {message} ({progress:.1%})")
                
                report = run_boot_check(
                    days=30,  # 最近30天
                    auto_fetch=True,
                    progress_hook=progress_hook,
                    force_refresh=False
                )
                
                # 执行RSS新闻拉取
                from app.ingest.rss import ingest_configured_rss
                rss_count = ingest_configured_rss(hours_back=24, max_items_per_feed=50)
                
                LOGGER.info("自动数据更新完成：日线数据覆盖%s-%s，RSS新闻%s条", 
                           report.start, report.end, rss_count, extra=LOG_EXTRA)
                st.success(f"✅ 自动数据更新完成：获取RSS新闻 {rss_count} 条")
                
        except Exception as exc:
            LOGGER.exception("自动数据更新失败", extra=LOG_EXTRA)
            st.error(f"❌ 自动数据更新失败：{exc}")
    
    render_global_dashboard()
    tabs = st.tabs(["今日计划", "回测与复盘", "数据与设置", "自检测试"])
    LOGGER.debug("Tabs 初始化完成：%s", ["今日计划", "回测与复盘", "数据与设置", "自检测试"], extra=LOG_EXTRA)
    with tabs[0]:
        render_today_plan()
    with tabs[1]:
        render_log_viewer()
    with tabs[2]:
        st.header("系统设置")
        settings_tabs = st.tabs(["基本配置", "投资组合", "数据源"])
        
        with settings_tabs[0]:
            render_settings()
            
        with settings_tabs[1]:
            from app.ui.portfolio_config import render_portfolio_config
            render_portfolio_config()
            
        with settings_tabs[2]:
            render_data_settings()
    with tabs[3]:
        render_tests()


if __name__ == "__main__":
    main()
