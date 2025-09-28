"""Streamlit UI scaffold for the investment assistant."""
from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.backtest.engine import BtConfig, run_backtest
from app.data.schema import initialize_database
from app.ingest.checker import run_boot_check
from app.ingest.tushare import FetchJob, run_ingestion
from app.llm.client import llm_config_snapshot, run_llm
from app.utils.config import (
    ALLOWED_LLM_STRATEGIES,
    DEFAULT_LLM_BASE_URLS,
    DEFAULT_LLM_MODEL_OPTIONS,
    DEFAULT_LLM_MODELS,
    DepartmentSettings,
    LLMProfile,
    LLMRoute,
    get_config,
    save_config,
)
from app.utils.db import db_session
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "ui"}


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
            }
            continue

        if agent_name.startswith("dept_"):
            code = agent_name.split("dept_", 1)[-1]
            signals = utils.get("_signals", [])
            risks = utils.get("_risks", [])
            dept_records.append(
                {
                    "部门": code,
                    "行动": action,
                    "信心": float(utils.get("_confidence", 0.0)),
                    "权重": weight,
                    "摘要": utils.get("_summary", ""),
                    "核心信号": "；".join(signals) if isinstance(signals, list) else signals,
                    "风险提示": "；".join(risks) if isinstance(risks, list) else risks,
                }
            )
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
    else:
        st.info("暂未写入全局策略摘要。")

    st.subheader("部门意见")
    if dept_records:
        dept_df = pd.DataFrame(dept_records)
        st.dataframe(dept_df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无部门记录。")

    st.subheader("代理评分")
    if agent_records:
        agent_df = pd.DataFrame(agent_records)
        st.dataframe(agent_df, use_container_width=True, hide_index=True)
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
        with st.spinner("正在执行回测..."):
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
                result = run_backtest(cfg)
                LOGGER.info(
                    "回测完成：nav_records=%s trades=%s",
                    len(result.nav_series),
                    len(result.trades),
                    extra=LOG_EXTRA,
                )
                st.success("回测执行完成，详见回测结果摘要。")
                st.json({"nav_records": result.nav_series, "trades": result.trades})
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("回测执行失败", extra=LOG_EXTRA)
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
    profiles = cfg.llm_profiles or {}
    routes = cfg.llm_routes or {}
    profile_keys = sorted(profiles.keys())
    route_keys = sorted(routes.keys())
    used_routes = {
        dept.llm_route for dept in cfg.departments.values() if dept.llm_route
    }
    st.caption("Profile 定义单个模型终端，Route 负责组合 Profile 与推理策略。")

    route_select_col, route_manage_col = st.columns([3, 1])
    if route_keys:
        try:
            active_index = route_keys.index(cfg.llm_route)
        except ValueError:
            active_index = 0
        selected_route = route_select_col.selectbox(
            "全局路由",
            route_keys,
            index=active_index,
            key="llm_route_select",
        )
    else:
        selected_route = None
        route_select_col.info("尚未配置路由，请先创建。")

    new_route_name = route_manage_col.text_input("新增路由", key="new_route_name")
    if route_manage_col.button("添加路由"):
        key = (new_route_name or "").strip()
        if not key:
            st.warning("请输入有效的路由名称。")
        elif key in routes:
            st.warning(f"路由 {key} 已存在。")
        else:
            routes[key] = LLMRoute(name=key)
            if not selected_route:
                selected_route = key
                cfg.llm_route = key
            save_config()
            st.success(f"已添加路由 {key}，请继续配置。")
            st.experimental_rerun()

    if selected_route:
        route_obj = routes.get(selected_route)
        if route_obj is None:
            route_obj = LLMRoute(name=selected_route)
            routes[selected_route] = route_obj
        strategy_choices = sorted(ALLOWED_LLM_STRATEGIES)
        try:
            strategy_index = strategy_choices.index(route_obj.strategy)
        except ValueError:
            strategy_index = 0
        route_title = st.text_input(
            "路由说明",
            value=route_obj.title or "",
            key=f"route_title_{selected_route}",
        )
        route_strategy = st.selectbox(
            "推理策略",
            strategy_choices,
            index=strategy_index,
            key=f"route_strategy_{selected_route}",
        )
        route_majority = st.number_input(
            "多数投票门槛",
            min_value=1,
            max_value=10,
            value=int(route_obj.majority_threshold or 1),
            step=1,
            key=f"route_majority_{selected_route}",
        )
        if not profile_keys:
            st.warning("暂无可用 Profile，请先在下方创建。")
        else:
            try:
                primary_index = profile_keys.index(route_obj.primary)
            except ValueError:
                primary_index = 0
            primary_key = st.selectbox(
                "主用 Profile",
                profile_keys,
                index=primary_index,
                key=f"route_primary_{selected_route}",
            )
            default_ensemble = [
                key for key in route_obj.ensemble if key in profile_keys and key != primary_key
            ]
            ensemble_keys = st.multiselect(
                "协作 Profile (可多选)",
                profile_keys,
                default=default_ensemble,
                key=f"route_ensemble_{selected_route}",
            )
            if st.button("保存路由设置", key=f"save_route_{selected_route}"):
                route_obj.title = route_title.strip()
                route_obj.strategy = route_strategy
                route_obj.majority_threshold = int(route_majority)
                route_obj.primary = primary_key
                route_obj.ensemble = [key for key in ensemble_keys if key != primary_key]
                cfg.llm_route = selected_route
                cfg.sync_runtime_llm()
                save_config()
                LOGGER.info(
                    "路由 %s 配置更新：%s",
                    selected_route,
                    route_obj.to_dict(),
                    extra=LOG_EXTRA,
                )
                st.success("路由配置已保存。")
                st.json({
                    "route": selected_route,
                    "route_detail": route_obj.to_dict(),
                    "resolved": llm_config_snapshot(),
                })
        route_in_use = selected_route in used_routes or selected_route == cfg.llm_route
        if st.button(
            "删除当前路由",
            key=f"delete_route_{selected_route}",
            disabled=route_in_use or len(routes) <= 1,
        ):
            routes.pop(selected_route, None)
            if cfg.llm_route == selected_route:
                cfg.llm_route = next((key for key in routes.keys()), "global")
            cfg.sync_runtime_llm()
            save_config()
            st.success("路由已删除。")
            st.experimental_rerun()

    st.divider()
    st.subheader("LLM Profile 管理")
    profile_select_col, profile_manage_col = st.columns([3, 1])
    if profile_keys:
        selected_profile = profile_select_col.selectbox(
            "选择 Profile",
            profile_keys,
            index=0,
            key="profile_select",
        )
    else:
        selected_profile = None
        profile_select_col.info("尚未配置 Profile，请先创建。")

    new_profile_name = profile_manage_col.text_input("新增 Profile", key="new_profile_name")
    if profile_manage_col.button("创建 Profile"):
        key = (new_profile_name or "").strip()
        if not key:
            st.warning("请输入有效的 Profile 名称。")
        elif key in profiles:
            st.warning(f"Profile {key} 已存在。")
        else:
            profiles[key] = LLMProfile(key=key)
            save_config()
            st.success(f"已创建 Profile {key}。")
            st.experimental_rerun()

    if selected_profile:
        profile = profiles[selected_profile]
        provider_choices = sorted(DEFAULT_LLM_MODEL_OPTIONS.keys())
        try:
            provider_index = provider_choices.index(profile.provider)
        except ValueError:
            provider_index = 0
        with st.form(f"profile_form_{selected_profile}"):
            provider_val = st.selectbox(
                "Provider",
                provider_choices,
                index=provider_index,
            )
            model_default = DEFAULT_LLM_MODELS.get(provider_val, profile.model or "")
            model_val = st.text_input(
                "模型",
                value=profile.model or model_default,
            )
            base_default = DEFAULT_LLM_BASE_URLS.get(provider_val, profile.base_url or "")
            base_val = st.text_input(
                "Base URL",
                value=profile.base_url or base_default,
            )
            api_val = st.text_input(
                "API Key",
                value=profile.api_key or "",
                type="password",
            )
            temp_val = st.slider(
                "温度",
                min_value=0.0,
                max_value=2.0,
                value=float(profile.temperature),
                step=0.05,
            )
            timeout_val = st.number_input(
                "超时(秒)",
                min_value=5,
                max_value=180,
                value=int(profile.timeout or 30),
                step=5,
            )
            title_val = st.text_input("备注", value=profile.title or "")
            enabled_val = st.checkbox("启用", value=profile.enabled)
            submitted = st.form_submit_button("保存 Profile")
        if submitted:
            profile.provider = provider_val
            profile.model = model_val.strip() or DEFAULT_LLM_MODELS.get(provider_val)
            profile.base_url = base_val.strip() or DEFAULT_LLM_BASE_URLS.get(provider_val)
            profile.api_key = api_val.strip() or None
            profile.temperature = temp_val
            profile.timeout = timeout_val
            profile.title = title_val.strip()
            profile.enabled = enabled_val
            profiles[selected_profile] = profile
            cfg.sync_runtime_llm()
            save_config()
            st.success("Profile 已保存。")

        profile_in_use = any(
            selected_profile == route.primary or selected_profile in route.ensemble
            for route in routes.values()
        )
        if st.button(
            "删除该 Profile",
            key=f"delete_profile_{selected_profile}",
            disabled=profile_in_use or len(profiles) <= 1,
        ):
            profiles.pop(selected_profile, None)
            fallback_key = next((key for key in profiles.keys()), None)
            for route in routes.values():
                if route.primary == selected_profile:
                    route.primary = fallback_key or route.primary
                route.ensemble = [key for key in route.ensemble if key != selected_profile]
            cfg.sync_runtime_llm()
            save_config()
            st.success("Profile 已删除。")
            st.experimental_rerun()

    st.divider()
    st.subheader("部门配置")

    dept_settings = cfg.departments or {}
    route_options_display = [""] + route_keys
    dept_rows = [
        {
            "code": code,
            "title": dept.title,
            "description": dept.description,
            "weight": float(dept.weight),
            "llm_route": dept.llm_route or "",
            "strategy": dept.llm.strategy,
            "primary_provider": (dept.llm.primary.provider or ""),
            "primary_model": dept.llm.primary.model or "",
            "ensemble_size": len(dept.llm.ensemble),
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
        use_container_width=True,
        hide_index=True,
        column_config={
            "code": st.column_config.TextColumn("编码", disabled=True),
            "title": st.column_config.TextColumn("名称"),
            "description": st.column_config.TextColumn("说明"),
            "weight": st.column_config.NumberColumn("权重", min_value=0.0, max_value=10.0, step=0.1),
            "llm_route": st.column_config.SelectboxColumn(
                "路由",
                options=route_options_display,
                help="选择预定义路由；留空表示使用自定义配置",
            ),
            "strategy": st.column_config.SelectboxColumn(
                "自定义策略",
                options=sorted(ALLOWED_LLM_STRATEGIES),
                help="仅当未选择路由时生效",
            ),
            "primary_provider": st.column_config.SelectboxColumn(
                "自定义 Provider",
                options=sorted(DEFAULT_LLM_MODEL_OPTIONS.keys()),
            ),
            "primary_model": st.column_config.TextColumn("自定义模型"),
            "ensemble_size": st.column_config.NumberColumn(
                "协作模型数量",
                disabled=True,
                help="路由模式下自动维护",
            ),
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

            route_name = (row.get("llm_route") or "").strip() or None
            existing.llm_route = route_name
            if route_name and route_name in routes:
                existing.llm = routes[route_name].resolve(profiles)
            else:
                strategy_val = (row.get("strategy") or existing.llm.strategy).lower()
                if strategy_val in ALLOWED_LLM_STRATEGIES:
                    existing.llm.strategy = strategy_val
                provider_before = existing.llm.primary.provider or ""
                provider_val = (row.get("primary_provider") or provider_before or "ollama").lower()
                existing.llm.primary.provider = provider_val
                model_val = (row.get("primary_model") or "").strip()
                existing.llm.primary.model = (
                    model_val or DEFAULT_LLM_MODELS.get(provider_val, existing.llm.primary.model)
                )
                if provider_before != provider_val:
                    default_base = DEFAULT_LLM_BASE_URLS.get(provider_val)
                    existing.llm.primary.base_url = default_base or existing.llm.primary.base_url
                existing.llm.primary.__post_init__()
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
        st.experimental_rerun()

    st.caption("选择路由可统一部门模型调用，自定义模式仍支持逐项配置。")
    st.caption("部门协作模型（ensemble）请在 config.json 中手动编辑，UI 将在后续版本补充。")


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
    st.plotly_chart(candle_fig, use_container_width=True)

    vol_fig = px.bar(
        df_reset,
        x="交易日",
        y="成交量(手)",
        labels={"成交量(手)": "成交量(手)"},
        title="成交量",
    )
    vol_fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(vol_fig, use_container_width=True)

    amt_fig = px.bar(
        df_reset,
        x="交易日",
        y="成交额(千元)",
        labels={"成交额(千元)": "成交额(千元)"},
        title="成交额",
    )
    amt_fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(amt_fig, use_container_width=True)

    df_reset["月份"] = df_reset["交易日"].dt.to_period("M").astype(str)
    box_fig = px.box(
        df_reset,
        x="月份",
        y="收盘价",
        points="outliers",
        title="月度收盘价分布",
    )
    box_fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(box_fig, use_container_width=True)

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
