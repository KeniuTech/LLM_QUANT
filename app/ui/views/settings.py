"""系统设置相关视图。"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.exceptions import RequestException
import streamlit as st

from app.llm.client import llm_config_snapshot
from app.llm.metrics import snapshot as llm_metrics_snapshot
from app.llm.templates import TemplateRegistry
from app.utils.config import (
    ALLOWED_LLM_STRATEGIES,
    DEFAULT_LLM_BASE_URLS,
    DepartmentSettings,
    LLMEndpoint,
    LLMProvider,
    get_config,
    save_config,
)
from app.utils.db import db_session

from app.ui.shared import LOGGER, LOG_EXTRA

_MODEL_CACHE: Dict[str, Dict[str, object]] = {}
_CACHE_TTL_SECONDS = 300

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

def render_config_overview() -> None:
    """Render a concise overview of persisted configuration values."""

    LOGGER.info("渲染配置概览页", extra=LOG_EXTRA)
    cfg = get_config()

    st.subheader("核心配置概览")
    col1, col2, col3 = st.columns(3)
    col1.metric("决策方式", cfg.decision_method.upper())
    col2.metric("自动更新数据", "启用" if cfg.auto_update_data else "关闭")
    col3.metric("数据更新间隔(天)", cfg.data_update_interval)

    col4, col5, col6 = st.columns(3)
    col4.metric("强制刷新", "开启" if cfg.force_refresh else "关闭")
    col5.metric("TuShare Token", "已配置" if cfg.tushare_token else "未配置")
    col6.metric("配置文件", cfg.data_paths.config_file.name)
    st.caption(f"配置文件路径：{cfg.data_paths.config_file}")

    st.divider()
    st.subheader("RSS 数据源状态")
    rss_sources = cfg.rss_sources or {}
    if rss_sources:
        rows: List[Dict[str, object]] = []
        for name, payload in rss_sources.items():
            if isinstance(payload, dict):
                rows.append(
                    {
                        "名称": name,
                        "启用": "是" if payload.get("enabled", True) else "否",
                        "URL": payload.get("url", "-"),
                        "关键词数": len(payload.get("keywords", []) or []),
                    }
                )
            elif isinstance(payload, bool):
                rows.append(
                    {
                        "名称": name,
                        "启用": "是" if payload else "否",
                        "URL": "-",
                        "关键词数": 0,
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("未配置 RSS 数据源。")
    else:
        st.info("未在配置文件中找到 RSS 数据源。")

    st.divider()
    st.subheader("部门配置")
    dept_rows: List[Dict[str, object]] = []
    for code, dept in cfg.departments.items():
        dept_rows.append(
            {
                "部门": dept.title or code,
                "代码": code,
                "权重": dept.weight,
                "LLM 策略": dept.llm.strategy,
                "模板": dept.prompt_template_id or f"{code}_dept",
                "模板版本": dept.prompt_template_version or "(激活版本)",
            }
        )
    if dept_rows:
        st.dataframe(pd.DataFrame(dept_rows), hide_index=True, width="stretch")
    else:
        st.info("尚未配置任何部门。")

    st.divider()
    st.subheader("LLM 成本控制")
    cost = cfg.llm_cost
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("成本控制", "启用" if cost.enabled else "关闭")
    col_b.metric("小时预算($)", f"{cost.hourly_budget:.2f}")
    col_c.metric("日预算($)", f"{cost.daily_budget:.2f}")
    col_d.metric("月预算($)", f"{cost.monthly_budget:.2f}")

    if cost.model_weights:
        weight_rows = (
            pd.DataFrame(
                [
                    {"模型": model, "占比上限": f"{limit * 100:.0f}%"}
                    for model, limit in cost.model_weights.items()
                ]
            )
        )
        st.dataframe(weight_rows, hide_index=True, width="stretch")
    else:
        st.caption("未配置模型占比限制。")

    st.divider()
    st.caption("提示：数据源、LLM 及投资组合设置可在对应标签页中调整。")

def render_llm_settings() -> None:
    cfg = get_config()
    st.subheader("LLM 设置")
    providers = cfg.llm_providers
    provider_keys = sorted(providers.keys())
    st.caption("先在 Provider 中维护基础连接（URL、Key、模型），再为全局与各部门设置个性化参数。")

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
            # Avoid modifying st.session_state after the widget with the same key
            # has been created (Streamlit raises an exception). Rerun so the
            # page is re-executed and widgets are recreated with updated values.
            st.rerun()

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

    st.divider()
    st.markdown("##### 提示模板治理")
    template_ids = TemplateRegistry.list_template_ids()
    if not template_ids:
        st.info("尚未注册任何提示模板。")
    else:
        template_id = st.selectbox(
            "选择模板",
            template_ids,
            key="prompt_template_select",
        )
        version_details = TemplateRegistry.list_version_details(template_id)
        raw_versions = TemplateRegistry.list_versions(template_id)
        active_version = None
        if version_details:
            for detail in version_details:
                if detail.get("is_active"):
                    active_version = detail["version"]
                    break
            if active_version is None:
                active_version = version_details[0]["version"]
        usage_snapshot = llm_metrics_snapshot()
        template_usage = usage_snapshot.get("template_usage", {}).get(template_id, {})

        table_rows: List[Dict[str, object]] = []
        for detail in version_details:
            metadata_preview = detail.get("metadata") or {}
            table_rows.append(
                {
                    "版本": detail["version"],
                    "创建时间": detail.get("created_at") or "-",
                    "激活": "是" if detail.get("is_active") else "否",
                    "元数据": json.dumps(metadata_preview, ensure_ascii=False, default=str) if metadata_preview else "{}",
                }
            )
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

        version_options = [row["版本"] for row in table_rows] if table_rows else []
        if not version_options:
            st.info("当前模板尚未创建版本，建议通过配置文件或 API 注册。")
        else:
            try:
                default_idx = version_options.index(active_version or version_options[0])
            except ValueError:
                default_idx = 0
            selected_version = st.selectbox(
                "查看版本",
                version_options,
                index=default_idx,
                key=f"{template_id}_version_select",
            )
            selected_detail = next(
                (detail for detail in version_details if detail["version"] == selected_version),
                {"metadata": {}},
            )
            usage_cols = st.columns(3)
            usage_cols[0].metric("累计调用", int(template_usage.get("total_calls", 0)))
            version_usage = (template_usage.get("versions") or {}).get(selected_version, {})
            usage_cols[1].metric("版本调用", int(version_usage.get("calls", 0)))
            usage_cols[2].metric(
                "Prompt Tokens",
                int(version_usage.get("prompt_tokens", 0)),
            )

            template_obj = TemplateRegistry.get(template_id, version=selected_version)
            if template_obj:
                with st.expander("模板内容预览", expanded=False):
                    st.write(f"名称：{template_obj.name}")
                    st.write(f"描述：{template_obj.description or '-'}")
                    st.write(f"变量：{', '.join(template_obj.variables) if template_obj.variables else '无'}")
                    st.code(template_obj.template, language="markdown")

            metadata_str = json.dumps(selected_detail.get("metadata") or {}, ensure_ascii=False, indent=2, default=str)
            metadata_input = st.text_area(
                "版本元数据（JSON）",
                value=metadata_str,
                height=200,
                key=f"{template_id}_{selected_version}_metadata",
            )
            meta_buttons = st.columns(3)
            enable_version_actions = bool(raw_versions)
            if meta_buttons[0].button(
                "保存元数据",
                key=f"{template_id}_{selected_version}_save_metadata",
                disabled=not enable_version_actions,
            ):
                try:
                    new_metadata = json.loads(metadata_input or "{}")
                except json.JSONDecodeError as exc:
                    st.error(f"元数据格式错误：{exc}")
                else:
                    try:
                        TemplateRegistry.update_version_metadata(template_id, selected_version, new_metadata)
                        st.success("元数据已更新。")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"更新元数据失败：{exc}")

            if meta_buttons[1].button(
                "设为激活版本",
                key=f"{template_id}_{selected_version}_activate",
                disabled=(selected_version == active_version) or not enable_version_actions,
            ):
                try:
                    TemplateRegistry.activate_version(template_id, selected_version)
                    st.success(f"{template_id} 已切换至版本 {selected_version}。")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"切换版本失败：{exc}")

            export_payload = TemplateRegistry.export_versions(template_id) if enable_version_actions else None
            meta_buttons[2].download_button(
                "导出版本 JSON",
                data=export_payload or "",
                file_name=f"{template_id}_versions.json",
                mime="application/json",
                key=f"{template_id}_download_versions",
                disabled=not export_payload,
            )

    st.divider()
    st.markdown("##### 部门遥测可视化")
    telemetry_limit = st.slider(
        "遥测查询条数",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="限制查询 agent_utils 表中的最新记录数量。",
        key="telemetry_limit",
    )
    telemetry_rows: List[Dict[str, object]] = []
    try:
        with db_session(read_only=True) as conn:
            raw_rows = conn.execute(
                """
                SELECT trade_date, ts_code, agent, utils
                FROM agent_utils
                ORDER BY trade_date DESC, ts_code
                LIMIT ?
                """,
                (telemetry_limit,),
            ).fetchall()
    except Exception:  # noqa: BLE001
        LOGGER.exception("读取 agent_utils 遥测失败", extra=LOG_EXTRA)
        raw_rows = []

    for row in raw_rows:
        trade_date = row["trade_date"]
        ts_code = row["ts_code"]
        agent = row["agent"]
        try:
            utils_payload = json.loads(row["utils"] or "{}")
        except json.JSONDecodeError:
            utils_payload = {}

        if agent == "global":
            telemetry_map = utils_payload.get("_department_telemetry") or {}
            for dept_code, payload in telemetry_map.items():
                if not isinstance(payload, dict):
                    payload = {"value": payload}
                record = {
                    "trade_date": trade_date,
                    "ts_code": ts_code,
                    "agent": agent,
                    "department": dept_code,
                    "source": "global",
                    "telemetry": json.dumps(payload, ensure_ascii=False, default=str),
                }
                for key, value in payload.items():
                    if isinstance(value, (int, float, bool, str)):
                        record.setdefault(key, value)
                telemetry_rows.append(record)
        elif agent.startswith("dept_"):
            dept_code = agent.split("dept_", 1)[-1]
            payload = utils_payload.get("_telemetry") or {}
            if not isinstance(payload, dict):
                payload = {"value": payload}
            record = {
                "trade_date": trade_date,
                "ts_code": ts_code,
                "agent": agent,
                "department": dept_code,
                "source": "department",
                "telemetry": json.dumps(payload, ensure_ascii=False, default=str),
            }
            for key, value in payload.items():
                if isinstance(value, (int, float, bool, str)):
                    record.setdefault(key, value)
            telemetry_rows.append(record)

    if not telemetry_rows:
        st.info("未找到遥测记录，可先运行部门评估流程。")
    else:
        telemetry_df = pd.DataFrame(telemetry_rows)
        telemetry_df["trade_date"] = telemetry_df["trade_date"].astype(str)
        trade_dates = sorted(telemetry_df["trade_date"].unique(), reverse=True)
        selected_trade_date = st.selectbox(
            "交易日",
            trade_dates,
            index=0,
            key="telemetry_trade_date",
        )
        filtered_df = telemetry_df[telemetry_df["trade_date"] == selected_trade_date]
        departments = sorted(filtered_df["department"].dropna().unique())
        selected_departments = st.multiselect(
            "部门过滤",
            departments,
            default=departments,
            key="telemetry_departments",
        )
        if selected_departments:
            filtered_df = filtered_df[filtered_df["department"].isin(selected_departments)]
        numeric_columns = [
            col
            for col in filtered_df.columns
            if col not in {"trade_date", "ts_code", "agent", "department", "source", "telemetry"}
            and pd.api.types.is_numeric_dtype(filtered_df[col])
        ]
        metric_cols = st.columns(min(3, max(1, len(numeric_columns))))
        for idx, column in enumerate(numeric_columns[: len(metric_cols)]):
            column_series = filtered_df[column].dropna()
            value = column_series.mean() if not column_series.empty else 0.0
            metric_cols[idx].metric(f"{column} 均值", f"{value:.2f}")
        st.dataframe(filtered_df, hide_index=True, width="stretch")
        st.download_button(
            "下载遥测 CSV",
            data=filtered_df.to_csv(index=False),
            file_name=f"telemetry_{selected_trade_date}.csv",
            mime="text/csv",
            key="telemetry_download",
        )
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
            SELECT job_type, status, 
                   CAST(created_at AS TIMESTAMP) as created_at,
                   CAST(updated_at AS TIMESTAMP) as updated_at,
                   error_msg
            FROM fetch_jobs 
            ORDER BY created_at DESC 
            LIMIT 50
            """,
            session
        )
        
    if not df.empty:
        df["duration"] = (pd.to_datetime(df["updated_at"]) - pd.to_datetime(df["created_at"])).dt.total_seconds().round(2)
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
