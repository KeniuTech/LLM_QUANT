"""回测与复盘相关视图。"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from app.agents.base import AgentContext
from app.agents.game import Decision
from app.agents.registry import default_agents
from app.agents.protocols import GameStructure
from app.backtest.decision_env import DecisionEnv, ParameterSpec
from app.backtest.optimizer import BanditConfig, EpsilonGreedyBandit
from app.rl import TORCH_AVAILABLE, DecisionEnvAdapter, PPOConfig, train_ppo
from app.backtest.engine import BacktestEngine, PortfolioState, BtConfig, run_backtest
from app.ingest.checker import run_boot_check
from app.ingest.tushare import run_ingestion
from app.llm.client import run_llm
from app.llm.metrics import reset as reset_llm_metrics
from app.llm.metrics import snapshot as snapshot_llm_metrics
from app.llm.templates import TemplateRegistry
from app.utils import alerts
from app.utils.config import get_config, save_config
from app.utils.tuning import log_tuning_result
from app.utils.portfolio import (
    get_candidate_pool,
    get_portfolio_settings_snapshot,
)

from app.utils.db import db_session

from app.ui.shared import LOGGER, LOG_EXTRA, default_backtest_range
from app.ui.views.dashboard import update_dashboard_sidebar

_DECISION_ENV_SINGLE_RESULT_KEY = "decision_env_single_result"
_DECISION_ENV_BATCH_RESULTS_KEY = "decision_env_batch_results"
_DECISION_ENV_BANDIT_RESULTS_KEY = "decision_env_bandit_results"
_DECISION_ENV_PPO_RESULTS_KEY = "decision_env_ppo_results"

def render_backtest_review() -> None:
    """渲染回测执行、调参与结果复盘页面。"""
    st.header("回测与复盘")
    st.caption("1. 基于历史数据复盘当前策略；2. 借助强化学习/调参探索更优参数组合。")
    app_cfg = get_config()
    portfolio_snapshot = get_portfolio_settings_snapshot()
    default_start, default_end = default_backtest_range(window_days=60)
    LOGGER.debug(
        "回测默认参数：start=%s end=%s universe=%s target=%s stop=%s hold_days=%s initial_capital=%s",
        default_start,
        default_end,
        "000001.SZ",
        0.035,
        -0.015,
        10,
        get_config().portfolio.initial_capital,
        extra=LOG_EXTRA,
    )

    st.markdown("### 回测参数")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("开始日期", value=default_start, key="bt_start_date")
    end_date = col2.date_input("结束日期", value=default_end, key="bt_end_date")

    candidate_records, candidate_fallback = get_candidate_pool(limit=50)
    candidate_codes = [item.ts_code for item in candidate_records]
    default_universe = ",".join(candidate_codes) if candidate_codes else "000001.SZ"
    universe_text = st.text_input(
        "股票列表（逗号分隔）",
        value=default_universe,
        key="bt_universe",
        help="默认载入最新候选池，如需自定义可直接编辑。",
    )
    if candidate_codes:
        message = f"候选池载入 {len(candidate_codes)} 个标的：{'、'.join(candidate_codes[:10])}{'…' if len(candidate_codes)>10 else ''}"
        if candidate_fallback:
            message += "（使用最新候选池作为回退）"
        st.caption(message)
    col_target, col_stop, col_hold, col_cap = st.columns(4)
    target = col_target.number_input("目标收益（例：0.035 表示 3.5%）", value=0.035, step=0.005, format="%.3f", key="bt_target")
    stop = col_stop.number_input("止损收益（例：-0.015 表示 -1.5%）", value=-0.015, step=0.005, format="%.3f", key="bt_stop")
    hold_days = col_hold.number_input("持有期（交易日）", value=10, step=1, key="bt_hold_days")
    initial_capital_default = float(portfolio_snapshot["initial_capital"])
    initial_capital = col_cap.number_input(
        "组合初始资金",
        value=initial_capital_default,
        step=100000.0,
        format="%.0f",
        key="bt_initial_capital",
    )
    initial_capital = max(0.0, float(initial_capital))
    position_limits = portfolio_snapshot.get("position_limits", {})
    backtest_params = {
        "target": float(target),
        "stop": float(stop),
        "hold_days": int(hold_days),
        "initial_capital": initial_capital,
        "max_position_weight": float(position_limits.get("max_position", 0.2)),
        "max_total_positions": int(position_limits.get("max_total_positions", 20)),
    }

    st.caption(
        "组合约束：单仓上限 {max_pos:.0%} ｜ 最大持仓 {max_count} ｜ 行业敞口 {sector:.0%}".format(
            max_pos=backtest_params["max_position_weight"],
            max_count=position_limits.get("max_total_positions", 20),
            sector=position_limits.get("max_sector_exposure", 0.35),
        )
    )
    structure_options = [item.value for item in GameStructure]
    selected_structure_values = st.multiselect(
        "选择博弈框架",
        structure_options,
        default=structure_options,
        key="bt_game_structures",
    )
    if not selected_structure_values:
        selected_structure_values = [GameStructure.REPEATED.value]
    selected_structures = [GameStructure(value) for value in selected_structure_values]
    LOGGER.debug(
        "当前回测表单输入：start=%s end=%s universe_text=%s target=%.3f stop=%.3f hold_days=%s initial_capital=%.2f",
        start_date,
        end_date,
        universe_text,
        target,
        stop,
        hold_days,
        initial_capital,
        extra=LOG_EXTRA,
    )

    tab_backtest, tab_rl = st.tabs(["回测验证", "强化学习调参"])

    with tab_backtest:
        st.markdown("#### 回测执行")
        if st.button("运行回测", key="bt_run_button"):
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
                update_dashboard_sidebar(stats)

            reset_llm_metrics()
            status_box.update(label="执行回测中...", state="running")
            try:
                universe = [code.strip() for code in universe_text.split(',') if code.strip()]
                LOGGER.info(
                    "回测参数：start=%s end=%s universe=%s target=%s stop=%s hold_days=%s initial_capital=%.2f",
                    start_date,
                    end_date,
                    universe,
                    target,
                    stop,
                    hold_days,
                    initial_capital,
                    extra=LOG_EXTRA,
                )
                backtest_cfg = BtConfig(
                    id="streamlit_demo",
                    name="Streamlit Demo Strategy",
                    start_date=start_date,
                    end_date=end_date,
                    universe=universe,
                    params=dict(backtest_params),
                    game_structures=selected_structures,
                )
                result = run_backtest(backtest_cfg, decision_callback=_decision_callback)
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
                update_dashboard_sidebar(metrics)
                st.session_state["backtest_last_result"] = {"nav_records": result.nav_series, "trades": result.trades}
                st.json(st.session_state["backtest_last_result"])
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("回测执行失败", extra=LOG_EXTRA)
                status_box.update(label="回测执行失败", state="error")
                st.error(f"回测执行失败：{exc}")

        last_result = st.session_state.get("backtest_last_result")
        if last_result:
            st.markdown("#### 最近回测输出")
            st.json(last_result)

        st.divider()
        # ADD: Comparison view for multiple backtest configurations
        st.caption("从历史回测配置中选择多个进行净值曲线与指标对比。")
        normalize_to_one = st.checkbox("归一化到 1 起点", value=True, key="bt_cmp_normalize")
        use_log_y = st.checkbox("对数坐标", value=False, key="bt_cmp_log_y")
        metric_options = ["总收益", "最大回撤", "交易数", "平均换手", "风险事件"]
        selected_metrics = st.multiselect("显示指标列", metric_options, default=metric_options, key="bt_cmp_metrics")
        try:
            with db_session(read_only=True) as conn:
                cfg_rows = conn.execute(
                    "SELECT id, name FROM bt_config ORDER BY rowid DESC LIMIT 50"
                ).fetchall()
        except Exception:  # noqa: BLE001
            LOGGER.exception("读取 bt_config 失败", extra=LOG_EXTRA)
            cfg_rows = []
        cfg_options = [f"{row['id']} | {row['name']}" for row in cfg_rows]
        selected_labels = st.multiselect("选择配置", cfg_options, default=cfg_options[:2], key="bt_cmp_configs")
        selected_ids = [label.split(" | ")[0].strip() for label in selected_labels]
        nav_df = pd.DataFrame()
        rpt_df = pd.DataFrame()
        risk_df = pd.DataFrame()
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
                    risk_df = pd.read_sql_query(
                        "SELECT cfg_id, trade_date, ts_code, reason, action, target_weight, confidence, metadata "
                        "FROM bt_risk_events WHERE cfg_id IN (%s)" % (",".join(["?"]*len(selected_ids))),
                        conn,
                        params=tuple(selected_ids),
                    )
            except Exception:  # noqa: BLE001
                LOGGER.exception("读取回测结果失败", extra=LOG_EXTRA)
                st.error("读取回测结果失败")
                nav_df = pd.DataFrame()
                rpt_df = pd.DataFrame()
                risk_df = pd.DataFrame()
            start_filter: Optional[date] = None
            end_filter: Optional[date] = None
            if not nav_df.empty:
                try:
                    nav_df["trade_date"] = pd.to_datetime(nav_df["trade_date"], errors="coerce")
                    # ADD: date window filter
                    overall_min = pd.to_datetime(nav_df["trade_date"].min()).date()
                    overall_max = pd.to_datetime(nav_df["trade_date"].max()).date()
                    col_d1, col_d2 = st.columns(2)
                    start_filter = col_d1.date_input("起始日期", value=overall_min, key="bt_cmp_start")
                    end_filter = col_d2.date_input("结束日期", value=overall_max, key="bt_cmp_end")
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
                            "风险分布": json.dumps(summary.get("risk_breakdown"), ensure_ascii=False)
                            if summary.get("risk_breakdown")
                            else None,
                            "缺失字段": json.dumps(summary.get("missing_field_counts"), ensure_ascii=False)
                            if summary.get("missing_field_counts")
                            else None,
                            "派生字段": json.dumps(summary.get("derived_field_counts"), ensure_ascii=False)
                            if summary.get("derived_field_counts")
                            else None,
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
            if not risk_df.empty:
                try:
                    risk_df["trade_date"] = pd.to_datetime(risk_df["trade_date"], errors="coerce")
                    risk_df = risk_df.dropna(subset=["trade_date"])
                    if start_filter is None or end_filter is None:
                        start_filter = pd.to_datetime(risk_df["trade_date"].min()).date()
                        end_filter = pd.to_datetime(risk_df["trade_date"].max()).date()
                    risk_df = risk_df[
                        (risk_df["trade_date"].dt.date >= start_filter)
                        & (risk_df["trade_date"].dt.date <= end_filter)
                    ]
                    parsed_cols: List[Dict[str, object]] = []
                    for _, row in risk_df.iterrows():
                        try:
                            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else (row["metadata"] or {})
                        except json.JSONDecodeError:
                            metadata = {}
                        assessment = metadata.get("risk_assessment") or {}
                        parsed_cols.append(
                            {
                                "cfg_id": row["cfg_id"],
                                "trade_date": row["trade_date"].date().isoformat(),
                                "ts_code": row["ts_code"],
                                "reason": row["reason"],
                                "action": row["action"],
                                "target_weight": row["target_weight"],
                                "confidence": row["confidence"],
                                "risk_status": assessment.get("status"),
                                "recommended_action": assessment.get("recommended_action"),
                                "execution_status": metadata.get("execution_status"),
                                "metadata": metadata,
                            }
                        )
                    risk_detail_df = pd.DataFrame(parsed_cols)
                    with st.expander("风险事件明细", expanded=False):
                        st.dataframe(risk_detail_df.drop(columns=["metadata"], errors="ignore"), hide_index=True, width='stretch')
                        try:
                            st.download_button(
                                "下载风险事件(CSV)",
                                data=risk_detail_df.to_csv(index=False),
                                file_name="bt_risk_events.csv",
                                mime="text/csv",
                                key="dl_risk_events",
                            )
                        except Exception:
                            pass
                        agg = risk_detail_df.groupby(["cfg_id", "reason", "risk_status"], dropna=False).size().reset_index(name="count")
                        st.dataframe(agg, hide_index=True, width='stretch')
                        try:
                            if not agg.empty:
                                agg_fig = px.bar(
                                    agg,
                                    x="reason",
                                    y="count",
                                    color="risk_status",
                                    facet_col="cfg_id",
                                    title="风险事件分布",
                                )
                                agg_fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=20))
                                st.plotly_chart(agg_fig, width="stretch")
                        except Exception:  # noqa: BLE001
                            LOGGER.debug("绘制风险事件分布失败", extra=LOG_EXTRA)
                except Exception:  # noqa: BLE001
                    LOGGER.debug("渲染风险事件失败", extra=LOG_EXTRA)
        else:
            st.info("请选择至少一个配置进行对比。")



    with tab_rl:
        st.caption("使用 DecisionEnv 对代理权重进行强化学习调参，支持单次与批量实验。")

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
            spec_labels: List[str] = []
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
                spec_labels.append(f"agent:{agent_name}")
                action_values.append(action_val)

            controls_valid = True
            
            st.divider()
            st.subheader("部门参数调整（可选）")
            dept_codes = sorted(app_cfg.departments.keys())
            if not dept_codes:
                st.caption("当前未配置部门。")
            else:
                selected_departments = st.multiselect(
                    "选择需要调整的部门",
                    dept_codes,
                    default=[],
                    key="decision_env_departments",
                )
                tool_policy_values = ["auto", "none", "required"]
                for dept_code in selected_departments:
                    settings = app_cfg.departments.get(dept_code)
                    if not settings:
                        continue
                    st.subheader(f"部门：{settings.title or dept_code}")
                    base_temp = 0.2
                    if settings.llm and settings.llm.primary and settings.llm.primary.temperature is not None:
                        base_temp = float(settings.llm.primary.temperature)
                    prefix = f"decision_env_dept_{dept_code}"
                    col_tmin, col_tmax, col_tslider = st.columns([1, 1, 2])
                    temp_min = col_tmin.number_input(
                        "温度最小值",
                        min_value=0.0,
                        max_value=2.0,
                        value=max(0.0, base_temp - 0.3),
                        step=0.05,
                        key=f"{prefix}_temp_min",
                    )
                    temp_max = col_tmax.number_input(
                        "温度最大值",
                        min_value=0.0,
                        max_value=2.0,
                        value=min(2.0, base_temp + 0.3),
                        step=0.05,
                        key=f"{prefix}_temp_max",
                    )
                    if temp_max <= temp_min:
                        controls_valid = False
                        st.warning("温度最大值必须大于最小值。")
                        temp_max = min(2.0, temp_min + 0.01)
                    span = temp_max - temp_min
                    if span <= 0:
                        ratio_default = 0.0
                    else:
                        clamped = min(max(base_temp, temp_min), temp_max)
                        ratio_default = (clamped - temp_min) / span
                    temp_action = col_tslider.slider(
                        "动作值（映射至温度区间）",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(ratio_default),
                        step=0.01,
                        key=f"{prefix}_temp_action",
                    )
                    specs.append(
                        ParameterSpec(
                            name=f"dept_temperature_{dept_code}",
                            target=f"department.{dept_code}.temperature",
                            minimum=temp_min,
                            maximum=temp_max,
                        )
                    )
                    spec_labels.append(f"department:{dept_code}:temperature")
                    action_values.append(temp_action)

                    col_tool, col_hint = st.columns([1, 2])
                    tool_choice = col_tool.selectbox(
                        "函数调用策略",
                        tool_policy_values,
                        index=tool_policy_values.index("auto"),
                        key=f"{prefix}_tool_choice",
                    )
                    col_hint.caption("映射提示：0→auto，0.5→none，1→required。")
                    if len(tool_policy_values) > 1:
                        tool_value = tool_policy_values.index(tool_choice) / (len(tool_policy_values) - 1)
                    else:
                        tool_value = 0.0
                    specs.append(
                        ParameterSpec(
                            name=f"dept_tool_{dept_code}",
                            target=f"department.{dept_code}.function_policy",
                            values=tool_policy_values,
                        )
                    )
                    spec_labels.append(f"department:{dept_code}:tool_choice")
                    action_values.append(tool_value)

                    template_id = (settings.prompt_template_id or f"{dept_code}_dept").strip()
                    versions = [ver for ver in TemplateRegistry.list_versions(template_id) if isinstance(ver, str)]
                    if versions:
                        active_version = TemplateRegistry.get_active_version(template_id)
                        default_version = (
                            settings.prompt_template_version
                            or active_version
                            or versions[0]
                        )
                        try:
                            default_index = versions.index(default_version)
                        except ValueError:
                            default_index = 0
                        version_choice = st.selectbox(
                            "提示模板版本",
                            versions,
                            index=default_index,
                            key=f"{prefix}_template_version",
                            help="离散动作将按版本列表顺序映射，可用于强化学习优化。",
                        )
                        selected_index = versions.index(version_choice)
                        ratio = (
                            0.0
                            if len(versions) == 1
                            else selected_index / (len(versions) - 1)
                        )
                        specs.append(
                            ParameterSpec(
                                name=f"dept_prompt_version_{dept_code}",
                                target=f"department.{dept_code}.prompt_template_version",
                                values=list(versions),
                            )
                        )
                        spec_labels.append(f"department:{dept_code}:prompt_version")
                        action_values.append(ratio)
                        st.caption(
                            f"激活版本：{active_version or '默认'} ｜ 当前选择：{version_choice}"
                        )
                    else:
                        st.caption("当前模板未注册可选提示词版本，继续沿用激活版本。")

            if specs:
                st.caption("动作维度顺序：" + "，".join(spec_labels))

            disable_departments = st.checkbox(
                "禁用部门 LLM（仅规则代理，适合离线快速评估）",
                value=True,
                help="关闭部门调用后不依赖外部 LLM 网络，仅根据规则代理权重模拟。",
            )
            
            st.divider()
            st.subheader("自动探索（epsilon-greedy）")
            col_ep, col_eps, col_seed = st.columns([1, 1, 1])
            bandit_episodes = int(
                col_ep.number_input(
                    "迭代次数",
                    min_value=1,
                    max_value=200,
                    value=10,
                    step=1,
                    key="decision_env_bandit_episodes",
                    help="探索的回合数，越大越充分但耗时越久。",
                )
            )
            bandit_epsilon = float(
                col_eps.slider(
                    "探索比例 ε",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.05,
                    key="decision_env_bandit_epsilon",
                    help="ε 越大，随机探索概率越高。",
                )
            )
            seed_text = col_seed.text_input(
                "随机种子（可选）",
                value="",
                key="decision_env_bandit_seed",
                help="填写整数可复现实验，不填写则随机。",
            ).strip()
            bandit_seed = None
            if seed_text:
                try:
                    bandit_seed = int(seed_text)
                except ValueError:
                    st.warning("随机种子需为整数，已忽略该值。")
                    bandit_seed = None

            run_bandit = st.button("执行自动探索", key="run_decision_env_bandit")
            if run_bandit:
                if not specs:
                    st.warning("请至少配置一个动作维度再执行探索。")
                elif selected_agents and not range_valid:
                    st.error("请确保所有代理的最大权重大于最小权重。")
                elif not controls_valid:
                    st.error("请修正部门参数的取值范围。")
                else:
                    baseline_weights = app_cfg.agent_weights.as_dict()
                    for agent in agent_objects:
                        baseline_weights.setdefault(agent.name, 1.0)

                    universe_env = [code.strip() for code in universe_text.split(',') if code.strip()]
                    if not universe_env:
                        st.error("请先指定至少一个股票代码。")
                    else:
                        bt_cfg_env = BtConfig(
                            id="decision_env_bandit",
                            name="DecisionEnv Bandit",
                            start_date=start_date,
                            end_date=end_date,
                            universe=universe_env,
                            params=dict(backtest_params),
                            method=app_cfg.decision_method,
                            game_structures=selected_structures,
                        )
                        env = DecisionEnv(
                            bt_config=bt_cfg_env,
                            parameter_specs=specs,
                            baseline_weights=baseline_weights,
                            disable_departments=disable_departments,
                        )
                        config = BanditConfig(
                            experiment_id=experiment_id or f"bandit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            strategy=strategy_label or "DecisionEnv",
                            episodes=bandit_episodes,
                            epsilon=bandit_epsilon,
                            seed=bandit_seed,
                        )
                        bandit = EpsilonGreedyBandit(env, config)
                        with st.spinner("自动探索进行中，请稍候..."):
                            summary = bandit.run()

                        episodes_dump: List[Dict[str, object]] = []
                        for idx, episode in enumerate(summary.episodes, start=1):
                            episodes_dump.append(
                                {
                                    "序号": idx,
                                    "奖励": episode.reward,
                                    "动作(raw)": json.dumps(episode.action, ensure_ascii=False),
                                    "参数值": json.dumps(episode.resolved_action, ensure_ascii=False),
                                    "总收益": episode.metrics.total_return,
                                    "最大回撤": episode.metrics.max_drawdown,
                                    "波动率": episode.metrics.volatility,
                                    "权重": json.dumps(episode.weights or {}, ensure_ascii=False),
                                    "部门控制": json.dumps(episode.department_controls or {}, ensure_ascii=False),
                                }
                            )

                        best_episode = summary.best_episode
                        best_index = summary.episodes.index(best_episode) + 1 if best_episode else None
                        st.session_state[_DECISION_ENV_BANDIT_RESULTS_KEY] = {
                            "episodes": episodes_dump,
                            "best_index": best_index,
                            "best": {
                                "reward": best_episode.reward if best_episode else None,
                                "action": best_episode.action if best_episode else None,
                                "resolved_action": best_episode.resolved_action if best_episode else None,
                                "weights": best_episode.weights if best_episode else None,
                                "department_controls": best_episode.department_controls if best_episode else None,
                            },
                            "experiment_id": config.experiment_id,
                            "strategy": config.strategy,
                        }
                        st.success(f"自动探索完成，共执行 {len(episodes_dump)} 轮。")

            bandit_state = st.session_state.get(_DECISION_ENV_BANDIT_RESULTS_KEY)
            if bandit_state:
                st.caption(
                    f"实验 ID：{bandit_state.get('experiment_id')} | 策略：{bandit_state.get('strategy')}"
                )
                episodes_dump = bandit_state.get("episodes") or []
                if episodes_dump:
                    st.dataframe(pd.DataFrame(episodes_dump), hide_index=True, width='stretch')
                best_payload = bandit_state.get("best") or {}
                if best_payload.get("reward") is not None:
                    st.success(
                        f"最佳结果：第 {bandit_state.get('best_index')} 轮，奖励 {best_payload['reward']:+.4f}"
                    )
                    col_best1, col_best2 = st.columns(2)
                    col_best1.write("动作(raw)：")
                    col_best1.json(best_payload.get("action") or {})
                    col_best2.write("参数值：")
                    col_best2.json(best_payload.get("resolved_action") or {})
                    weights_payload = best_payload.get("weights") or {}
                    if weights_payload:
                        st.write("对应代理权重：")
                        st.json(weights_payload)
                        if st.button(
                            "将最佳权重写入默认配置",
                            key="save_decision_env_bandit_weights",
                        ):
                            try:
                                app_cfg.agent_weights.update_from_dict(weights_payload)
                                save_config(app_cfg)
                            except Exception as exc:  # noqa: BLE001
                                LOGGER.exception(
                                    "保存 bandit 权重失败",
                                    extra={**LOG_EXTRA, "error": str(exc)},
                                )
                                st.error(f"写入配置失败：{exc}")
                            else:
                                st.success("最佳权重已写入 config.json")
                    dept_ctrl = best_payload.get("department_controls") or {}
                    if dept_ctrl:
                        with st.expander("最佳部门控制参数", expanded=False):
                            st.json(dept_ctrl)
                if st.button("清除自动探索结果", key="clear_decision_env_bandit"):
                    st.session_state.pop(_DECISION_ENV_BANDIT_RESULTS_KEY, None)
                    st.success("已清除自动探索结果。")

            st.divider()
            st.subheader("PPO 训练（逐日强化学习）")

            if TORCH_AVAILABLE:
                col_ts, col_rollout, col_epochs = st.columns(3)
                ppo_timesteps = int(
                    col_ts.number_input(
                        "总时间步",
                        min_value=256,
                        max_value=200_000,
                        value=4096,
                        step=256,
                        key="decision_env_ppo_timesteps",
                    )
                )
                ppo_rollout = int(
                    col_rollout.number_input(
                        "每次收集步数",
                        min_value=32,
                        max_value=2048,
                        value=256,
                        step=32,
                        key="decision_env_ppo_rollout",
                    )
                )
                ppo_epochs = int(
                    col_epochs.number_input(
                        "每批优化轮数",
                        min_value=1,
                        max_value=30,
                        value=8,
                        step=1,
                        key="decision_env_ppo_epochs",
                    )
                )

                col_mb, col_gamma, col_lambda = st.columns(3)
                ppo_minibatch = int(
                    col_mb.number_input(
                        "最小批次规模",
                        min_value=16,
                        max_value=1024,
                        value=128,
                        step=16,
                        key="decision_env_ppo_minibatch",
                    )
                )
                ppo_gamma = float(
                    col_gamma.number_input(
                        "折现系数 γ",
                        min_value=0.5,
                        max_value=0.999,
                        value=0.99,
                        step=0.01,
                        format="%.3f",
                        key="decision_env_ppo_gamma",
                    )
                )
                ppo_lambda = float(
                    col_lambda.number_input(
                        "GAE λ",
                        min_value=0.5,
                        max_value=0.999,
                        value=0.95,
                        step=0.01,
                        format="%.3f",
                        key="decision_env_ppo_lambda",
                    )
                )

                col_clip, col_entropy, col_value = st.columns(3)
                ppo_clip = float(
                    col_clip.number_input(
                        "裁剪范围 ε",
                        min_value=0.05,
                        max_value=0.5,
                        value=0.2,
                        step=0.01,
                        format="%.2f",
                        key="decision_env_ppo_clip",
                    )
                )
                ppo_entropy = float(
                    col_entropy.number_input(
                        "熵系数",
                        min_value=0.0,
                        max_value=0.1,
                        value=0.01,
                        step=0.005,
                        format="%.3f",
                        key="decision_env_ppo_entropy",
                    )
                )
                ppo_value_coef = float(
                    col_value.number_input(
                        "价值损失系数",
                        min_value=0.0,
                        max_value=2.0,
                        value=0.5,
                        step=0.1,
                        format="%.2f",
                        key="decision_env_ppo_value_coef",
                    )
                )

                col_lr_p, col_lr_v, col_grad = st.columns(3)
                ppo_policy_lr = float(
                    col_lr_p.number_input(
                        "策略学习率",
                        min_value=1e-5,
                        max_value=1e-2,
                        value=3e-4,
                        step=1e-5,
                        format="%.5f",
                        key="decision_env_ppo_policy_lr",
                    )
                )
                ppo_value_lr = float(
                    col_lr_v.number_input(
                        "价值学习率",
                        min_value=1e-5,
                        max_value=1e-2,
                        value=3e-4,
                        step=1e-5,
                        format="%.5f",
                        key="decision_env_ppo_value_lr",
                    )
                )
                ppo_max_grad_norm = float(
                    col_grad.number_input(
                        "梯度裁剪", value=0.5, min_value=0.0, max_value=5.0, step=0.1, format="%.1f",
                        key="decision_env_ppo_grad_norm",
                    )
                )

                col_hidden, col_seed, _ = st.columns(3)
                ppo_hidden_text = col_hidden.text_input(
                    "隐藏层结构 (逗号分隔)", value="128,128", key="decision_env_ppo_hidden"
                )
                ppo_seed_text = col_seed.text_input(
                    "随机种子 (可选)", value="42", key="decision_env_ppo_seed"
                )
                try:
                    ppo_hidden = tuple(int(v.strip()) for v in ppo_hidden_text.split(",") if v.strip())
                except ValueError:
                    ppo_hidden = ()
                ppo_seed = None
                if ppo_seed_text.strip():
                    try:
                        ppo_seed = int(ppo_seed_text.strip())
                    except ValueError:
                        st.warning("PPO 随机种子需为整数，已忽略该值。")
                        ppo_seed = None

                if st.button("启动 PPO 训练", key="run_decision_env_ppo"):
                    if not specs:
                        st.warning("请先配置可调节参数，以构建动作空间。")
                    elif not ppo_hidden:
                        st.error("请提供合法的隐藏层结构，例如 128,128。")
                    else:
                        baseline_weights = app_cfg.agent_weights.as_dict()
                        for agent in agent_objects:
                            baseline_weights.setdefault(agent.name, 1.0)

                        universe_env = [code.strip() for code in universe_text.split(',') if code.strip()]
                        if not universe_env:
                            st.error("请先指定至少一个股票代码。")
                        else:
                            bt_cfg_env = BtConfig(
                                id="decision_env_ppo",
                                name="DecisionEnv PPO",
                                start_date=start_date,
                                end_date=end_date,
                                universe=universe_env,
                                params=dict(backtest_params),
                                method=app_cfg.decision_method,
                                game_structures=selected_structures,
                            )
                            env = DecisionEnv(
                                bt_config=bt_cfg_env,
                                parameter_specs=specs,
                                baseline_weights=baseline_weights,
                                disable_departments=disable_departments,
                            )
                            adapter = DecisionEnvAdapter(env)
                            config = PPOConfig(
                                total_timesteps=ppo_timesteps,
                                rollout_steps=ppo_rollout,
                                gamma=ppo_gamma,
                                gae_lambda=ppo_lambda,
                                clip_range=ppo_clip,
                                policy_lr=ppo_policy_lr,
                                value_lr=ppo_value_lr,
                                epochs=ppo_epochs,
                                minibatch_size=ppo_minibatch,
                                entropy_coef=ppo_entropy,
                                value_coef=ppo_value_coef,
                                max_grad_norm=ppo_max_grad_norm,
                                hidden_sizes=ppo_hidden,
                                seed=ppo_seed,
                            )
                            with st.spinner("PPO 训练进行中，请耐心等待..."):
                                try:
                                    summary = train_ppo(adapter, config)
                                except Exception as exc:  # noqa: BLE001
                                    LOGGER.exception(
                                        "PPO 训练失败",
                                        extra={**LOG_EXTRA, "error": str(exc)},
                                    )
                                    st.error(f"PPO 训练失败：{exc}")
                                else:
                                    payload = {
                                        "timesteps": summary.timesteps,
                                        "episode_rewards": summary.episode_rewards,
                                        "episode_lengths": summary.episode_lengths,
                                        "diagnostics": summary.diagnostics[-25:],
                                        "observation_keys": adapter.keys(),
                                    }
                                    st.session_state[_DECISION_ENV_PPO_RESULTS_KEY] = payload
                                    st.success("PPO 训练完成。")

            ppo_state = st.session_state.get(_DECISION_ENV_PPO_RESULTS_KEY)
            if ppo_state:
                st.caption(
                    f"最近一次 PPO 训练时间步：{ppo_state.get('timesteps')}"
                )
                rewards = ppo_state.get("episode_rewards") or []
                if rewards:
                    st.line_chart(rewards, height=200)
                lengths = ppo_state.get("episode_lengths") or []
                if lengths:
                    st.bar_chart(lengths, height=200)
                diagnostics = ppo_state.get("diagnostics") or []
                if diagnostics:
                    st.dataframe(pd.DataFrame(diagnostics), hide_index=True, width='stretch')
                st.download_button(
                    "下载 PPO 结果 (JSON)",
                    data=json.dumps(ppo_state, ensure_ascii=False, indent=2),
                    file_name="ppo_training_summary.json",
                    mime="application/json",
                    key="decision_env_ppo_json",
                )
                if st.button("清除 PPO 训练结果", key="clear_decision_env_ppo"):
                    st.session_state.pop(_DECISION_ENV_PPO_RESULTS_KEY, None)
                    st.success("已清除 PPO 训练结果。")
