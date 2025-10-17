"""今日计划页面视图。"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from app.backtest.engine import BacktestEngine, PortfolioState, BtConfig
from app.utils.portfolio import (
    InvestmentCandidate,
    PortfolioPosition,
    get_portfolio_settings_snapshot,
    list_investment_pool,
    list_positions,
)
from app.utils.db import db_session

from app.ui.shared import (
    LOGGER,
    LOG_EXTRA,
    get_latest_trade_date,
    get_query_params,
    set_query_params,
)


def _parse_trade_date(trade_date: str | int | date) -> date:
    """Parse trade date inputs that may come in multiple string formats."""
    if isinstance(trade_date, date):
        return trade_date
    value = str(trade_date)
    try:
        return date.fromisoformat(value)
    except ValueError:
        pass
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"无法解析交易日：{trade_date}") from exc


def _resolve_trade_date_variants(trade_date: str | int | date) -> List[str]:
    """Return possible string representations for the given trade date."""
    variants: List[str] = []
    str_val = str(trade_date).strip()
    if str_val:
        variants.append(str_val)
    try:
        dt = _parse_trade_date(trade_date)
    except ValueError:
        dt = None
    if dt is not None:
        variants.extend(
            [
                dt.strftime("%Y%m%d"),
                dt.isoformat(),
            ]
        )
    # 保持顺序同时去重
    seen = set()
    unique_variants: List[str] = []
    for item in variants:
        if item and item not in seen:
            unique_variants.append(item)
            seen.add(item)
    return unique_variants


def _load_candidate_records(
    trade_date: str | int | date,
) -> tuple[List[InvestmentCandidate], Optional[str], bool]:
    """Load candidate pool records trying multiple trade date representations."""
    variants = _resolve_trade_date_variants(trade_date)
    for variant in variants:
        records = list_investment_pool(trade_date=variant)
        if records:
            return records, variant, False
    latest_records = list_investment_pool()
    if latest_records:
        return latest_records, latest_records[0].trade_date, True
    return [], None, False


def _fetch_agent_actions(trade_date: str, symbols: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
    unique_symbols = list(dict.fromkeys(symbols))
    if not unique_symbols:
        return {}
    placeholder = ", ".join("?" for _ in unique_symbols)
    sql = (
        f"""
        SELECT ts_code, agent, action
        FROM agent_utils
        WHERE trade_date = ?
        AND ts_code IN ({placeholder})
        """
    )
    params: List[object] = [trade_date, *unique_symbols]
    with db_session(read_only=True) as conn:
        rows = conn.execute(sql, params).fetchall()
    mapping: Dict[str, Dict[str, Optional[str]]] = {}
    for row in rows:
        ts_code = row["ts_code"]
        agent = row["agent"]
        action = row["action"]
        mapping.setdefault(ts_code, {})[agent] = action
    return mapping


@dataclass
class ReevaluationOutcome:
    changes: List[Dict[str, object]]
    summary: List[Dict[str, object]]
    monitor_only: bool = False


def _summarize_decisions(
    records: List[tuple[str, object, object]],
    engine: BacktestEngine,
    expected_symbols: List[str],
) -> tuple[Dict[str, Dict[str, Optional[str]]], List[Dict[str, object]]]:
    agent_names = [agent.name for agent in getattr(engine, "agents", [])]
    after_map: Dict[str, Dict[str, Optional[str]]] = {}
    summary_rows: List[Dict[str, object]] = []
    seen_codes: set[str] = set()

    for ts_code, _context, decision in records:
        seen_codes.add(ts_code)
        agent_actions: Dict[str, Optional[str]] = {}
        final_action = getattr(decision, "action", None)
        agent_actions["global"] = getattr(final_action, "value", final_action)

        department_map = getattr(decision, "department_decisions", {}) or {}
        for dept_code, dept_decision in department_map.items():
            agent_actions[f"dept_{dept_code}"] = getattr(dept_decision.action, "value", dept_decision.action)

        utilities = getattr(decision, "utilities", {}) or {}
        for agent_name in agent_names:
            best_action: Optional[str] = None
            best_score: Optional[float] = None
            for action, agent_scores in utilities.items():
                if not isinstance(agent_scores, dict):
                    continue
                score = agent_scores.get(agent_name)
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = getattr(action, "value", action)
            if best_action is not None:
                agent_actions[agent_name] = best_action

        after_map[ts_code] = agent_actions
        target_weight = getattr(decision, "target_weight", None)
        summary_rows.append(
            {
                "代码": ts_code,
                "最终动作": getattr(final_action, "value", final_action),
                "信心": float(getattr(decision, "confidence", 0.0) or 0.0),
                "目标权重": float(target_weight) if target_weight is not None else None,
                "需复核": bool(getattr(decision, "requires_review", False)),
                "备注": "",
            }
        )

    expected_unique = list(dict.fromkeys(expected_symbols))
    for code in expected_unique:
        if code not in seen_codes:
            summary_rows.append(
                {
                    "代码": code,
                    "最终动作": None,
                    "信心": None,
                    "目标权重": None,
                    "需复核": None,
                    "备注": "未返回决策（可能缺少数据或行情）",
                }
            )

    return after_map, summary_rows


def _render_summary_table(rows: List[Dict[str, object]], *, title: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if df.empty:
        return
    formatted = df.copy()
    if "信心" in formatted.columns:
        formatted["信心"] = formatted["信心"].apply(
            lambda x: f"{x:.2f}" if x is not None else "-"
        )
    if "目标权重" in formatted.columns:
        formatted["目标权重"] = formatted["目标权重"].apply(
            lambda x: f"{x:+.2%}" if x is not None else "-"
        )
    if "需复核" in formatted.columns:
        formatted["需复核"] = formatted["需复核"].apply(
            lambda x: "是" if x is True else ("否" if x is False else "-")
        )
    if "备注" in formatted.columns:
        formatted["备注"] = formatted["备注"].apply(lambda x: x if x else "")
    st.write(title)
    st.dataframe(formatted, hide_index=True, width="stretch")


def _reevaluate_symbols(
    trade_date_obj: date,
    symbols: List[str],
    cfg_id: str,
    cfg_name: str,
    *,
    monitor_only: bool = False,
) -> ReevaluationOutcome:
    unique_symbols = list(dict.fromkeys(symbols))
    if not unique_symbols:
        return ReevaluationOutcome(changes=[], summary=[], monitor_only=monitor_only)
    trade_date_str = trade_date_obj.isoformat()
    before_map = _fetch_agent_actions(trade_date_str, unique_symbols)
    engine_params: Dict[str, object] = {}
    target_val = st.session_state.get("bt_target")
    if target_val is not None:
        try:
            engine_params["target"] = float(target_val)
        except (TypeError, ValueError):
            pass
    stop_val = st.session_state.get("bt_stop")
    if stop_val is not None:
        try:
            engine_params["stop"] = float(stop_val)
        except (TypeError, ValueError):
            pass
    hold_val = st.session_state.get("bt_hold_days")
    if hold_val is not None:
        try:
            engine_params["hold_days"] = int(hold_val)
        except (TypeError, ValueError):
            pass
    capital_val = st.session_state.get("bt_initial_capital")
    if capital_val is not None:
        try:
            engine_params["initial_capital"] = float(capital_val)
        except (TypeError, ValueError):
            pass
    cfg = BtConfig(
        id=cfg_id,
        name=cfg_name,
        start_date=trade_date_obj,
        end_date=trade_date_obj,
        universe=unique_symbols,
        params=engine_params,
    )
    engine = BacktestEngine(cfg, persist_results=not monitor_only)
    state = PortfolioState(cash=engine.initial_cash)
    records = engine.simulate_day(trade_date_obj, state)
    after_map, summary_rows = _summarize_decisions(records, engine, unique_symbols)
    if monitor_only:
        for row in summary_rows:
            if not row.get("备注"):
                row["备注"] = "监控模式预览"
    symbol_order = list(
        dict.fromkeys(unique_symbols + list(before_map.keys()) + list(after_map.keys()))
    )
    changes: List[Dict[str, object]] = []
    for code in symbol_order:
        before_agents = before_map.get(code, {})
        after_agents = after_map.get(code, {})
        agents = set(before_agents.keys()) | set(after_agents.keys())
        for agent in sorted(agents):
            old_action = before_agents.get(agent)
            new_action = after_agents.get(agent)
            if old_action != new_action:
                changes.append(
                    {
                        "代码": code,
                        "代理": agent,
                        "原动作": old_action,
                        "新动作": new_action,
                    }
                )
    return ReevaluationOutcome(changes=changes, summary=summary_rows, monitor_only=monitor_only)


def render_today_plan() -> None:
    LOGGER.info("渲染今日计划页面", extra=LOG_EXTRA)
    st.header("今日计划")
    latest_trade_date = get_latest_trade_date()
    if latest_trade_date:
        st.caption(f"最新交易日：{latest_trade_date.isoformat()}（统计数据请见左侧系统监控）")
    else:
        st.caption("统计与决策概览现已移至左侧'系统监控'侧栏。")

    portfolio_snapshot = get_portfolio_settings_snapshot()
    limits = portfolio_snapshot.get("position_limits", {})
    st.caption(
        "组合约束：单仓上限 {max_pos:.0%} ｜ 最小仓位 {min_pos:.0%} ｜ 最大持仓数 {max_cnt} ｜ 行业上限 {sector:.0%}".format(
            max_pos=limits.get("max_position", 0.2),
            min_pos=limits.get("min_position", 0.02),
            max_cnt=int(limits.get("max_total_positions", 20)),
            sector=limits.get("max_sector_exposure", 0.35),
        )
    )
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

    query = get_query_params()
    # 确保默认日期格式正确
    default_trade_date = str(query.get("date", [trade_dates[0]])[0])
    try:
        default_idx = trade_dates.index(default_trade_date)
    except ValueError:
        default_idx = 0
    # 确保日期格式统一为 YYYYMMDD
    formatted_trade_dates = [str(td) for td in trade_dates]
    selector_col, actions_col = st.columns([3, 2])
    with selector_col:
        trade_date = st.selectbox("交易日", formatted_trade_dates, index=default_idx)

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
    agent_symbols = [row["ts_code"] for row in code_rows]

    candidate_records, candidate_data_date, candidate_fallback = _load_candidate_records(trade_date)
    if candidate_records:
        preview_codes = "、".join(item.ts_code for item in candidate_records[:12])
        if len(candidate_records) > 12:
            preview_codes += "…"
        source_note = ""
        if candidate_data_date and candidate_data_date != trade_date:
            source_note = f"（基于 {candidate_data_date} 数据）"
        elif candidate_fallback:
            source_note = "（使用最近候选池）"
        st.caption(
            f"候选池包含 {len(candidate_records)} 个标的：{preview_codes}{source_note}"
        )
    else:
        st.caption("所选日期暂无候选池数据，仍可查看代理决策记录或触发重评估。")

    candidate_codes = [item.ts_code for item in candidate_records]
    positions = list_positions(active_only=True)
    position_map: Dict[str, PortfolioPosition] = {pos.ts_code: pos for pos in positions}
    position_codes = [pos.ts_code for pos in positions]

    symbols = list(
        dict.fromkeys(candidate_codes + agent_symbols + position_codes)
    )

    with actions_col:
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("已产出决策数", len(agent_symbols))
        metrics_cols[1].metric("候选池标的", len(candidate_records))
        metrics_cols[2].metric("当前持仓", len(position_codes))
        monitor_only = st.checkbox(
            "仅监控模式（不写入数据库）",
            key="reeval_monitor_only",
        )
        if monitor_only:
            st.caption("监控模式：仅生成策略预览，不更新 agent_utils 或候选池记录。")
        else:
            st.caption("对当前交易日所有标的触发策略重评估。")
        if st.button("一键重评估全部", type="primary", width="stretch"):
            with st.spinner("正在对所有标的进行重评估，请稍候..."):
                try:
                    trade_date_obj = _parse_trade_date(trade_date)
                    progress = st.progress(0.0)
                    progress.progress(0.3 if symbols else 1.0)
                    outcome = _reevaluate_symbols(
                        trade_date_obj,
                        symbols,
                        "reeval_ui_all",
                        "UI All Re-eval",
                        monitor_only=monitor_only,
                    )
                    progress.progress(1.0)
                    st.success(f"一键重评估完成：共处理 {len(symbols)} 个标的")
                    if outcome.changes:
                        st.write("检测到以下动作变更：")
                        st.dataframe(pd.DataFrame(outcome.changes), hide_index=True, width='stretch')
                    if monitor_only:
                        _render_summary_table(
                            outcome.summary,
                            title="最新策略摘要（监控模式预览）",
                        )
                    else:
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("一键重评估失败", extra=LOG_EXTRA)
                    st.error(f"一键重评估执行过程中发生错误：{exc}")

    overview_tab, detail_tab = st.tabs(["组合总览", "标的详情"])
    with overview_tab:
        _render_today_plan_assistant_view(
            trade_date,
            candidate_records,
            candidate_data_date,
            candidate_fallback,
            positions,
        )

    with detail_tab:
        if not symbols:
            st.info("所选交易日暂无 agent_utils 记录。")
        else:
            _render_today_plan_symbol_view(
                trade_date,
                symbols,
                query,
                candidate_records,
                position_map,
            )


def _render_today_plan_assistant_view(
    trade_date: str | int | date,
    candidate_records: List[InvestmentCandidate],
    candidate_data_date: Optional[str],
    fallback_used: bool,
    positions: List[PortfolioPosition],
) -> None:
    display_date = candidate_data_date or str(trade_date)
    st.info("组合总览聚焦候选池与仓位概况，用于快速判断今日是否需要调整策略。")
    if fallback_used and candidate_data_date and candidate_data_date != str(trade_date):
        st.caption(f"候选池数据来自 {candidate_data_date}（最近可用日期）。")
    elif candidate_data_date and candidate_data_date != str(trade_date):
        st.caption(f"候选池数据基于 {candidate_data_date}。")

    candidates = candidate_records
    if not candidates:
        st.warning("尚未生成候选池，请先运行回测/评估流程或触发上方的一键重评估。")
        return

    scores = [float(item.score or 0.0) for item in candidates]
    statuses = [item.status or "UNKNOWN" for item in candidates]
    tags: List[str] = []
    rationales: List[str] = []
    for item in candidates:
        if getattr(item, "tags", None):
            tags.extend(item.tags)
        if getattr(item, "rationale", None):
            rationales.append(str(item.rationale))

    cnt = Counter(statuses)
    tag_cnt = Counter(tags)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("候选数", f"{len(candidates)}")
    col_b.metric("平均评分", f"{np.mean(scores):.3f}" if scores else "-")
    col_c.metric("中位评分", f"{np.median(scores):.3f}" if scores else "-")

    if positions:
        total_market_value = sum(float(pos.market_value or 0.0) for pos in positions)
        position_cols = st.columns(3)
        position_cols[0].metric("持仓数", len(positions))
        position_cols[1].metric(
            "持仓总市值",
            f"{total_market_value:,.0f}" if total_market_value else "-",
        )
        exposure = sum(float(pos.target_weight or 0.0) for pos in positions)
        position_cols[2].metric("目标权重合计", f"{exposure:.0%}" if exposure else "-")

    st.write("状态分布：")
    st.json(dict(cnt))

    if tag_cnt:
        st.write("常见标签（示例）：")
        st.json(dict(tag_cnt.most_common(10)))

    if rationales:
        st.write("汇总理由（节选，不含代码）：")
        seen = set()
        excerpts = []
        for rationale in rationales:
            text = rationale.strip()
            if text and text not in seen:
                seen.add(text)
                excerpts.append(text)
            if len(excerpts) >= 3:
                break
        for idx, excerpt in enumerate(excerpts, start=1):
            st.markdown(f"**理由 {idx}:** {excerpt}")

    avg_score = float(np.mean(scores)) if scores else 0.0
    suggest_pct = max(0.0, min(0.3, 0.10 + (avg_score - 0.5) * 0.2))
    st.subheader("组合级建议（不指定标的）")
    st.write(
        f"基于候选池平均评分 {avg_score:.3f}，建议今日用于新增买入的现金比例约为 {suggest_pct:.0%}。"
    )
    st.write(
        "建议分配思路：在候选池中挑选若干得分较高的标的按目标权重等比例分配，或以分批买入的方式分摊入场时点。"
    )
    if st.button("生成组合级操作建议（仅输出，不执行）"):
        st.success("已生成组合级建议（仅供参考）。")
        st.write(
            {
                "候选数": len(candidates),
                "平均评分": avg_score,
                "建议新增买入比例": f"{suggest_pct:.0%}",
                "候选数据日期": display_date,
            }
        )


def _render_today_plan_symbol_view(
    trade_date: str | int | date,
    symbols: List[str],
    query_params: Dict[str, List[str]],
    candidate_records: List[InvestmentCandidate],
    position_map: Dict[str, PortfolioPosition],
) -> None:
    default_ts = query_params.get("code", [symbols[0]])[0]
    try:
        default_ts_idx = symbols.index(default_ts)
    except ValueError:
        default_ts_idx = 0
    ts_code = st.selectbox("标的", symbols, index=default_ts_idx)
    candidate_code_set = {item.ts_code for item in candidate_records}
    default_batch = [code for code in symbols if code in candidate_code_set]
    batch_symbols = st.multiselect("批量重评估（可多选）", symbols, default=default_batch[:10])

    set_query_params(date=str(trade_date), code=str(ts_code))

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

    candidate_map = {item.ts_code: item for item in candidate_records}
    candidate_info = candidate_map.get(ts_code)
    position_info = position_map.get(ts_code)
    if candidate_info:
        info_cols = st.columns(4)
        info_cols[0].metric("候选评分", f"{(candidate_info.score or 0):.3f}")
        info_cols[1].metric("状态", candidate_info.status or "-")
        info_cols[2].metric("更新时间", candidate_info.created_at or "-")
        info_cols[3].metric("行业", candidate_info.industry or "-")
        if candidate_info.rationale:
            st.caption(f"候选理由：{candidate_info.rationale}")
    if position_info:
        pos_cols = st.columns(4)
        pos_cols[0].metric("持仓数量", f"{position_info.quantity:.0f}")
        pos_cols[1].metric(
            "成本价",
            f"{position_info.cost_price:.2f}",
        )
        market_price = position_info.market_price
        pos_cols[2].metric("市价", f"{market_price:.2f}" if market_price else "-")
        pnl = (position_info.unrealized_pnl or 0.0) + (position_info.realized_pnl or 0.0)
        pos_cols[3].metric("累计盈亏", f"{pnl:,.0f}")
        if position_info.target_weight is not None:
            st.caption(f"目标权重：{position_info.target_weight:.0%}")

    _render_today_plan_reevaluation_controls(
        trade_date,
        ts_code,
        batch_symbols,
        add_divider=False,
    )

    if not rows:
        st.info("该标的尚未生成部门/代理决策，可尝试触发重评估。")
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
                    import csv
                    import io

                    buf = io.StringIO()
                    writer = csv.writer(buf)
                    writer.writerow(["trade_date", "close"])
                    for dt_val, val in close_series:
                        writer.writerow([dt_val, val])
                    st.download_button(
                        "下载收盘价(CSV)",
                        data=buf.getvalue(),
                        file_name=f"{ts_code}_{trade_date}_close_series.csv",
                        mime="text/csv",
                        key="dl_close_csv",
                    )
                except Exception:  # noqa: BLE001
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
        keyword = st.text_input("筛选摘要/信号关键词", value="")
        filtered = dept_records
        if keyword.strip():
            kw = keyword.strip()
            filtered = [
                item
                for item in dept_records
                if kw in str(item.get("摘要", "")) or kw in str(item.get("核心信号", ""))
            ]
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
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            pass
    else:
        st.info("暂无基础代理评分。")

    st.divider()
    st.subheader("相关新闻")
    try:
        with db_session(read_only=True) as conn:
            try:
                trade_date_obj = _parse_trade_date(trade_date)
            except ValueError:
                trade_date_obj = date.today()

            news_query = """
                SELECT id, title, source, pub_time, sentiment, heat, entities
                FROM news
                WHERE ts_code = ? AND pub_time >= ?
                ORDER BY pub_time DESC
                LIMIT 10
            """
            seven_days_ago = (trade_date_obj - timedelta(days=7)).strftime("%Y-%m-%d")
            news_rows = conn.execute(news_query, (ts_code, seven_days_ago)).fetchall()

        if news_rows:
            news_data = []
            for row in news_rows:
                entities_info = {}
                try:
                    if row["entities"]:
                        entities_info = json.loads(row["entities"])
                except (json.JSONDecodeError, TypeError):
                    pass

                news_item = {
                    "标题": row["title"],
                    "来源": row["source"],
                    "发布时间": row["pub_time"],
                    "情感指数": f"{row['sentiment']:.2f}" if row["sentiment"] is not None else "-",
                    "热度评分": f"{row['heat']:.2f}" if row["heat"] is not None else "-",
                }

                industries = entities_info.get("industries", [])
                if industries:
                    news_item["相关行业"] = "、".join(industries[:3])

                news_data.append(news_item)

            news_df = pd.DataFrame(news_data)
            for col in news_df.columns:
                news_df[col] = news_df[col].astype(str)
            st.dataframe(news_df, width='stretch', hide_index=True)

            st.write("详细新闻内容：")
            for idx, row in enumerate(news_rows):
                with st.expander(f"{idx+1}. {row['title']}", expanded=False):
                    st.write(f"**来源：** {row['source']}")
                    st.write(f"**发布时间：** {row['pub_time']}")

                    entities_info = {}
                    try:
                        if row["entities"]:
                            entities_info = json.loads(row["entities"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                    sentiment_display = f"{row['sentiment']:.2f}" if row["sentiment"] is not None else "-"
                    heat_display = f"{row['heat']:.2f}" if row["heat"] is not None else "-"
                    st.write(f"**情感指数：** {sentiment_display} | **热度评分：** {heat_display}")

                    industries = entities_info.get("industries", [])
                    if industries:
                        st.write(f"**相关行业：** {'、'.join(industries)}")

                    important_keywords = entities_info.get("important_keywords", [])
                    if important_keywords:
                        st.write(f"**重要关键词：** {'、'.join(important_keywords)}")

                    url = entities_info.get("source_url", "")
                    if url:
                        st.markdown(f"[查看原文]({url})", unsafe_allow_html=True)
        else:
            st.info(f"近7天内暂无关于 {ts_code} 的新闻。")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("获取新闻数据失败", extra=LOG_EXTRA)
        st.error(f"获取新闻数据时发生错误：{exc}")

    st.divider()
    st.info("投资池与仓位概览已移至单独页面。请在侧边或页面导航中选择“投资池/仓位”以查看详细信息。")

def _render_today_plan_reevaluation_controls(
    trade_date: str | int | date,
    ts_code: str,
    batch_symbols: List[str],
    *,
    add_divider: bool = True,
) -> None:
    if add_divider:
        st.divider()
    st.subheader("策略重评估")
    monitor_only = bool(st.session_state.get("reeval_monitor_only", False))
    if monitor_only:
        st.caption("监控模式已开启：生成策略预览但不会写入数据库。")
    else:
        st.caption("对当前选中的交易日与标的，立即触发一次策略评估并回写 agent_utils。")
    cols_re = st.columns([1, 1])
    if cols_re[0].button("对该标的重评估", key=f"reevaluate_{ts_code}"):
        with st.spinner("正在重评估..."):
            try:
                trade_date_obj = _parse_trade_date(trade_date)
                outcome = _reevaluate_symbols(
                    trade_date_obj,
                    [ts_code],
                    "reeval_ui",
                    "UI Re-evaluation",
                    monitor_only=monitor_only,
                )
                changes = outcome.changes
                if changes:
                    for change in changes:
                        change.setdefault("代码", ts_code)
                    df_changes = pd.DataFrame(changes)
                    if "代码" in df_changes.columns:
                        df_changes = df_changes[["代码", "代理", "原动作", "新动作"]]
                    st.success("重评估完成，检测到动作变更：")
                    st.dataframe(df_changes, hide_index=True, width='stretch')
                else:
                    st.success("重评估完成，无动作变更。")
                if monitor_only:
                    _render_summary_table(
                        outcome.summary,
                        title="策略结果预览（监控模式）",
                    )
                else:
                    st.rerun()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("重评估失败", extra=LOG_EXTRA)
                st.error(f"重评估失败：{exc}")
    disabled = not batch_symbols
    if cols_re[1].button(
        "批量重评估（所选）",
        key="reevaluate_batch_symbols",
        disabled=disabled,
    ):
        with st.spinner("批量重评估执行中..."):
            try:
                trade_date_obj = _parse_trade_date(trade_date)
                progress = st.progress(0.0)
                progress.progress(0.3 if batch_symbols else 1.0)
                outcome = _reevaluate_symbols(
                    trade_date_obj,
                    batch_symbols,
                    "reeval_ui_batch",
                    "UI Batch Re-eval",
                    monitor_only=monitor_only,
                )
                progress.progress(1.0)
                st.success(f"批量重评估完成：共处理 {len(batch_symbols)} 个标的")
                if outcome.changes:
                    st.dataframe(pd.DataFrame(outcome.changes), hide_index=True, width='stretch')
                if monitor_only:
                    _render_summary_table(
                        outcome.summary,
                        title="批量策略结果预览（监控模式）",
                    )
                else:
                    st.rerun()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("批量重评估失败", extra=LOG_EXTRA)
                st.error(f"批量重评估失败：{exc}")
