"""投资池与仓位概览页面。"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils.db import db_session
from app.utils.portfolio import (
    get_latest_snapshot,
    list_investment_pool,
    list_positions,
    list_recent_trades,
)

from app.ui.shared import LOGGER, LOG_EXTRA, get_latest_trade_date


def render_pool_overview() -> None:
    """单独的投资池与仓位概览页面（从今日计划中提取）。"""
    LOGGER.info("渲染投资池与仓位概览页面", extra=LOG_EXTRA)
    st.header("投资池与仓位概览")

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

    try:
        latest_date = get_latest_trade_date()
        candidates = list_investment_pool(trade_date=latest_date)
    except Exception:  # noqa: BLE001
        LOGGER.exception("加载候选池失败", extra=LOG_EXTRA)
        candidates = []

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

    if st.button("执行对比", type="secondary"):
        with st.spinner("执行日志对比分析中..."):
            try:
                with db_session(read_only=True) as conn:
                    query_date1 = f"{compare_date1.isoformat()}T00:00:00Z"  # type: ignore[name-defined]
                    query_date2 = f"{compare_date1.isoformat()}T23:59:59Z"  # type: ignore[name-defined]
                    logs1 = conn.execute(
                        "SELECT level, COUNT(*) as count FROM run_log WHERE ts BETWEEN ? AND ? GROUP BY level",
                        (query_date1, query_date2),
                    ).fetchall()

                    query_date3 = f"{compare_date2.isoformat()}T00:00:00Z"  # type: ignore[name-defined]
                    query_date4 = f"{compare_date2.isoformat()}T23:59:59Z"  # type: ignore[name-defined]
                    logs2 = conn.execute(
                        "SELECT level, COUNT(*) as count FROM run_log WHERE ts BETWEEN ? AND ? GROUP BY level",
                        (query_date3, query_date4),
                    ).fetchall()

                    df1 = pd.DataFrame(logs1, columns=["level", "count"])
                    df1["date"] = compare_date1.strftime("%Y-%m-%d")  # type: ignore[name-defined]
                    df2 = pd.DataFrame(logs2, columns=["level", "count"])
                    df2["date"] = compare_date2.strftime("%Y-%m-%d")  # type: ignore[name-defined]

                    for df in (df1, df2):
                        for col in df.columns:
                            if col != "level":
                                df[col] = df[col].astype(object)

                    compare_df = pd.concat([df1, df2])
                    fig = px.bar(
                        compare_df,
                        x="level",
                        y="count",
                        color="date",
                        barmode="group",
                        title=f"日志级别分布对比 ({compare_date1} vs {compare_date2})",  # type: ignore[name-defined]
                    )
                    st.plotly_chart(fig, width='stretch')

                    st.write("日志统计对比：")
                    date1_str = compare_date1.strftime("%Y%m%d")  # type: ignore[name-defined]
                    date2_str = compare_date2.strftime("%Y%m%d")  # type: ignore[name-defined]
                    merged_df = df1.merge(
                        df2,
                        on="level",
                        suffixes=(f"_{date1_str}", f"_{date2_str}"),
                        how="outer",
                    ).fillna(0)
                    st.dataframe(merged_df, hide_index=True, width="stretch")
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("日志对比失败", extra=LOG_EXTRA)
                st.error(f"日志对比分析失败：{exc}")

    return
