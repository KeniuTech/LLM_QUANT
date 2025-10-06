"""行情可视化页面。"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.db import db_session

from app.ui.shared import LOGGER, LOG_EXTRA


def _load_stock_options(limit: int = 500) -> list[str]:
    try:
        with db_session(read_only=True) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT ts_code
                FROM daily
                ORDER BY trade_date DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except Exception:  # noqa: BLE001
        LOGGER.exception("加载股票列表失败", extra=LOG_EXTRA)
        return []
    return [row["ts_code"] for row in rows]


def _parse_ts_code(selection: str) -> str:
    return selection.split(" ", 1)[0]


def _load_daily_frame(ts_code: str, start: date, end: date) -> pd.DataFrame:
    with db_session(read_only=True) as conn:
        df = pd.read_sql_query(
            """
            SELECT trade_date, open, high, low, close, vol, amount
            FROM daily
            WHERE ts_code = ? AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
            """,
            conn,
            params=(ts_code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")),
        )
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return df


def render_market_visualization() -> None:
    st.header("行情可视化")
    st.caption("按标的查看 K 线、成交量以及常用指标。")

    options = _load_stock_options()
    if not options:
        st.warning("暂未加载到可用的行情标的，请先执行数据同步。")
        return

    selection = st.selectbox("选择标的", options, index=0)
    ts_code = _parse_ts_code(selection)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=date.today() - timedelta(days=120))
    with col2:
        end_date = st.date_input("结束日期", value=date.today())

    if start_date > end_date:
        st.error("开始日期不能晚于结束日期。")
        return

    try:
        df = _load_daily_frame(ts_code, start_date, end_date)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("加载行情数据失败", extra=LOG_EXTRA)
        st.error(f"加载行情数据失败：{exc}")
        return

    if df.empty:
        st.info("所选区间内无行情数据。")
        return

    st.metric("最新收盘价", f"{df['close'].iloc[-1]:.2f}")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["trade_date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="K线",
            )
        ]
    )
    fig.update_layout(title=f"{ts_code} K线图", xaxis_title="日期", yaxis_title="价格")
    st.plotly_chart(fig, use_container_width=True)

    fig_vol = px.bar(df, x="trade_date", y="vol", title="成交量")
    st.plotly_chart(fig_vol, use_container_width=True)

    df_ma = df.copy()
    df_ma["MA5"] = df_ma["close"].rolling(window=5).mean()
    df_ma["MA20"] = df_ma["close"].rolling(window=20).mean()
    df_ma["MA60"] = df_ma["close"].rolling(window=60).mean()

    fig_ma = px.line(df_ma, x="trade_date", y=["close", "MA5", "MA20", "MA60"], title="均线对比")
    st.plotly_chart(fig_ma, use_container_width=True)

    st.dataframe(df, hide_index=True, width='stretch')
