"""行情可视化页面。"""
from __future__ import annotations

from datetime import date, datetime, timedelta

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


def _load_trade_date_range(ts_code: str) -> tuple[date | None, date | None]:
    """Fetch earliest and latest available trade dates for a stock."""

    with db_session(read_only=True) as conn:
        row = conn.execute(
            "SELECT MIN(trade_date) AS min_date, MAX(trade_date) AS max_date FROM daily WHERE ts_code = ?",
            (ts_code,),
        ).fetchone()
    if not row:
        return None, None

    min_raw = row["min_date"]
    max_raw = row["max_date"]
    if not min_raw or not max_raw:
        return None, None

    min_date = datetime.strptime(min_raw, "%Y%m%d").date()
    max_date = datetime.strptime(max_raw, "%Y%m%d").date()
    return min_date, max_date


def render_market_visualization() -> None:
    st.header("行情可视化")
    st.caption("按标的查看 K 线、成交量以及常用指标。")

    options = _load_stock_options()
    if not options:
        st.warning("暂未加载到可用的行情标的，请先执行数据同步。")
        return

    selection = st.selectbox("选择标的", options, index=0)
    ts_code = _parse_ts_code(selection)
    min_date, max_date = _load_trade_date_range(ts_code)
    if not max_date:
        st.info("所选标的暂无可视化数据，请先同步行情。")
        return

    default_end = max_date
    default_start = max(min_date, max_date - timedelta(days=180)) if min_date else max_date - timedelta(days=180)

    session = st.session_state
    last_ts_code = session.get("market_selected_ts_code")
    start_store_key = "market_start_date_value"
    end_store_key = "market_end_date_value"
    start_widget_key = "market_start_date_picker"
    end_widget_key = "market_end_date_picker"
    if last_ts_code != ts_code:
        session["market_selected_ts_code"] = ts_code
        session[start_store_key] = default_start
        session[end_store_key] = default_end
        session[start_widget_key] = default_start
        session[end_widget_key] = default_end
    else:
        start_state = session.get(start_store_key, default_start)
        end_state = session.get(end_store_key, default_end)
        if min_date:
            start_state = min(max(start_state, min_date), max_date)
            end_state = min(max(end_state, min_date), max_date)
        if start_state > end_state:
            start_state = max(min_date or start_state, max_date - timedelta(days=30))
            end_state = max_date
        session[start_store_key] = start_state
        session[end_store_key] = end_state
        if session.get(start_widget_key) != start_state:
            session[start_widget_key] = start_state
        if session.get(end_widget_key) != end_state:
            session[end_widget_key] = end_state

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "开始日期",
            value=session.get(start_store_key, default_start),
            key=start_widget_key,
            min_value=min_date,
            max_value=max_date,
        )
    with col2:
        end_date = st.date_input(
            "结束日期",
            value=session.get(end_store_key, default_end),
            key=end_widget_key,
            min_value=min_date,
            max_value=max_date,
        )

    if min_date:
        start_date = max(start_date, min_date)
        end_date = max(end_date, min_date)
    end_date = min(end_date, max_date)
    start_date = min(start_date, max_date)

    session[start_store_key] = start_date
    session[end_store_key] = end_date

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
