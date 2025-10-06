"""日志钻取视图。"""
from __future__ import annotations

from datetime import date, timedelta
import pandas as pd
import streamlit as st

from app.utils.db import db_session

from app.ui.shared import LOGGER, LOG_EXTRA

def render_log_viewer() -> None:
    """渲染日志钻取与历史对比视图页面。"""
    LOGGER.info("渲染日志视图页面", extra=LOG_EXTRA)
    st.header("日志钻取与历史对比")
    st.write("查看系统运行日志，支持时间范围筛选、关键词搜索和历史对比功能。")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=date.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("结束日期", value=date.today())

    log_levels = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    selected_level = st.selectbox("日志级别", log_levels, index=1)

    search_query = st.text_input("搜索关键词")

    with db_session(read_only=True) as conn:
        stages = [row["stage"] for row in conn.execute("SELECT DISTINCT stage FROM run_log").fetchall()]
    stages = [s for s in stages if s]
    stages.insert(0, "ALL")
    selected_stage = st.selectbox("执行阶段", stages)

    with st.spinner("加载日志数据中..."):
        try:
            with db_session(read_only=True) as conn:
                query_parts = ["SELECT ts, stage, level, msg FROM run_log WHERE 1=1"]
                params: list[object] = []

                start_ts = f"{start_date.isoformat()}T00:00:00Z"
                end_ts = f"{end_date.isoformat()}T23:59:59Z"
                query_parts.append("AND ts BETWEEN ? AND ?")
                params.extend([start_ts, end_ts])

                if selected_level != "ALL":
                    query_parts.append("AND level = ?")
                    params.append(selected_level)

                if search_query:
                    query_parts.append("AND msg LIKE ?")
                    params.append(f"%{search_query}%")

                if selected_stage != "ALL":
                    query_parts.append("AND stage = ?")
                    params.append(selected_stage)

                query_parts.append("ORDER BY ts DESC")

                query = " ".join(query_parts)
                rows = conn.execute(query, params).fetchall()

                if rows:
                    rows_dict = [{key: row[key] for key in row.keys()} for row in rows]
                    log_df = pd.DataFrame(rows_dict)
                    log_df["ts"] = pd.to_datetime(log_df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    for col in log_df.columns:
                        log_df[col] = log_df[col].astype(str)
                else:
                    log_df = pd.DataFrame(columns=["ts", "stage", "level", "msg"])

                st.dataframe(
                    log_df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "ts": st.column_config.TextColumn("时间"),
                        "stage": st.column_config.TextColumn("执行阶段"),
                        "level": st.column_config.TextColumn("日志级别"),
                        "msg": st.column_config.TextColumn("日志消息", width="large"),
                    },
                )

                if not log_df.empty:
                    csv_data = log_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="下载日志CSV",
                        data=csv_data,
                        file_name=f"logs_{start_date}_{end_date}.csv",
                        mime="text/csv",
                        key="download_logs",
                    )

                    json_data = log_df.to_json(orient="records", force_ascii=False, indent=2)
                    st.download_button(
                        label="下载日志JSON",
                        data=json_data,
                        file_name=f"logs_{start_date}_{end_date}.json",
                        mime="application/json",
                        key="download_logs_json",
                    )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("加载日志失败", extra=LOG_EXTRA)
            st.error(f"加载日志数据失败：{exc}")

    st.subheader("历史对比")
    st.write("选择两个时间点的日志进行对比分析。")

    col3, col4 = st.columns(2)
    with col3:
        compare_date1 = st.date_input("对比日期1", value=date.today() - timedelta(days=1))
    with col4:
        compare_date2 = st.date_input("对比日期2", value=date.today())

    comparison_stage = st.selectbox("对比阶段", stages, key="compare_stage")
    st.write("选择需要比较的日志数量。")
    compare_limit = st.slider("对比日志数量", min_value=10, max_value=200, value=50, step=10)

    if st.button("生成历史对比报告"):
        with st.spinner("生成对比报告中..."):
            try:
                with db_session(read_only=True) as conn:
                    def load_logs(d: date) -> pd.DataFrame:
                        start_ts = f"{d.isoformat()}T00:00:00Z"
                        end_ts = f"{d.isoformat()}T23:59:59Z"
                        query = ["SELECT ts, level, msg FROM run_log WHERE ts BETWEEN ? AND ?"]
                        params: list[object] = [start_ts, end_ts]
                        if comparison_stage != "ALL":
                            query.append("AND stage = ?")
                            params.append(comparison_stage)
                        query.append("ORDER BY ts DESC LIMIT ?")
                        params.append(compare_limit)
                        sql = " ".join(query)
                        rows = conn.execute(sql, params).fetchall()
                        if not rows:
                            return pd.DataFrame(columns=["ts", "level", "msg"])
                        df = pd.DataFrame([{k: row[k] for k in row.keys()} for row in rows])
                        df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                        return df

                    df1 = load_logs(compare_date1)
                    df2 = load_logs(compare_date2)

                if df1.empty and df2.empty:
                    st.info("选定日期暂无日志可对比。")
                else:
                    st.write("### 对比结果")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"{compare_date1} 日日志")
                        st.dataframe(df1, hide_index=True, width="stretch")
                    with col_b:
                        st.write(f"{compare_date2} 日日志")
                        st.dataframe(df2, hide_index=True, width="stretch")

                    summary = {
                        "日期1日志条数": int(len(df1)),
                        "日期2日志条数": int(len(df2)),
                        "新增日志条数": max(len(df2) - len(df1), 0),
                    }
                    st.write("摘要：", summary)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("历史对比生成失败", extra=LOG_EXTRA)
                st.error(f"生成历史对比失败：{exc}")
