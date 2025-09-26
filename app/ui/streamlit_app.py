"""Streamlit UI scaffold for the investment assistant."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.data.schema import initialize_database
from app.ingest.tushare import FetchJob, run_ingestion
from app.llm.explain import make_human_card


def render_today_plan() -> None:
    st.header("今日计划")
    st.write("待接入候选池筛选与多智能体决策结果。")
    sample = make_human_card("000001.SZ", "2025-01-01", {"decisions": []})
    st.json(sample)


def render_backtest() -> None:
    st.header("回测与复盘")
    st.write("在此运行回测、展示净值曲线与代理贡献。")
    st.button("开始回测")


def render_settings() -> None:
    st.header("数据与设置")
    st.text_input("TuShare Token")
    st.write("新闻源开关与数据库备份将在此配置。")


def render_tests() -> None:
    st.header("自检测试")
    st.write("用于快速检查数据库与数据拉取是否正常工作。")

    if st.button("测试数据库初始化"):
        with st.spinner("正在检查数据库..."):
            result = initialize_database()
            if result.skipped:
                st.success("数据库已存在，检查通过。")
            else:
                st.success(f"数据库初始化完成，共执行 {result.executed} 条语句。")

    st.divider()

    if st.button("测试 TuShare 拉取（示例 2024-01-01 至 2024-01-03）"):
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
                st.success("TuShare 示例拉取完成，数据已写入数据库。")
            except Exception as exc:  # noqa: BLE001
                st.error(f"拉取失败：{exc}")

    st.info("注意：TuShare 拉取依赖网络与 Token，若环境未配置将出现错误提示。")


def main() -> None:
    st.set_page_config(page_title="多智能体投资助理", layout="wide")
    tabs = st.tabs(["今日计划", "回测与复盘", "数据与设置", "自检测试"])
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
