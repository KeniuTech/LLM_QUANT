"""Streamlit UI scaffold for the investment assistant."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.schema import initialize_database
from app.ingest.checker import run_boot_check
from app.ingest.rss import ingest_configured_rss
from app.ui.portfolio_config import render_portfolio_config
from app.ui.shared import LOGGER, LOG_EXTRA
from app.ui.views import (
    render_backtest_review,
    render_config_overview,
    render_data_settings,
    render_global_dashboard,
    render_llm_settings,
    render_log_viewer,
    render_market_visualization,
    render_pool_overview,
    render_stock_evaluation,
    render_tests,
    render_today_plan,
)
from app.utils.config import get_config


def main() -> None:
    LOGGER.info("初始化 Streamlit UI", extra=LOG_EXTRA)
    st.set_page_config(page_title="多智能体个人投资助理", layout="wide")

    initialize_database()

    cfg = get_config()
    if cfg.auto_update_data:
        LOGGER.info("检测到自动更新数据选项已启用，开始执行数据拉取", extra=LOG_EXTRA)
        try:
            with st.spinner("正在自动更新数据..."):
                def progress_hook(message: str, progress: float) -> None:
                    st.write(f"📊 {message} ({progress:.1%})")

                report = run_boot_check(
                    days=30,
                    auto_fetch=True,
                    progress_hook=progress_hook,
                    force_refresh=False,
                )
                rss_count = ingest_configured_rss(hours_back=24, max_items_per_feed=50)
                LOGGER.info(
                    "自动数据更新完成：日线数据覆盖%s-%s，RSS新闻%s条",
                    report.start,
                    report.end,
                    rss_count,
                    extra=LOG_EXTRA,
                )
                st.success(f"✅ 自动数据更新完成：获取RSS新闻 {rss_count} 条")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("自动数据更新失败", extra=LOG_EXTRA)
            st.error(f"❌ 自动数据更新失败：{exc}")

    render_global_dashboard()

    tabs = st.tabs(["今日计划", "投资池/仓位", "回测与复盘", "行情可视化", "日志钻取", "数据与设置", "自检测试"])
    LOGGER.debug(
        "Tabs 初始化完成：%s",
        ["今日计划", "投资池/仓位", "回测与复盘", "行情可视化", "日志钻取", "数据与设置", "自检测试"],
        extra=LOG_EXTRA,
    )

    with tabs[0]:
        render_today_plan()
    with tabs[1]:
        render_pool_overview()
    with tabs[2]:
        backtest_tabs = st.tabs(["回测复盘", "股票评估"])
        with backtest_tabs[0]:
            render_backtest_review()
        with backtest_tabs[1]:
            render_stock_evaluation()
    with tabs[3]:
        render_market_visualization()
    with tabs[4]:
        render_log_viewer()
    with tabs[5]:
        st.header("系统设置")
        settings_tabs = st.tabs(["配置概览", "LLM 设置", "投资组合", "数据源"])
        with settings_tabs[0]:
            render_config_overview()
        with settings_tabs[1]:
            render_llm_settings()
        with settings_tabs[2]:
            render_portfolio_config()
        with settings_tabs[3]:
            render_data_settings()
    with tabs[6]:
        render_tests()


if __name__ == "__main__":
    main()
