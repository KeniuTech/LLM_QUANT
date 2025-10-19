"""Streamlit UI scaffold for the investment assistant."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.schema import initialize_database
from app.ingest.checker import run_boot_check
from app.ingest.news import ingest_latest_news
from app.ui.portfolio_config import render_portfolio_config
from app.ui.progress_state import render_factor_progress
from app.ui.shared import LOGGER, LOG_EXTRA, render_tuning_backtest_hints
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
    render_factor_calculation,
    render_tuning_lab,
)
from app.utils.config import get_config

from app.ui.navigation import TOP_NAV_STATE_KEY

def main() -> None:
    LOGGER.info("初始化 Streamlit UI", extra=LOG_EXTRA)
    st.set_page_config(page_title="多智能体个人投资助理", layout="wide")

    initialize_database()

    cfg = get_config()
    # 仅在首次运行时执行自动数据更新，避免 Streamlit 每次重跑都触发该逻辑
    AUTO_UPDATE_FLAG = "auto_update_has_run"
    if cfg.auto_update_data and not st.session_state.get(AUTO_UPDATE_FLAG):
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
                news_count = ingest_latest_news(days_back=1, force=False)
                LOGGER.info(
                    "自动数据更新完成：日线数据覆盖%s-%s，GDELT新闻%s条",
                    report.start,
                    report.end,
                    news_count,
                    extra=LOG_EXTRA,
                )
                st.success(f"✅ 自动数据更新完成：获取 GDELT 新闻 {news_count} 条")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("自动数据更新失败", extra=LOG_EXTRA)
            st.error(f"❌ 自动数据更新失败：{exc}")
        finally:
            st.session_state[AUTO_UPDATE_FLAG] = True

    render_global_dashboard()

    # --- 顶部导航（第三方组件 streamlit-option-menu） ---
    top_labels = ["今日计划", "投资池/仓位", "回测与复盘", "实验调参", "行情可视化", "日志钻取", "数据与设置", "自检测试"]
    if TOP_NAV_STATE_KEY not in st.session_state:
        st.session_state[TOP_NAV_STATE_KEY] = top_labels[0]
    try:
        default_index = top_labels.index(st.session_state[TOP_NAV_STATE_KEY])
    except ValueError:
        default_index = 0
    selected_top = option_menu(
        menu_title=None,
        options=top_labels,
        icons=["calendar", "briefcase", "bar-chart", "cpu", "activity", "file-text", "gear", "bug"],
        orientation="horizontal",
        default_index=default_index,
    )
    st.session_state[TOP_NAV_STATE_KEY] = selected_top
    LOGGER.debug("Top menu selected: %s", selected_top, extra=LOG_EXTRA)

    render_tuning_backtest_hints(selected_top)

    # --- 仅渲染当前选中页（懒加载） ---
    if selected_top == "今日计划":
        render_today_plan()

    elif selected_top == "投资池/仓位":
        render_pool_overview()

    elif selected_top == "回测与复盘":
        sub_labels = ["回测复盘", "股票评估", "因子计算"]
        selected_backtest = option_menu(
            menu_title=None,
            options=sub_labels,
            icons=["bar-chart", "graph-up", "calculator"],
            orientation="horizontal",
        )
        LOGGER.debug("Backtest submenu selected: %s", selected_backtest, extra=LOG_EXTRA)

        if selected_backtest == "回测复盘":
            render_backtest_review()
        elif selected_backtest == "股票评估":
            render_stock_evaluation()
        else:
            render_factor_calculation()

    elif selected_top == "实验调参":
        render_tuning_lab()

    elif selected_top == "行情可视化":
        render_market_visualization()

    elif selected_top == "日志钻取":
        render_log_viewer()

    elif selected_top == "数据与设置":
        st.header("系统设置")
        settings_labels = ["配置概览", "LLM 设置", "投资组合", "数据源"]
        selected_settings = option_menu(
            menu_title=None,
            options=settings_labels,
            icons=["list-task", "cpu", "bank", "database"],
            orientation="horizontal",
        )
        LOGGER.debug("Settings submenu selected: %s", selected_settings, extra=LOG_EXTRA)

        if selected_settings == "配置概览":
            render_config_overview()
        elif selected_settings == "LLM 设置":
            render_llm_settings()
        elif selected_settings == "投资组合":
            render_portfolio_config()
        else:
            render_data_settings()

    elif selected_top == "自检测试":
        render_tests()


if __name__ == "__main__":
    main()
