"""自检测试视图。"""
from __future__ import annotations

from datetime import date

import streamlit as st

from app.data.schema import initialize_database
from app.ingest.checker import run_boot_check
from app.ingest.tushare import FetchJob, run_ingestion
from app.llm.client import llm_config_snapshot, run_llm
from app.utils import alerts
from app.utils.config import get_config, save_config

from app.ui.shared import LOGGER, LOG_EXTRA
from app.ui.views.dashboard import update_dashboard_sidebar

def render_tests() -> None:
    LOGGER.info("渲染自检页面", extra=LOG_EXTRA)
    st.header("自检测试")
    st.write("用于快速检查数据库与数据拉取是否正常工作。")

    if st.button("测试数据库初始化"):
        LOGGER.info("点击测试数据库初始化按钮", extra=LOG_EXTRA)
        with st.spinner("正在检查数据库..."):
            result = initialize_database()
            if result.skipped:
                LOGGER.info("数据库已存在，无需初始化", extra=LOG_EXTRA)
                st.success("数据库已存在，检查通过。")
            else:
                LOGGER.info("数据库初始化完成，执行语句数=%s", result.executed, extra=LOG_EXTRA)
                st.success(f"数据库初始化完成，共执行 {result.executed} 条语句。")

    st.divider()

    if st.button("测试 TuShare 拉取（示例 2024-01-01 至 2024-01-03）"):
        LOGGER.info("点击示例 TuShare 拉取按钮", extra=LOG_EXTRA)
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
                LOGGER.info("示例 TuShare 拉取成功", extra=LOG_EXTRA)
                st.success("TuShare 示例拉取完成，数据已写入数据库。")
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("示例 TuShare 拉取失败", extra=LOG_EXTRA)
                st.error(f"拉取失败：{exc}")
                alerts.add_warning("TuShare", "示例拉取失败", str(exc))
                update_dashboard_sidebar()

    st.info("注意：TuShare 拉取依赖网络与 Token，若环境未配置将出现错误提示。")

    st.divider()

    st.subheader("RSS 数据测试")
    st.write("用于验证 RSS 配置是否能够正常抓取新闻并写入数据库。")
    rss_url = st.text_input(
        "测试 RSS 地址",
        value="https://rsshub.app/cls/depth/1000",
        help="留空则使用默认配置的全部 RSS 来源。",
    ).strip()
    rss_hours = int(
        st.number_input(
            "回溯窗口（小时）",
            min_value=1,
            max_value=168,
            value=24,
            step=6,
            help="仅抓取最近指定小时内的新闻。",
        )
    )
    rss_limit = int(
        st.number_input(
            "单源抓取条数",
            min_value=1,
            max_value=200,
            value=50,
            step=10,
        )
    )
    if st.button("运行 RSS 测试"):
        from app.ingest import rss as rss_ingest

        LOGGER.info(
            "点击 RSS 测试按钮 rss_url=%s hours=%s limit=%s",
            rss_url,
            rss_hours,
            rss_limit,
            extra=LOG_EXTRA,
        )
        with st.spinner("正在抓取 RSS 新闻..."):
            try:
                if rss_url:
                    items = rss_ingest.fetch_rss_feed(
                        rss_url,
                        hours_back=rss_hours,
                        max_items=rss_limit,
                    )
                    count = rss_ingest.save_news_items(items)
                else:
                    count = rss_ingest.ingest_configured_rss(
                        hours_back=rss_hours,
                        max_items_per_feed=rss_limit,
                    )
                st.success(f"RSS 测试完成，新增 {count} 条新闻记录。")
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("RSS 测试失败", extra=LOG_EXTRA)
                st.error(f"RSS 测试失败：{exc}")
                alerts.add_warning("RSS", "RSS 测试执行失败", str(exc))
                update_dashboard_sidebar()

    st.divider()
    days = int(
        st.number_input(
            "检查窗口（天数）",
            min_value=30,
            max_value=10950,
            value=365,
            step=30,
        )
    )
    LOGGER.debug("检查窗口天数=%s", days, extra=LOG_EXTRA)
    cfg = get_config()
    force_refresh = st.checkbox(
        "强制刷新数据（关闭增量跳过）",
        value=cfg.force_refresh,
        help="勾选后将重新拉取所选区间全部数据",
    )
    if force_refresh != cfg.force_refresh:
        cfg.force_refresh = force_refresh
        LOGGER.info("更新 force_refresh=%s", force_refresh, extra=LOG_EXTRA)
        save_config()

    if st.button("执行手动数据同步"):
        LOGGER.info("点击执行手动数据同步按钮", extra=LOG_EXTRA)
        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()
        log_placeholder = st.empty()
        messages: list[str] = []

        def hook(message: str, value: float) -> None:
            progress_bar.progress(min(max(value, 0.0), 1.0))
            status_placeholder.write(message)
            messages.append(message)
            LOGGER.debug("手动数据同步进度：%s -> %.2f", message, value, extra=LOG_EXTRA)

        with st.spinner("正在执行手动数据同步..."):
            try:
                report = run_boot_check(
                    days=days,
                    progress_hook=hook,
                    force_refresh=force_refresh,
                )
                LOGGER.info("手动数据同步成功", extra=LOG_EXTRA)
                st.success("手动数据同步完成，以下为数据覆盖摘要。")
                st.json(report.to_dict())
                if messages:
                    log_placeholder.markdown("\n".join(f"- {msg}" for msg in messages))
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("手动数据同步失败", extra=LOG_EXTRA)
                st.error(f"手动数据同步失败：{exc}")
                alerts.add_warning("数据同步", "手动数据同步失败", str(exc))
                update_dashboard_sidebar()
                if messages:
                    log_placeholder.markdown("\n".join(f"- {msg}" for msg in messages))
            finally:
                progress_bar.progress(1.0)

    st.divider()
    st.subheader("LLM 接口测试")
    st.json(llm_config_snapshot())
    llm_prompt = st.text_area("测试 Prompt", value="请概述今天的市场重点。", height=160)
    system_prompt = st.text_area(
        "System Prompt (可选)",
        value="你是一名量化策略研究助手，用简洁中文回答。",
        height=100,
    )
    if st.button("执行 LLM 测试"):
        with st.spinner("正在调用 LLM..."):
            try:
                response = run_llm(llm_prompt, system=system_prompt or None)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("LLM 测试失败", extra=LOG_EXTRA)
                st.error(f"LLM 调用失败：{exc}")
            else:
                LOGGER.info("LLM 测试成功", extra=LOG_EXTRA)
                st.success("LLM 调用成功，以下为返回内容：")
                st.write(response)
