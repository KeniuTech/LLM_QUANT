"""自检测试视图。"""
from __future__ import annotations

from collections import Counter
from datetime import date

import streamlit as st
import pandas as pd

from app.data.schema import initialize_database
from app.ingest.checker import run_boot_check
from app.ingest.tushare import FetchJob, run_ingestion
from app.llm.client import llm_config_snapshot, run_llm
from app.utils import alerts
from app.utils.config import get_config, save_config
from app.utils.data_quality import run_data_quality_checks

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
            # Defensive: some environments may provide a stub or return None.
            if result is None:
                LOGGER.warning("initialize_database 返回 None（可能为 stub），无法读取执行详情", extra=LOG_EXTRA)
                st.warning("数据库初始化返回空结果，已跳过详细检查。")
            else:
                skipped = getattr(result, "skipped", False)
                executed = getattr(result, "executed", None)
                if skipped:
                    LOGGER.info("数据库已存在，无需初始化", extra=LOG_EXTRA)
                    st.success("数据库已存在，检查通过。")
                else:
                    LOGGER.info("数据库初始化完成，执行语句数=%s", executed, extra=LOG_EXTRA)
                    if executed is None:
                        st.success("数据库初始化完成。")
                    else:
                        st.success(f"数据库初始化完成，共执行 {executed} 条语句。")

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

    st.subheader("新闻数据测试（GDELT）")
    st.write("用于验证 GDELT 配置是否能够正常抓取新闻并写入数据库。")
    news_days = int(
        st.number_input(
            "回溯窗口（天）",
            min_value=1,
            max_value=30,
            value=1,
            step=1,
            help="按天抓取最近区间的新闻。",
        )
    )
    force_news = st.checkbox(
        "强制重新抓取（忽略增量状态）",
        value=False,
    )
    if st.button("运行 GDELT 新闻测试"):
        from app.ingest.news import ingest_latest_news

        LOGGER.info(
            "点击 GDELT 新闻测试按钮 days=%s force=%s",
            news_days,
            force_news,
            extra=LOG_EXTRA,
        )
        with st.spinner("正在抓取 GDELT 新闻..."):
            try:
                count = ingest_latest_news(days_back=news_days, force=force_news)
                st.success(f"GDELT 新闻测试完成，新增 {count} 条新闻记录。")
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("GDELT 新闻测试失败", extra=LOG_EXTRA)
                st.error(f"GDELT 新闻测试失败：{exc}")
                alerts.add_warning("GDELT", "GDELT 新闻测试执行失败", str(exc))
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

    st.divider()
    st.subheader("数据质量验证")
    st.write("快速检查候选池、策略评估、持仓快照与新闻数据的更新情况。")
    dq_window = int(
        st.number_input(
            "数据更新窗口（天）",
            min_value=3,
            max_value=60,
            value=7,
            step=1,
            help="用于判断数据是否过期的时间窗口。",
        )
    )
    if st.button("运行数据质量验证"):
        LOGGER.info("执行数据质量验证 window_days=%s", dq_window, extra=LOG_EXTRA)
        with st.spinner("正在执行数据质量检查..."):
            results = run_data_quality_checks(window_days=dq_window)

        level_counts = Counter(item.severity for item in results)
        metric_cols = st.columns(3)
        metric_cols[0].metric("错误", level_counts.get("ERROR", 0))
        metric_cols[1].metric("警告", level_counts.get("WARN", 0))
        metric_cols[2].metric("提示", level_counts.get("INFO", 0))

        if not results:
            st.info("未返回任何检查结果。")
        else:
            df = pd.DataFrame(
                [
                    {
                        "检查项": item.check,
                        "级别": item.severity,
                        "说明": item.detail,
                        "附加信息": item.extras or {},
                    }
                    for item in results
                ]
            )
            st.dataframe(df, hide_index=True, width="stretch")
            with st.expander("导出检查结果", expanded=False):
                st.download_button(
                    "下载 CSV",
                    data=df.to_csv(index=False),
                    file_name="data_quality_results.csv",
                    mime="text/csv",
                )
