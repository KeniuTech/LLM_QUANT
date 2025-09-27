"""Streamlit UI scaffold for the investment assistant."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.backtest.engine import BtConfig, run_backtest
from app.data.schema import initialize_database
from app.ingest.checker import run_boot_check
from app.ingest.tushare import FetchJob, run_ingestion
from app.llm.explain import make_human_card
from app.utils.config import get_config
from app.utils.db import db_session




def _load_stock_options(limit: int = 500) -> list[str]:
    try:
        with db_session(read_only=True) as conn:
            rows = conn.execute(
                "SELECT ts_code, name FROM stock_basic WHERE list_status = 'L' ORDER BY ts_code"
            ).fetchall()
    except Exception:
        return []
    options: list[str] = []
    for row in rows[:limit]:
        code = row["ts_code"]
        name = row["name"] or ""
        label = f"{code} | {name}" if name else code
        options.append(label)
    return options


def _parse_ts_code(selection: str) -> str:
    return selection.split(' | ')[0].strip().upper()


def _load_daily_frame(ts_code: str, start: date, end: date) -> pd.DataFrame:
    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')
    range_query = (
        "SELECT trade_date, open, high, low, close, vol, amount "
        "FROM daily WHERE ts_code = ? AND trade_date BETWEEN ? AND ? ORDER BY trade_date"
    )
    fallback_query = (
        "SELECT trade_date, open, high, low, close, vol, amount "
        "FROM daily WHERE ts_code = ? ORDER BY trade_date DESC LIMIT 200"
    )
    with db_session(read_only=True) as conn:
        df = pd.read_sql_query(range_query, conn, params=(ts_code, start_str, end_str))
        if df.empty:
            df = pd.read_sql_query(fallback_query, conn, params=(ts_code,))
            if df.empty:
                return df
            df = df.sort_values('trade_date')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    return df
def render_today_plan() -> None:
    st.header("今日计划")
    st.write("待接入候选池筛选与多智能体决策结果。")
    sample = make_human_card("000001.SZ", "2025-01-01", {"decisions": []})
    st.json(sample)


def render_backtest() -> None:
    st.header("回测与复盘")
    st.write("在此运行回测、展示净值曲线与代理贡献。")

    default_start = date(2020, 1, 1)
    default_end = date(2020, 3, 31)

    col1, col2 = st.columns(2)
    start_date = col1.date_input("开始日期", value=default_start)
    end_date = col2.date_input("结束日期", value=default_end)
    universe_text = st.text_input("股票列表（逗号分隔）", value="000001.SZ")
    target = st.number_input("目标收益（例：0.035 表示 3.5%）", value=0.035, step=0.005, format="%.3f")
    stop = st.number_input("止损收益（例：-0.015 表示 -1.5%）", value=-0.015, step=0.005, format="%.3f")
    hold_days = st.number_input("持有期（交易日）", value=10, step=1)

    if st.button("运行回测"):
        with st.spinner("正在执行回测..."):
            try:
                universe = [code.strip() for code in universe_text.split(',') if code.strip()]
                cfg = BtConfig(
                    id="streamlit_demo",
                    name="Streamlit Demo Strategy",
                    start_date=start_date,
                    end_date=end_date,
                    universe=universe,
                    params={
                        "target": target,
                        "stop": stop,
                        "hold_days": int(hold_days),
                    },
                )
                result = run_backtest(cfg)
                st.success("回测执行完成，详见回测结果摘要。")
                st.json({"nav_records": result.nav_series, "trades": result.trades})
            except Exception as exc:  # noqa: BLE001
                st.error(f"回测执行失败：{exc}")


def render_settings() -> None:
    st.header("数据与设置")
    cfg = get_config()
    token = st.text_input("TuShare Token", value=cfg.tushare_token or "", type="password")
    if st.button("保存 Token"):
        cfg.tushare_token = token.strip() or None
        st.success("TuShare Token 已更新，仅保存在当前会话。")

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

    st.divider()
    days = int(st.number_input("检查窗口（天数）", min_value=30, max_value=1095, value=365, step=30))
    if st.button("执行开机检查"):
        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()
        log_placeholder = st.empty()
        messages: list[str] = []

        def hook(message: str, value: float) -> None:
            progress_bar.progress(min(max(value, 0.0), 1.0))
            status_placeholder.write(message)
            messages.append(message)

        with st.spinner("正在执行开机检查..."):
            try:
                report = run_boot_check(days=days, progress_hook=hook)
                st.success("开机检查完成，以下为数据覆盖摘要。")
                st.json(report.to_dict())
                if messages:
                    log_placeholder.markdown("\n".join(f"- {msg}" for msg in messages))
            except Exception as exc:  # noqa: BLE001
                st.error(f"开机检查失败：{exc}")
                if messages:
                    log_placeholder.markdown("\n".join(f"- {msg}" for msg in messages))
            finally:
                progress_bar.progress(1.0)

    st.divider()
    st.subheader("股票行情可视化")
    options = _load_stock_options()
    default_code = options[0] if options else "000001.SZ"

    if options:
        selection = st.selectbox("选择股票", options, index=0)
        ts_code = _parse_ts_code(selection)
    else:
        ts_code = st.text_input("输入股票代码（如 000001.SZ）", value=default_code).strip().upper()

    viz_col1, viz_col2 = st.columns(2)
    default_start = date.today() - timedelta(days=180)
    start_date = viz_col1.date_input("开始日期", value=default_start, key="viz_start")
    end_date = viz_col2.date_input("结束日期", value=date.today(), key="viz_end")

    if start_date > end_date:
        st.error("开始日期不能晚于结束日期")
        return

    with st.spinner("正在加载行情数据..."):
        try:
            df = _load_daily_frame(ts_code, start_date, end_date)
        except Exception as exc:  # noqa: BLE001
            st.error(f"读取数据失败：{exc}")
            return

    if df.empty:
        st.warning("未查询到该区间的交易数据，请确认数据库已拉取对应日线。")
        return

    price_df = df[["close"]].rename(columns={"close": "收盘价"})
    volume_df = df[["vol"]].rename(columns={"vol": "成交量(手)"})

    if price_df.shape[0] > 180:
        sampled = price_df.resample('3D').last().dropna()
    else:
        sampled = price_df

    if volume_df.shape[0] > 180:
        volume_sampled = volume_df.resample('3D').mean().dropna()
    else:
        volume_sampled = volume_df

    first_close = sampled.iloc[0, 0]
    last_close = sampled.iloc[-1, 0]
    delta_abs = last_close - first_close
    delta_pct = (delta_abs / first_close * 100) if first_close else 0.0

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("最新收盘价", f"{last_close:.2f}", delta=f"{delta_abs:+.2f}")
    metric_col2.metric("区间涨跌幅", f"{delta_pct:+.2f}%")
    metric_col3.metric("平均成交量", f"{volume_sampled['成交量(手)'].mean():.0f}")

    st.line_chart(sampled, width='stretch')
    st.bar_chart(volume_sampled, width='stretch')
    st.caption("提示：成交量单位为手，若需更长区间请调整日期后重新加载。")
    st.dataframe(df.reset_index().tail(10), width='stretch')


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
