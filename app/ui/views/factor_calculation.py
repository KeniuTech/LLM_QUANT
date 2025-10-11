"""因子计算页面。"""
from datetime import date, datetime, timedelta
from typing import List, Optional, Sequence

import streamlit as st

from app.features.factors import DEFAULT_FACTORS, FactorSpec, compute_factor_range
from app.ui.progress_state import factor_progress
from app.ui.shared import LOGGER, LOG_EXTRA
from app.utils.data_access import DataBroker
from app.utils.db import db_session


def _get_latest_trading_date() -> datetime.date:
    """获取数据库中的最新交易日期"""
    with db_session(read_only=True) as conn:
        result = conn.execute(
            """
            SELECT trade_date 
            FROM daily_basic 
            WHERE trade_date <= :today
            GROUP BY trade_date 
            ORDER BY trade_date DESC 
            LIMIT 1
            """,
            {"today": datetime.now().strftime("%Y%m%d")}
        ).fetchone()
        
        if result and result[0]:
            return datetime.strptime(str(result[0]), "%Y%m%d").date()
        return datetime.now().date() - timedelta(days=1)


def _get_all_stocks() -> List[str]:
    """获取所有股票代码"""
    try:
        # 从daily表获取所有股票代码
        with db_session(read_only=True) as conn:
            latest_date = _get_latest_trading_date()
            result = conn.execute(
                """
                SELECT DISTINCT ts_code 
                FROM daily 
                WHERE trade_date = :trade_date
                """,
                {"trade_date": latest_date.strftime("%Y%m%d")}
            ).fetchall()
            
            return [row["ts_code"] for row in result if row and row["ts_code"]] if result else []
    except Exception as exc:
        LOGGER.exception("获取股票列表失败", extra={**LOG_EXTRA, "error": str(exc)})
        st.error(f"获取股票列表失败: {exc}")
        return []


def _normalize_universe(universe: Optional[Sequence[str]]) -> List[str]:
    """去重并规范股票代码格式。"""
    if not universe:
        return []
    seen: dict[str, None] = {}
    for code in universe:
        normalized = code.strip().upper()
        if normalized and normalized not in seen:
            seen[normalized] = None
    return list(seen.keys())


def _get_trade_dates_between(
    start: date,
    end: date,
    universe: Optional[Sequence[str]] = None,
) -> List[date]:
    """获取区间内存在行情数据的交易日期列表。"""

    if end < start:
        return []

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    params: List[str] = [start_str, end_str]
    query = (
        "SELECT DISTINCT trade_date FROM daily "
        "WHERE trade_date BETWEEN ? AND ?"
    )
    scoped_universe = _normalize_universe(universe)
    if scoped_universe:
        placeholders = ", ".join("?" for _ in scoped_universe)
        query += f" AND ts_code IN ({placeholders})"
        params.extend(scoped_universe)
    query += " ORDER BY trade_date"

    with db_session(read_only=True) as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        datetime.strptime(str(row["trade_date"]), "%Y%m%d").date()
        for row in rows
        if row and row["trade_date"]
    ]


def _estimate_total_workload(
    trade_dates: Sequence[date],
    universe: Optional[Sequence[str]],
) -> int:
    """估算本次计算需要处理的证券数量，用于驱动进度条。"""

    trade_days = list(trade_dates)
    if not trade_days:
        return 0

    scoped_universe = _normalize_universe(universe)
    if scoped_universe:
        return len(scoped_universe) * len(trade_days)

    start_str = min(trade_days).strftime("%Y%m%d")
    end_str = max(trade_days).strftime("%Y%m%d")
    with db_session(read_only=True) as conn:
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT ts_code) AS cnt
            FROM daily
            WHERE trade_date BETWEEN ? AND ?
            """,
            (start_str, end_str),
        ).fetchone()
    universe_size = int(row["cnt"]) if row and row["cnt"] is not None else 0
    return universe_size * len(trade_days)


def render_factor_calculation() -> None:
    """渲染因子计算页面。"""
    st.subheader("📊 因子计算")
    st.caption("计算指定日期范围的因子值")
    
    # 1. 时间范围选择
    col1, col2 = st.columns(2)
    with col1:
        latest_date = _get_latest_trading_date()
        end_date = st.date_input(
            "计算截止日期",
            value=latest_date,
            help="选择因子计算的截止日期"
        )
    with col2:
        lookback_days = st.slider(
            "回溯天数",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="选择计算的历史数据长度"
        )
    start_date = end_date - timedelta(days=lookback_days)
    
    st.info(f"计算范围: {start_date} 至 {end_date} (共{lookback_days}天)")
    
    # 2. 因子选择
    st.markdown("##### 选择要计算的因子")
    
    # 按因子类型分组
    factor_groups = {
        "动量类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("mom_")],
        "波动率类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("volat_")],
        "换手率类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("turn_")],
        "估值类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("val_")],
        "量价类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("volume_")],
        "市场类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("market_")],
        "其他因子": [f for f in DEFAULT_FACTORS if not any(f.name.startswith(prefix) 
                          for prefix in ["mom_", "volat_", "turn_", "val_", "volume_", "market_"])]
    }
    
    selected_factors = []
    for group_name, factors in factor_groups.items():
        if factors:
            st.markdown(f"###### {group_name}")
            cols = st.columns(3)
            for i, factor in enumerate(factors):
                if cols[i % 3].checkbox(
                    factor.name,
                    value=True,  # 默认全选
                    help=factor.description if hasattr(factor, 'description') else None,
                    key=f"factor_checkbox_{factor.name}_{group_name}"  # 添加唯一key
                ):
                    selected_factors.append(factor)
    
    if not selected_factors:
        st.warning("请至少选择一个因子进行计算")
        return
    
    # 3. 股票池选择
    st.markdown("##### 股票池范围")
    pool_type = st.radio(
        "选择股票池",
        ["全部A股", "沪深300", "中证500", "中证1000", "自定义"],
        index=0,
        horizontal=True
    )
    
    universe: Optional[List[str]] = None
    if pool_type != "全部A股":
        broker = DataBroker()
        if pool_type == "自定义":
            custom_codes = st.text_area(
                "输入股票代码列表(每行一个)",
                help="请输入股票代码，每行一个，例如: 000001.SZ"
            )
            if custom_codes:
                universe = [
                    code.strip()
                    for code in custom_codes.split("\n")
                    if code.strip()
                ]
        else:
            index_code = {
                "沪深300": "000300.SH",
                "中证500": "000905.SH", 
                "中证1000": "000852.SH"
            }[pool_type]
            universe = broker.get_index_stocks(
                index_code,
                end_date.strftime("%Y%m%d")
            )
    
    # 4. 计算选项
    st.markdown("##### 计算选项")
    skip_existing = st.checkbox(
        "跳过已计算的因子",
        value=True,
        help="如果勾选，将跳过数据库中已存在的因子计算结果"
    )

    st.markdown("##### 数据维护")
    maintenance_col1, maintenance_col2 = st.columns([1, 2])
    with maintenance_col1:
        clear_confirm = st.checkbox("确认清空因子表", key="factor_clear_confirm")
    with maintenance_col2:
        if st.button("清空因子表数据", disabled=not clear_confirm):
            try:
                with db_session() as conn:
                    conn.execute("DELETE FROM factors")
                st.session_state.pop('factor_calculation_results', None)
                st.session_state.pop('factor_calculation_error', None)
                factor_progress.reset()
                st.success("因子表数据已清空。")
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("清空因子表失败", extra={**LOG_EXTRA, "error": str(exc)})
                st.error(f"清空因子表失败：{exc}")
            finally:
                # 使用st.rerun()代替直接修改session_state
                # 清空因子表后重置页面状态
                st.rerun()

    # 5. 开始计算按钮
    if st.button("开始计算因子", disabled=not selected_factors):
        # 重置状态
        st.session_state.pop('factor_calculation_results', None)
        st.session_state.pop('factor_calculation_error', None)
        factor_progress.reset()

        scoped_universe = _normalize_universe(universe) or None
        trade_dates = _get_trade_dates_between(start_date, end_date, scoped_universe)
        if not trade_dates:
            st.warning("所选时间窗口内无可用交易日数据，请先执行数据同步。")
            return

        total_workload = _estimate_total_workload(trade_dates, scoped_universe)
        factor_progress.start_calculation(
            total_securities=max(total_workload, 1),
            total_batches=len(trade_dates),
        )

        with st.spinner("正在计算因子..."):
            try:
                results = compute_factor_range(
                    start=min(trade_dates),
                    end=max(trade_dates),
                    factors=selected_factors,
                    ts_codes=scoped_universe,
                    skip_existing=skip_existing,
                )
            except Exception as exc:
                LOGGER.exception("因子计算失败", extra={**LOG_EXTRA, "error": str(exc)})
                factor_progress.error_occurred(str(exc))
                st.session_state.factor_calculation_error = str(exc)
                st.error(f"❌ 因子计算失败: {exc}")
            else:
                factor_progress.complete_calculation(
                    f"因子计算完成，共生成 {len(results)} 条因子记录"
                )
                factor_names = [spec.name for spec in selected_factors]
                stock_count = len({item.ts_code for item in results}) if results else 0
                st.session_state.factor_calculation_results = {
                    'results': results,
                    'factors': factor_names,
                    'date_range': f"{trade_dates[0]} 至 {trade_dates[-1]}",
                    'stock_count': stock_count,
                    'trade_days': len(trade_dates),
                }
                st.success("✅ 因子计算完成！")
    
    # 6. 显示计算结果
    if 'factor_calculation_results' in st.session_state and st.session_state.factor_calculation_results:
        results = st.session_state.factor_calculation_results
        
        st.success("✅ 因子计算完成！")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("计算因子数量", len(results['factors']))
        with col2:
            st.metric("涉及股票数量", results['stock_count'])
        with col3:
            st.metric("交易日数量", results.get('trade_days', 0))
        st.caption(f"时间范围：{results['date_range']}")
        
        # 显示计算详情
        with st.expander("查看计算详情"):
            if results['results']:
                # 转换为DataFrame显示
                import pandas as pd
                df_data = []
                for result in results['results']:
                    for factor_name, value in result.values.items():
                        df_data.append({
                            '日期': result.trade_date,
                            '股票代码': result.ts_code,
                            '因子名称': factor_name,
                            '因子值': value
                        })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    st.dataframe(df.head(100), width="stretch")  # 只显示前100条
                    st.info(f"共 {len(df_data)} 条因子记录（显示前100条）")
            else:
                st.info("没有找到因子计算结果")
    
    # 7. 显示错误信息
    if 'factor_calculation_error' in st.session_state and st.session_state.factor_calculation_error:
        st.error(f"❌ 因子计算失败: {st.session_state.factor_calculation_error}")
