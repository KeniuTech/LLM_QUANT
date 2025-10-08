"""因子计算页面。"""
from datetime import datetime, timedelta
from typing import List, Optional

import streamlit as st

from app.features.factors import compute_factors, DEFAULT_FACTORS, FactorSpec
from app.ui.progress_state import factor_progress
from app.utils.data_access import DataBroker
from app.utils.db import db_session


def _get_latest_trading_date() -> datetime.date:
    """获取数据库中的最新交易日期"""
    with db_session() as session:
        result = session.execute(
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
        with db_session() as session:
            latest_date = _get_latest_trading_date()
            result = session.execute(
                """
                SELECT DISTINCT ts_code 
                FROM daily 
                WHERE trade_date = :trade_date
                """,
                {"trade_date": latest_date.strftime("%Y%m%d")}
            ).fetchall()
            
            return [row[0] for row in result] if result else []
    except Exception as e:
        st.error(f"获取股票列表失败: {str(e)}")
        return []


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
    
    # 5. 同步计算函数
    def run_factor_calculation_sync():
        """同步执行因子计算"""
        # 计算参数
        total_stocks = len(universe) if universe else len(_get_all_stocks())
        total_batches = len(selected_factors)
        
        try:
            # 执行因子计算
            results = []
            for i, factor in enumerate(selected_factors):
                # 更新批次进度
                factor_progress.update_progress(
                    current_securities=0,
                    current_batch=i+1,
                    message=f"正在计算因子: {factor.name}"
                )
                
                # 计算单个交易日的因子
                current_date = start_date
                while current_date <= end_date:
                    try:
                        # 计算指定日期的因子
                        daily_results = compute_factors(
                            current_date,
                            [factor],
                            ts_codes=universe,
                            skip_existing=skip_existing
                        )
                        results.extend(daily_results)
                        
                    except Exception as e:
                        # 记录错误但不中断计算
                        error_msg = f"计算因子 {factor.name} 在日期 {current_date} 时出错: {str(e)}"
                        print(f"ERROR: {error_msg}")
                    
                    current_date += timedelta(days=1)
            
            # 计算完成
            factor_progress.complete_calculation(f"因子计算完成！共计算 {len(results)} 条因子记录")
            
            return {
                'success': True,
                'results': results,
                'factors': [f.name for f in selected_factors],
                'date_range': f"{start_date} 至 {end_date}",
                'stock_count': len(set(r.ts_code for r in results)) if results else 0,
                'message': f"因子计算完成！共计算 {len(results)} 条因子记录"
            }
                
        except Exception as e:
            # 计算失败
            factor_progress.error_occurred(f"因子计算失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': f"因子计算失败: {str(e)}"
            }
    
    # 6. 开始计算按钮
    if st.button("开始计算因子", disabled=not selected_factors):
        # 重置状态
        if 'factor_calculation_results' in st.session_state:
            st.session_state.factor_calculation_results = None
        if 'factor_calculation_error' in st.session_state:
            st.session_state.factor_calculation_error = None
        
        # 初始化进度状态
        total_stocks = len(universe) if universe else len(_get_all_stocks())
        factor_progress.start_calculation(
            total_securities=total_stocks,
            total_batches=len(selected_factors)
        )
        
        # 直接调用同步计算函数
        result = run_factor_calculation_sync()
        
        # 处理计算结果
        if result['success']:
            st.session_state.factor_calculation_results = {
                'results': result['results'],
                'factors': result['factors'],
                'date_range': result['date_range'],
                'stock_count': result['stock_count']
            }
            st.success("✅ 因子计算完成！")
        else:
            st.session_state.factor_calculation_error = result['error']
            st.error(f"❌ 因子计算失败: {result['error']}")
    
    # 7. 显示计算结果
    if 'factor_calculation_results' in st.session_state and st.session_state.factor_calculation_results:
        results = st.session_state.factor_calculation_results
        
        st.success("✅ 因子计算完成！")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("计算因子数量", len(results['factors']))
        with col2:
            st.metric("涉及股票数量", results['stock_count'])
        with col3:
            st.metric("计算时间范围", results['date_range'])
        
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
                    st.dataframe(df.head(100), use_container_width=True)  # 只显示前100条
                    st.info(f"共 {len(df_data)} 条因子记录（显示前100条）")
            else:
                st.info("没有找到因子计算结果")
    
    # 8. 移除异步线程检查逻辑（已改为同步模式）
    
    # 9. 显示错误信息
    if 'factor_calculation_error' in st.session_state and st.session_state.factor_calculation_error:
        st.error(f"❌ 因子计算失败: {st.session_state.factor_calculation_error}")