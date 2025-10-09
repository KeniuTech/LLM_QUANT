"""股票筛选与评估视图。"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app.features.evaluation import evaluate_factor
from app.features.factors import DEFAULT_FACTORS
from app.features.validation import check_data_sufficiency
from app.utils.config import get_config
from app.utils.data_access import DataBroker
from app.utils.db import db_session
from app.utils.logging import get_logger


def _get_latest_trading_date() -> datetime.date:
    """获取数据库中的最新交易日期"""
    with db_session() as session:
        # 获取当前日期的上一个有效交易日
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
        return datetime.now().date() - timedelta(days=1)  # 如果查询失败才返回昨天

def render_stock_evaluation() -> None:
    """渲染股票筛选与评估页面。"""
    LOGGER = get_logger(__name__)
    LOG_EXTRA = {"stage": "stock_evaluation_ui"}
    
    st.subheader("股票筛选与评估")
    
    # 记录页面加载
    LOGGER.info("股票筛选与评估页面已加载", extra=LOG_EXTRA)
    
    # 1. 时间范围选择
    col1, col2 = st.columns(2)
    with col1:
        latest_date = _get_latest_trading_date()
        end_date = st.date_input(
            "评估截止日期",
            value=latest_date,
            help="选择评估的截止日期"
        )
    with col2:
        lookback_days = st.slider(
            "回溯天数",
            min_value=30,
            max_value=360,
            value=180,
            step=30,
            help="选择评估的历史数据长度"
        )
    start_date = end_date - timedelta(days=lookback_days)
    
    # 2. 因子选择
    st.markdown("##### 评估因子选择")
    factor_groups = {
        "动量类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("mom_")],
        "波动率类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("volat_")],
        "换手率类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("turn_")],
        "估值类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("val_")],
        "量价类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("volume_")],
        "市场类因子": [f for f in DEFAULT_FACTORS if f.name.startswith("market_")]
    }
    
    # 定义默认选中的关键常用因子
    DEFAULT_SELECTED_FACTORS = {
        "mom_5",   # 5日动量
        "mom_20",  # 20日动量
        "mom_60",  # 60日动量
        "volat_20",  # 20日波动率
        "turn_5",   # 5日换手率
        "turn_20",  # 20日换手率
        "val_pe_score",  # PE评分
        "val_pb_score",  # PB评分
        "volume_ratio_score",  # 量比评分
        "risk_penalty"  # 风险惩罚项
    }
    
    selected_factors = []
    for group_name, factors in factor_groups.items():
        if factors:
            st.markdown(f"###### {group_name}")
            cols = st.columns(3)
            for i, factor in enumerate(factors):
                if cols[i % 3].checkbox(
                    factor.name,
                    value=factor.name in DEFAULT_SELECTED_FACTORS,
                    help=factor.description if hasattr(factor, 'description') else None
                ):
                    selected_factors.append(factor.name)
    
    if not selected_factors:
        st.warning("请至少选择一个评估因子")
        return
        
    # 3. 股票池范围
    st.markdown("##### 股票池范围")
    pool_type = st.radio(
        "选择股票池",
        ["沪深300", "中证500", "中证1000", "全部A股", "自定义"],
        index=0,  # 默认选择沪深300
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
            # 从数据库获取对应指数成分股
            index_code = {
                "沪深300": "000300.SH",
                "中证500": "000905.SH", 
                "中证1000": "000852.SH"
            }[pool_type]
            universe = broker.get_index_stocks(
                index_code,
                end_date.strftime("%Y%m%d")
            )
            
    # 4. 评估结果
    
    # 初始化会话状态
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'evaluation_status' not in st.session_state:
        st.session_state.evaluation_status = 'idle'  # idle, running, completed, error
    if 'current_factor' not in st.session_state:
        st.session_state.current_factor = ''
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    
    # 同步评估函数
    def run_evaluation_sync():
        try:
            # 记录评估开始
            LOGGER.info(
                "开始因子评估 因子数量=%s 评估日期=%s 至 %s",
                len(selected_factors),
                start_date,
                end_date,
                extra=LOG_EXTRA
            )
            
            st.session_state.evaluation_status = 'running'
            results = []
            
            for i, factor_name in enumerate(selected_factors):
                st.session_state.current_factor = factor_name
                st.session_state.progress = (i / len(selected_factors)) * 100
                
                performance = evaluate_factor(
                    factor_name,
                    start_date,
                    end_date,
                    universe=universe
                )
                results.append({
                    "因子": factor_name,
                    "IC均值": f"{performance.ic_mean:.4f}",
                    "RankIC均值": f"{performance.rank_ic_mean:.4f}",
                    "IC信息比率": f"{performance.ic_ir:.4f}",
                    "夏普比率": f"{performance.sharpe_ratio:.4f}" if performance.sharpe_ratio else "N/A",
                    "换手率": f"{performance.turnover_rate*100:.1f}%" if performance.turnover_rate else "N/A"
                })
            
            st.session_state.evaluation_results = results
            st.session_state.evaluation_status = 'completed'
            st.session_state.progress = 100
            
        except Exception as e:
            st.session_state.evaluation_status = 'error'
            st.session_state.evaluation_error = str(e)
    
    # 显示进度
    if st.session_state.evaluation_status == 'running':
        st.info(f"正在评估因子: {st.session_state.current_factor}")
        st.progress(st.session_state.progress / 100)
    elif st.session_state.evaluation_status == 'completed':
        st.success("因子评估完成！")
    elif st.session_state.evaluation_status == 'error':
        st.error(f"评估失败: {st.session_state.evaluation_error}")
    
    # 开始评估按钮
    if st.button("开始评估", disabled=not selected_factors or st.session_state.evaluation_status == 'running'):
        # 重置状态
        st.session_state.evaluation_results = None
        st.session_state.evaluation_status = 'running'
        st.session_state.progress = 0
        
        # 直接调用同步评估函数
        run_evaluation_sync()
    
    # 显示结果
    if st.session_state.evaluation_results:
        results = st.session_state.evaluation_results
        
        st.markdown("##### 因子评估结果")
        result_df = pd.DataFrame(results)
        st.dataframe(
            result_df,
            hide_index=True,
            width="stretch"
        )
        
        # 绘制IC均值分布
        ic_means = [float(r["IC均值"]) for r in results]
        chart_df = pd.DataFrame({
            "因子": [r["因子"] for r in results],
            "IC均值": ic_means
        })
        st.bar_chart(chart_df.set_index("因子"))
        
        # 生成股票评分
        with st.spinner("正在生成股票评分..."):
            # 使用IC均值作为权重，但如果IC均值全为零，则使用均匀分布
            if all(mean == 0 for mean in ic_means):
                factor_weights = [1.0 / len(ic_means)] * len(ic_means)
                LOGGER.info("所有因子IC均值均为零，使用均匀权重", extra=LOG_EXTRA)
            else:
                # 将IC均值归一化为权重
                abs_sum = sum(abs(m) for m in ic_means)
                factor_weights = [m / abs_sum for m in ic_means]
                LOGGER.info("使用IC均值作为权重: %s", factor_weights, extra=LOG_EXTRA)
                
            scores = _calculate_stock_scores(
                universe,
                selected_factors,
                end_date,
                factor_weights
            )
            
            if scores:
                st.markdown("##### 股票综合评分 (Top 20)")
                score_df = pd.DataFrame(scores).sort_values(
                    "综合评分",
                    ascending=False
                ).head(20)
                st.dataframe(
                    score_df,
                    hide_index=True,
                    width="stretch"
                )
                
                # 添加入池功能
                if st.button("将Top 20股票加入股票池"):
                    _add_to_stock_pool(
                        score_df["股票代码"].tolist(),
                        end_date
                    )
                    st.success("已成功将选中股票加入股票池！")


def _calculate_stock_scores(
    universe: Optional[List[str]],
    factors: List[str],
    eval_date: datetime.date,
    factor_weights: List[float]
) -> List[Dict[str, str]]:
    """计算股票的综合评分。"""
    LOGGER = get_logger(__name__)
    LOG_EXTRA = {"stage": "stock_evaluation"}
    
    broker = DataBroker()
    
    # 记录评估开始
    LOGGER.info(
        "开始股票评估评估日期=%s 因子数量=%d 权重=%s",
        eval_date.strftime("%Y-%m-%d"),
        len(factors),
        factor_weights,
        extra=LOG_EXTRA
    )
    
    # 标准化权重
    weights = np.array(factor_weights)
    abs_sum = np.sum(np.abs(weights))
    if abs_sum > 0:  # 避免除以零
        weights = weights / abs_sum
    else:
        # 如果所有权重都是零，则使用均匀分布
        weights = np.ones_like(weights) / len(weights)
    
    # 获取所有股票的因子值
    stocks = universe or broker.get_all_stocks(eval_date.strftime("%Y%m%d"))
    
    # 记录股票列表信息
    LOGGER.info(
        "获取股票列表 universe_size=%d total_stocks=%d",
        len(universe) if universe else 0,
        len(stocks),
        extra=LOG_EXTRA
    )
    
    results = []
    
    evaluated_count = 0
    skipped_count = 0
    
    for ts_code in stocks:
        # 检查数据是否充分
        if not check_data_sufficiency(ts_code, eval_date.strftime("%Y%m%d")):
            skipped_count += 1
            continue
            
        # 获取股票信息
        info = broker.get_stock_info(ts_code)
        if not info:
            skipped_count += 1
            continue
            
        # 获取因子值
        factor_values = []
        for factor in factors:
            value = broker.fetch_latest_factor(ts_code, factor, eval_date)
            if value is None:
                skipped_count += 1
                break
            factor_values.append(value)
            
        # 检查是否所有因子值都已获取
        if len(factor_values) != len(factors):
            skipped_count += 1
            continue
            
        # 计算综合评分
        score = np.dot(factor_values, weights)
        evaluated_count += 1
        
        results.append({
            "股票代码": ts_code,
            "股票名称": info.get("name", ""),
            "行业": info.get("industry", ""),
            "综合评分": f"{score:.4f}"
        })
        
    # 记录评估完成信息
    LOGGER.info(
        "股票评估完成 总股票数=%d 已评估=%d 跳过=%d 结果数=%d",
        len(stocks),
        evaluated_count,
        skipped_count,
        len(results),
        extra=LOG_EXTRA
    )
    
    return results


def _add_to_stock_pool(
    ts_codes: List[str],
    eval_date: datetime.date
) -> None:
    """将股票添加到股票池。"""
    with db_session() as session:
        # 删除已有记录
        session.execute(
            """
            DELETE FROM stock_pool 
            WHERE entry_date = :entry_date
            """,
            {"entry_date": eval_date}
        )
        
        # 插入新记录
        values = [
            {
                "ts_code": code,
                "entry_date": eval_date,
                "entry_reason": "factor_evaluation"
            }
            for code in ts_codes
        ]
        
        session.execute(
            """
            INSERT INTO stock_pool (ts_code, entry_date, entry_reason)
            VALUES (:ts_code, :entry_date, :entry_reason)
            """,
            values
        )
        
        session.commit()