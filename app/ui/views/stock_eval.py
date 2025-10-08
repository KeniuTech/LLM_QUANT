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


def render_stock_evaluation() -> None:
    """渲染股票筛选与评估页面。"""
    st.subheader("股票筛选与评估")
    
    # 1. 时间范围选择
    col1, col2 = st.columns(2)
    with col1:
        end_date = st.date_input(
            "评估截止日期",
            value=datetime.now().date() - timedelta(days=1),
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
    if st.button("开始评估", disabled=not selected_factors):
        with st.spinner("正在评估因子表现..."):
            results = []
            for factor_name in selected_factors:
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
                
            if results:
                st.markdown("##### 因子评估结果")
                result_df = pd.DataFrame(results)
                st.dataframe(
                    result_df,
                    hide_index=True,
                    use_container_width=True
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
                    scores = _calculate_stock_scores(
                        universe,
                        selected_factors,
                        end_date,
                        ic_means
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
                            use_container_width=True
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
    broker = DataBroker()
    
    # 标准化权重
    weights = np.array(factor_weights)
    weights = weights / np.sum(np.abs(weights))
    
    # 获取所有股票的因子值
    stocks = universe or broker.get_all_stocks(eval_date.strftime("%Y%m%d"))
    results = []
    
    for ts_code in stocks:
        if not check_data_sufficiency(ts_code, eval_date.strftime("%Y%m%d")):
            continue
            
        # 获取股票信息
        info = broker.get_stock_info(ts_code)
        if not info:
            continue
            
        # 获取因子值
        factor_values = []
        for factor in factors:
            value = broker.fetch_latest_factor(ts_code, factor, eval_date)
            if value is None:
                break
            factor_values.append(value)
            
        if len(factor_values) != len(factors):
            continue
            
        # 计算综合评分
        score = np.dot(factor_values, weights)
        
        results.append({
            "股票代码": ts_code,
            "股票名称": info.get("name", ""),
            "行业": info.get("industry", ""),
            "综合评分": f"{score:.4f}"
        })
        
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