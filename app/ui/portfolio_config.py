"""Portfolio configuration UI components."""
from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from app.utils.portfolio_init import get_portfolio_config, update_portfolio_config


def render_portfolio_config() -> None:
    """渲染投资组合配置界面."""
    st.title("投资组合配置")
    
    # 获取当前配置
    config = get_portfolio_config()
    
    # 基本配置部分
    st.header("基本配置")
    col1, col2 = st.columns(2)
    
    with col1:
        initial_capital = st.number_input(
            "初始投资金额",
            min_value=100000,
            max_value=100000000,
            value=int(config["initial_capital"]),
            step=100000,
            format="%d"
        )
        
    with col2:
        currency = st.selectbox(
            "币种",
            options=["CNY", "USD", "HKD"],
            index=["CNY", "USD", "HKD"].index(config["currency"])
        )
        
    # 仓位限制配置
    st.header("仓位限制")
    position_limits = config["position_limits"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_position = st.slider(
            "单个持仓上限",
            min_value=0.05,
            max_value=0.50,
            value=float(position_limits["max_position"]),
            step=0.01,
            format="%.2f",
            help="单个股票最大持仓比例"
        )
        
        min_position = st.slider(
            "单个持仓下限",
            min_value=0.01,
            max_value=0.10,
            value=float(position_limits["min_position"]),
            step=0.01,
            format="%.2f",
            help="单个股票最小持仓比例"
        )
        
    with col2:
        max_total_positions = st.slider(
            "最大持仓数",
            min_value=5,
            max_value=50,
            value=int(position_limits["max_total_positions"]),
            step=1,
            help="投资组合中的最大股票数量"
        )
        
        max_sector_exposure = st.slider(
            "行业敞口上限",
            min_value=0.20,
            max_value=0.50,
            value=float(position_limits["max_sector_exposure"]),
            step=0.05,
            format="%.2f",
            help="单个行业的最大持仓比例"
        )
        
    # 配置预览
    st.header("当前配置概览")
    df = pd.DataFrame([
        ["初始资金", f"{initial_capital:,} {currency}"],
        ["单个持仓上限", f"{max_position:.1%}"],
        ["单个持仓下限", f"{min_position:.1%}"],
        ["最大持仓数", max_total_positions],
        ["行业敞口上限", f"{max_sector_exposure:.1%}"],
    ], columns=["配置项", "当前值"])

    # 统一转为字符串以避免 Arrow 在混合类型列上报错
    df["当前值"] = df["当前值"].astype(str)

    st.table(df.set_index("配置项"))
    
    # 保存按钮
    if st.button("保存配置"):
        try:
            update_portfolio_config({
                "initial_capital": initial_capital,
                "currency": currency,
                "position_limits": {
                    "max_position": max_position,
                    "min_position": min_position,
                    "max_total_positions": max_total_positions,
                    "max_sector_exposure": max_sector_exposure
                }
            })
            st.success("配置已更新！")
        except Exception as e:
            st.error(f"配置更新失败：{str(e)}")
            
    # 投资组合限制可视化
    st.header("仓位限制可视化")
    
    # 生成示例数据
    example_positions = np.random.uniform(
        min_position, 
        max_position, 
        min(max_total_positions, 10)
    )
    example_positions = example_positions / example_positions.sum()
    
    example_sectors = {
        "科技": 0.30,
        "金融": 0.25,
        "消费": 0.20,
        "医药": 0.15,
        "其他": 0.10
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("示例持仓分布")
        positions_df = pd.DataFrame({
            "股票": [f"股票{i+1}" for i in range(len(example_positions))],
            "持仓比例": example_positions
        })
        positions_df = positions_df.sort_values("持仓比例", ascending=True)
        
        st.bar_chart(
            positions_df.set_index("股票")["持仓比例"],
            use_container_width=True
        )
        
    with col2:
        st.subheader("示例行业分布")
        sectors_df = pd.DataFrame({
            "行业": list(example_sectors.keys()),
            "敞口": list(example_sectors.values())
        })
        
        st.bar_chart(
            sectors_df.set_index("行业")["敞口"],
            use_container_width=True
        )
