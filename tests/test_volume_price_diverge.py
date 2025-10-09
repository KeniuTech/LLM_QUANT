#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""测试volume_price_diverge因子计算逻辑"""

import numpy as np


def compute_volume_price_divergence(close_series, volume_series):
    """量价背离：价格和成交量趋势的背离程度"""
    window = 10
    if len(close_series) < window or len(volume_series) < window:
        return None
    
    # 计算价格和成交量趋势
    price_trend = sum(1 if close_series[i] > close_series[i+1] else -1 for i in range(window-1))
    volume_trend = sum(1 if volume_series[i] > volume_series[i+1] else -1 for i in range(window-1))
    
    # 计算背离程度
    divergence = price_trend * volume_trend * -1  # 反向为背离
    return np.clip(divergence / (window - 1), -1, 1)


def test_volume_price_divergence():
    """测试volume_price_divergence因子计算"""
    
    # 测试场景1：价格和成交量同向变动（无背离）
    print("测试场景1：价格和成交量同向变动")
    close_prices_1 = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    volume_prices_1 = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    result_1 = compute_volume_price_divergence(close_prices_1, volume_prices_1)
    print(f"同向变动结果: {result_1}")
    
    # 测试场景2：价格和成交量反向变动（强背离）
    print("\n测试场景2：价格和成交量反向变动")
    close_prices_2 = [109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
    volume_prices_2 = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    result_2 = compute_volume_price_divergence(close_prices_2, volume_prices_2)
    print(f"反向变动结果: {result_2}")
    
    # 测试场景3：价格上升，成交量下降（背离）
    print("\n测试场景3：价格上升，成交量下降")
    close_prices_3 = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    volume_prices_3 = [1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000]
    result_3 = compute_volume_price_divergence(close_prices_3, volume_prices_3)
    print(f"价格上涨成交量下降结果: {result_3}")
    
    # 测试场景4：价格下降，成交量上升（背离）
    print("\n测试场景4：价格下降，成交量上升")
    close_prices_4 = [109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
    volume_prices_4 = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    result_4 = compute_volume_price_divergence(close_prices_4, volume_prices_4)
    print(f"价格下降成交量上升结果: {result_4}")
    
    # 测试场景5：震荡市场（弱背离）
    print("\n测试场景5：震荡市场")
    close_prices_5 = [100, 100.5, 99.5, 101, 99, 100.5, 99.5, 100, 100.5, 99.5]
    volume_prices_5 = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    result_5 = compute_volume_price_divergence(close_prices_5, volume_prices_5)
    print(f"震荡市场结果: {result_5}")


if __name__ == "__main__":
    test_volume_price_divergence()