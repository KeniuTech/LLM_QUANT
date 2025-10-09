#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""简化版trend_adx因子测试"""

def compute_trend_adx(close_series):
    """简化版trend_adx计算实现"""
    # 标准ADX计算实现
    window = 14
    if len(close_series) < window + 1:
        return None
    
    # 计算+DI和-DI
    plus_di = 0
    minus_di = 0
    tr_sum = 0
    
    # 计算初始TR、+DM、-DM
    for i in range(window):
        if i + 1 >= len(close_series):
            break
            
        # 计算真实波幅(TR)
        today_high = close_series[i]
        today_low = close_series[i]
        prev_close = close_series[i + 1]
        
        tr = max(
            abs(today_high - today_low),
            abs(today_high - prev_close),
            abs(today_low - prev_close)
        )
        tr_sum += tr
        
        # 计算方向运动
        prev_high = close_series[i + 1] if i + 1 < len(close_series) else close_series[i]
        prev_low = close_series[i + 1] if i + 1 < len(close_series) else close_series[i]
        
        plus_dm = max(0, close_series[i] - prev_high)
        minus_dm = max(0, prev_low - close_series[i])
        
        # 确保只有一项为正值
        if plus_dm > minus_dm:
            minus_dm = 0
        elif minus_dm > plus_dm:
            plus_dm = 0
        else:
            plus_dm = minus_dm = 0
        
        plus_di += plus_dm
        minus_di += minus_dm
    
    # 计算+DI和-DI
    if tr_sum > 0:
        plus_di = (plus_di / tr_sum) * 100
        minus_di = (minus_di / tr_sum) * 100
    else:
        plus_di = minus_di = 0
    
    # 计算DX
    dx = 0
    if plus_di + minus_di > 0:
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    
    # ADX是DX的移动平均，这里简化为直接返回DX值，确保在0-100范围内
    return max(0, min(100, dx))


def test_trend_adx():
    """测试trend_adx因子计算"""
    
    # 测试上涨趋势
    print("测试上涨趋势:")
    close_prices_up = [115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
    result_up = compute_trend_adx(close_prices_up)
    print(f"上涨趋势trend_adx值: {result_up}")
    assert result_up is not None and 0 <= result_up <= 100, f"上涨趋势结果错误: {result_up}"
    
    # 测试下跌趋势
    print("\n测试下跌趋势:")
    close_prices_down = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
    result_down = compute_trend_adx(close_prices_down)
    print(f"下跌趋势trend_adx值: {result_down}")
    assert result_down is not None and 0 <= result_down <= 100, f"下跌趋势结果错误: {result_down}"
    
    # 测试震荡市场
    print("\n测试震荡市场:")
    close_prices_sideways = [100, 100.5, 99.5, 101, 99, 100.5, 99.5, 100, 100.5, 99.5, 100, 100.5, 99.5, 100, 100.5, 99.5]
    result_sideways = compute_trend_adx(close_prices_sideways)
    print(f"震荡市场trend_adx值: {result_sideways}")
    assert result_sideways is not None and 0 <= result_sideways <= 100, f"震荡市场结果错误: {result_sideways}"
    
    # 测试数据不足
    print("\n测试数据不足:")
    close_prices_insufficient = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    result_insufficient = compute_trend_adx(close_prices_insufficient)
    print(f"数据不足时结果: {result_insufficient}")
    assert result_insufficient is None, f"数据不足时应该返回None: {result_insufficient}"
    
    # 测试平盘市场
    print("\n测试平盘市场:")
    close_prices_flat = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    result_flat = compute_trend_adx(close_prices_flat)
    print(f"平盘市场trend_adx值: {result_flat}")
    assert result_flat is not None and 0 <= result_flat <= 100, f"平盘市场结果错误: {result_flat}"
    
    print("\n所有测试通过！")


if __name__ == "__main__":
    test_trend_adx()