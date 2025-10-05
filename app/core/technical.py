"""Technical analysis indicators implementation."""
from typing import List, Optional, Sequence
import numpy as np
from .indicators import rolling_mean

def rsi(prices: Sequence[float], period: int = 14) -> Optional[float]:
    """计算相对强弱指标(RSI)。
    
    Args:
        prices: 价格序列，从新到旧排序
        period: RSI周期，默认14天
        
    Returns:
        RSI值 (0-100) 或 None（数据不足时）
    """
    if len(prices) < period + 1:
        return None
        
    # 计算价格变化
    deltas = [prices[i] - prices[i+1] for i in range(len(prices)-1)]
    deltas = deltas[:period]  # 只使用所需周期的数据
    
    gain = [delta if delta > 0 else 0 for delta in deltas]
    loss = [-delta if delta < 0 else 0 for delta in deltas]
    
    avg_gain = sum(gain) / period
    avg_loss = sum(loss) / period
    
    if avg_loss == 0:
        return 100.0
        
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(prices: Sequence[float], 
         fast_period: int = 12,
         slow_period: int = 26,
         signal_period: int = 9) -> Optional[float]:
    """计算MACD信号。
    
    Args:
        prices: 价格序列，从新到旧排序
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
        
    Returns:
        MACD柱状值，或None（数据不足时）
    """
    if len(prices) < slow_period + signal_period:
        return None
        
    # 计算快慢线EMA
    fast_ema = _ema(prices, fast_period)
    slow_ema = _ema(prices, slow_period)
    if fast_ema is None or slow_ema is None:
        return None
        
    # 计算MACD线
    macd_line = fast_ema - slow_ema
    
    # 计算信号线
    macd_values = []
    for i in range(len(prices) - slow_period + 1):
        fast_ema = _ema(prices[i:], fast_period)
        slow_ema = _ema(prices[i:], slow_period)
        if fast_ema is not None and slow_ema is not None:
            macd_values.append(fast_ema - slow_ema)
            
    if len(macd_values) < signal_period:
        return None
        
    signal_line = _ema(macd_values, signal_period)
    if signal_line is None:
        return None
        
    # 返回MACD柱状图值
    return macd_line - signal_line

def _ema(data: Sequence[float], period: int) -> Optional[float]:
    """计算指数移动平均。"""
    if len(data) < period:
        return None
        
    multiplier = 2 / (period + 1)
    ema = data[0]
    for price in data[1:period]:
        ema = (price - ema) * multiplier + ema
    return ema

def bollinger_bands(prices: Sequence[float], 
                   period: int = 20,
                   std_dev: float = 2.0) -> Optional[float]:
    """计算布林带位置。
    
    Args:
        prices: 价格序列，从新到旧排序
        period: 移动平均周期
        std_dev: 标准差倍数
        
    Returns:
        价格在布林带中的位置(-1到1)，或None（数据不足时）
    """
    if len(prices) < period:
        return None
        
    # 获取周期内数据
    price_window = prices[:period]
    
    # 计算移动平均和标准差
    ma = sum(price_window) / period
    std = np.std(price_window)
    
    # 计算布林带
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    
    # 计算当前价格在带中的位置
    current_price = prices[0]
    band_width = upper - lower
    if band_width == 0:
        return 0
        
    position = 2 * (current_price - lower) / band_width - 1
    return max(-1, min(1, position))  # 确保值在-1到1之间

def obv_momentum(volumes: Sequence[float], 
                prices: Sequence[float],
                period: int = 20) -> Optional[float]:
    """计算能量潮(OBV)动量。
    
    Args:
        volumes: 成交量序列，从新到旧排序
        prices: 价格序列，从新到旧排序
        period: 计算周期
        
    Returns:
        OBV动量值，或None（数据不足时）
    """
    if len(volumes) < period + 1 or len(prices) < period + 1:
        return None
        
    # 计算OBV序列
    obv = [0.0]  # 初始OBV值
    for i in range(1, period):
        price_change = prices[i-1] - prices[i]
        if price_change > 0:
            obv.append(obv[-1] + volumes[i-1])
        elif price_change < 0:
            obv.append(obv[-1] - volumes[i-1])
        else:
            obv.append(obv[-1])
            
    # 计算OBV动量（当前值与N日前的差值）
    obv_momentum = (obv[0] - obv[-1]) / sum(volumes[:period])
    return obv_momentum

def price_volume_trend(prices: Sequence[float],
                      volumes: Sequence[float],
                      period: int = 20) -> Optional[float]:
    """计算价量趋势指标。
    
    Args:
        prices: 价格序列，从新到旧排序
        volumes: 成交量序列，从新到旧排序
        period: 计算周期
        
    Returns:
        价量趋势值，或None（数据不足时）
    """
    if len(prices) < period or len(volumes) < period:
        return None
        
    # 计算价格变动和成交量的乘积
    pv_values = []
    for i in range(period-1):
        price_change = (prices[i] - prices[i+1]) / prices[i+1]
        pv_values.append(price_change * volumes[i])
        
    # 使用移动平均平滑处理
    pv_trend = sum(pv_values) / sum(volumes[:period])
    return pv_trend
