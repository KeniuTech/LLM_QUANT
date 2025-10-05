"""Advanced volatility and volume-price indicators."""
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model

def volatility(prices: Sequence[float], window: int = 20) -> Optional[float]:
    """计算历史波动率。
    
    Args:
        prices: 价格序列，从新到旧排序
        window: 计算窗口
        
    Returns:
        年化波动率，或None（数据不足时）
    """
    if len(prices) < window:
        return None
    
    returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
    if len(returns) < window:
        return None
        
    vol = np.std(returns[:window]) * np.sqrt(252)
    return vol

def garch_volatility(returns: Sequence[float], 
                    lookback: int = 20) -> Optional[float]:
    """使用GARCH(1,1)模型估计波动率。
    
    Args:
        returns: 收益率序列，从新到旧排序
        lookback: 回看期数
        
    Returns:
        预测的波动率，或None（数据不足时）
    """
    if len(returns) < lookback:
        return None
        
    # 使用简化的EWMA代替GARCH
    data = np.array(returns[:lookback])
    decay = 0.94
    
    # 计算历史波动率
    hist_vol = np.std(data) * np.sqrt(252)  # 年化
    
    # 计算EWMA波动率
    variance = hist_vol * hist_vol
    for i in range(1, len(data)):
        innovation = data[i-1] * data[i-1]
        variance = decay * variance + (1 - decay) * innovation
        
    return np.sqrt(variance)

def volatility_regime(prices: Sequence[float],
                     volumes: Sequence[float],
                     lookback: int = 20) -> Optional[float]:
    """识别波动率状态。
    
    Args:
        prices: 价格序列，从新到旧排序
        volumes: 成交量序列，从新到旧排序
        lookback: 回看期数
        
    Returns:
        波动率状态指标(-1到1)，或None（数据不足时）
    """
    if len(prices) < lookback or len(volumes) < lookback:
        return None
        
    # 计算收益率和成交量变化
    returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(lookback-1)]
    vol_changes = [(volumes[i] - volumes[i+1]) / volumes[i+1] for i in range(lookback-1)]
    
    # 计算波动率特征
    vol = np.std(returns) * np.sqrt(252)
    vol_of_vol = np.std(vol_changes)
    
    # 结合价格波动和成交量波动判断状态
    if vol > 0 and vol_of_vol > 0:
        regime = 0.5 * (vol / np.mean(abs(returns)) - 1) + \
                0.5 * (vol_of_vol / np.mean(abs(vol_changes)) - 1)
        return np.clip(regime, -1, 1)
    return 0.0

def range_volatility_prediction(prices: Sequence[float],
                              window: int = 10) -> Optional[float]:
    """基于价格区间的波动率预测。
    
    Args:
        prices: 价格序列，从新到旧排序
        window: 预测窗口
        
    Returns:
        预测的波动率，或None（数据不足时）
    """
    if len(prices) < window + 5:  # 需要额外的数据来建立预测
        return None
        
    # 计算历史真实波动率
    ranges = []
    for i in range(len(prices)-window):
        price_range = max(prices[i:i+window]) - min(prices[i:i+window])
        ranges.append(price_range / prices[i])
        
    if not ranges:
        return None
        
    # 使用历史区间分布预测未来波动率
    pred_vol = np.percentile(ranges, 75)  # 使用75分位数作为预测
    return pred_vol

def volume_price_correlation(prices: Sequence[float],
                           volumes: Sequence[float],
                           window: int = 20) -> Optional[float]:
    """计算量价相关性。
    
    Args:
        prices: 价格序列，从新到旧排序
        volumes: 成交量序列，从新到旧排序
        window: 计算窗口
        
    Returns:
        量价相关系数(-1到1)，或None（数据不足时）
    """
    if len(prices) < window + 1 or len(volumes) < window + 1:
        return None
        
    # 计算价格和成交量的变化率
    price_changes = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(window)]
    volume_changes = [(volumes[i] - volumes[i+1]) / volumes[i+1] for i in range(window)]
    
    # 计算相关系数
    if len(price_changes) >= 2:
        corr, _ = stats.pearsonr(price_changes, volume_changes)
        return corr
    return None

def volume_price_divergence(prices: Sequence[float],
                          volumes: Sequence[float],
                          window: int = 10) -> Optional[float]:
    """检测量价背离。
    
    Args:
        prices: 价格序列，从新到旧排序
        volumes: 成交量序列，从新到旧排序
        window: 检测窗口
        
    Returns:
        背离强度(-1到1)，或None（数据不足时）
    """
    if len(prices) < window or len(volumes) < window:
        return None
        
    # 计算价格和成交量趋势
    price_trend = sum(1 if prices[i] > prices[i+1] else -1 for i in range(window-1))
    volume_trend = sum(1 if volumes[i] > volumes[i+1] else -1 for i in range(window-1))
    
    # 计算背离程度
    divergence = price_trend * volume_trend * -1  # 反向为背离
    return np.clip(divergence / (window - 1), -1, 1)

def volume_intensity(volumes: Sequence[float],
                    prices: Sequence[float],
                    window: int = 5) -> Optional[float]:
    """计算成交强度。
    
    Args:
        volumes: 成交量序列，从新到旧排序
        prices: 价格序列，从新到旧排序
        window: 计算窗口
        
    Returns:
        成交强度指标，或None（数据不足时）
    """
    if len(volumes) < window + 1 or len(prices) < window + 1:
        return None
        
    # 计算价格变动
    price_changes = [abs(prices[i] - prices[i+1]) for i in range(window)]
    
    # 计算成交量加权的价格变动
    weighted_changes = sum(price_changes[i] * volumes[i] for i in range(window))
    total_volume = sum(volumes[:window])
    
    if total_volume > 0:
        intensity = weighted_changes / (total_volume * np.mean(prices[:window]))
        return np.clip(intensity * 100, -100, 100)  # 归一化到合理范围
    return None
