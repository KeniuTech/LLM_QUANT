"""Momentum related indicators."""
from typing import List, Optional, Sequence, Tuple
import numpy as np
from scipy import stats

def adaptive_momentum(prices: Sequence[float], 
                     volumes: Sequence[float],
                     lookback: int = 20) -> Optional[float]:
    """计算自适应动量指标。
    
    基于价格趋势和成交量变化自适应调整动量周期。
    
    Args:
        prices: 价格序列，从新到旧排序
        volumes: 成交量序列，从新到旧排序
        lookback: 基准回看期数
        
    Returns:
        动量值，或None（数据不足时）
    """
    if len(prices) < lookback or len(volumes) < lookback:
        return None
        
    # 计算价格收益率和成交量变化率
    price_returns = np.array([(prices[i] - prices[i+1])/prices[i+1] 
                            for i in range(len(prices)-1)])
    vol_changes = np.array([(volumes[i] - volumes[i+1])/volumes[i+1] 
                           for i in range(len(volumes)-1)])
    
    if len(price_returns) < lookback:
        return None
        
    # 根据成交量变化调整权重
    weights = np.exp(vol_changes[:lookback])
    weights = weights / np.sum(weights)
    
    # 计算加权动量
    momentum = np.sum(price_returns[:lookback] * weights)
    
    return momentum

def momentum_quality(prices: Sequence[float],
                    window: int = 20) -> Optional[float]:
    """计算动量质量指标。
    
    基于价格趋势的一致性和强度评估动量质量。
    
    Args:
        prices: 价格序列，从新到旧排序
        window: 计算窗口
        
    Returns:
        动量质量指标(-1到1)，或None（数据不足时）
    """
    if len(prices) < window:
        return None
        
    # 计算收益率
    returns = np.array([(prices[i] - prices[i+1])/prices[i+1] 
                       for i in range(len(prices)-1)])
    
    if len(returns) < window:
        return None
    
    # 计算趋势一致性
    returns = returns[:window]
    sign_consistency = np.mean(np.sign(returns))
    
    # 计算趋势强度
    trend_strength = abs(np.sum(returns)) / (np.sum(abs(returns)) + 1e-8)
    
    # 组合指标
    quality = sign_consistency * trend_strength
    
    return np.clip(quality, -1, 1)

def momentum_regime(prices: Sequence[float],
                   volumes: Sequence[float], 
                   window: int = 20) -> Optional[float]:
    """识别动量趋势状态。
    
    结合价格趋势和成交量特征识别动量状态。
    
    Args:
        prices: 价格序列，从新到旧排序
        volumes: 成交量序列，从新到旧排序
        window: 计算窗口
        
    Returns:
        动量状态指标(-1到1)，或None（数据不足时）
    """
    if len(prices) < window or len(volumes) < window:
        return None
        
    # 计算价格动量
    mom = adaptive_momentum(prices, volumes, window)
    if mom is None:
        return None
        
    # 计算动量质量
    quality = momentum_quality(prices, window)
    if quality is None:
        return None
        
    # 组合指标
    regime = 0.7 * np.sign(mom) * abs(quality) + 0.3 * quality
    
    return np.clip(regime, -1, 1)
