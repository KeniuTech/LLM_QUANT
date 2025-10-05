"""Market sentiment indicators."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
from scipy import stats

def news_sentiment_momentum(
    sentiment_series: Sequence[float],
    window: int = 20
) -> Optional[float]:
    """计算新闻情感动量指标
    
    Args:
        sentiment_series: 新闻情感得分序列，从新到旧排序
        window: 计算窗口
        
    Returns:
        情感动量得分 (-1到1)，或 None（数据不足时）
    """
    if len(sentiment_series) < window:
        return None
        
    # 计算情感趋势
    sentiment_series = np.array(sentiment_series[:window])
    slope, _, r_value, _, _ = stats.linregress(
        np.arange(len(sentiment_series)), 
        sentiment_series
    )
    
    # 结合斜率和拟合度
    trend = np.tanh(slope * 10)  # 归一化斜率
    quality = abs(r_value)  # 趋势显著性
    
    return float(trend * quality)

def news_impact_score(
    sentiment: float,
    heat: float,
    entity_count: int
) -> float:
    """计算新闻影响力得分
    
    Args:
        sentiment: 情感得分 (-1到1)
        heat: 热度得分 (0到1) 
        entity_count: 涉及实体数量
        
    Returns:
        影响力得分 (0到1)
    """
    # 新闻影响力 = 情感强度 * 热度 * 实体覆盖度
    sentiment_strength = abs(sentiment)
    entity_coverage = min(entity_count / 5, 1.0)  # 标准化实体数量
    
    return sentiment_strength * heat * (0.7 + 0.3 * entity_coverage)

def market_sentiment_index(
    sentiment_scores: Sequence[float],
    heat_scores: Sequence[float],
    volume_ratios: Sequence[float],
    window: int = 20
) -> Optional[float]:
    """计算综合市场情绪指数
    
    Args:
        sentiment_scores: 个股情感得分序列
        heat_scores: 个股热度得分序列
        volume_ratios: 个股成交量比序列
        window: 计算窗口
        
    Returns:
        市场情绪指数 (-1到1)，或 None（数据不足时）
    """
    if len(sentiment_scores) < window or \
       len(heat_scores) < window or \
       len(volume_ratios) < window:
        return None
    
    # 截取窗口数据    
    sentiment_scores = np.array(sentiment_scores[:window])
    heat_scores = np.array(heat_scores[:window])
    volume_ratios = np.array(volume_ratios[:window])
    
    # 计算带量化权重的情感得分
    volume_weights = volume_ratios / np.mean(volume_ratios)
    weighted_sentiment = sentiment_scores * volume_weights
    
    # 计算热度加权平均
    heat_weights = heat_scores / np.sum(heat_scores)
    market_mood = np.sum(weighted_sentiment * heat_weights)
    
    return float(np.tanh(market_mood))  # 压缩到[-1,1]区间

def industry_sentiment_divergence(
    industry_sentiment: float,
    peer_sentiments: Sequence[float]
) -> Optional[float]:
    """计算行业情绪背离度
    
    Args:
        industry_sentiment: 行业整体情感得分
        peer_sentiments: 成分股情感得分序列
        
    Returns:
        情绪背离度 (-1到1)，或 None（数据不足时）
    """
    if not peer_sentiments:
        return None
        
    peer_sentiments = np.array(peer_sentiments)
    peer_mean = np.mean(peer_sentiments)
    peer_std = np.std(peer_sentiments)
    
    if peer_std == 0:
        return 0.0
        
    # 计算Z分数衡量背离程度
    z_score = (industry_sentiment - peer_mean) / peer_std
    return float(np.tanh(z_score))  # 压缩到[-1,1]区间
