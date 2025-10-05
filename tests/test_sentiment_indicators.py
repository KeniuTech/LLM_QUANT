"""Tests for market sentiment indicators."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.sentiment import (
    news_sentiment_momentum,
    news_impact_score,
    market_sentiment_index,
    industry_sentiment_divergence
)


def test_news_sentiment_momentum():
    # 生成测试数据
    window = 20
    uptrend = np.linspace(-0.5, 0.5, window)  # 上升趋势
    downtrend = np.linspace(0.5, -0.5, window)  # 下降趋势
    flat = np.zeros(window)  # 平稳趋势
    
    # 测试上升趋势
    result = news_sentiment_momentum(uptrend)
    assert result is not None
    assert result > 0
    
    # 测试下降趋势
    result = news_sentiment_momentum(downtrend)
    assert result is not None
    assert result < 0
    
    # 测试平稳趋势
    result = news_sentiment_momentum(flat)
    assert result is not None
    assert abs(result) < 0.1
    
    # 测试数据不足
    result = news_sentiment_momentum(uptrend[:10])
    assert result is None


def test_news_impact_score():
    # 测试典型场景
    score = news_impact_score(sentiment=0.8, heat=0.6, entity_count=3)
    assert 0 <= score <= 1
    assert score > news_impact_score(sentiment=0.4, heat=0.6, entity_count=3)
    
    # 测试边界情况
    assert news_impact_score(sentiment=0, heat=0.5, entity_count=1) == 0
    assert 0 < news_impact_score(sentiment=1, heat=1, entity_count=10) <= 1
    
    # 测试实体数量影响
    low_entity = news_impact_score(sentiment=0.5, heat=0.5, entity_count=1)
    high_entity = news_impact_score(sentiment=0.5, heat=0.5, entity_count=5)
    assert high_entity > low_entity


def test_market_sentiment_index():
    window = 20
    
    # 生成测试数据
    sentiment_scores = np.random.uniform(-1, 1, window)
    heat_scores = np.random.uniform(0, 1, window)
    volume_ratios = np.random.uniform(0.5, 2, window)
    
    # 测试正常计算
    result = market_sentiment_index(
        sentiment_scores,
        heat_scores,
        volume_ratios
    )
    assert result is not None
    assert -1 <= result <= 1
    
    # 测试数据缺失
    result = market_sentiment_index(
        sentiment_scores[:10],
        heat_scores,
        volume_ratios
    )
    assert result is None


def test_industry_sentiment_divergence():
    # 测试显著背离
    high_divergence = industry_sentiment_divergence(
        industry_sentiment=0.8,
        peer_sentiments=[-0.2, -0.1, 0, 0.1]
    )
    assert high_divergence is not None
    assert high_divergence > 0
    
    # 测试一致性好
    low_divergence = industry_sentiment_divergence(
        industry_sentiment=0.1,
        peer_sentiments=[0, 0.1, 0.2]
    )
    assert low_divergence is not None
    assert abs(low_divergence) < abs(high_divergence)
    
    # 测试空数据
    assert industry_sentiment_divergence(0.5, []) is None
