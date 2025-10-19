"""Tests for sentiment factor computation."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List

import pytest

from app.features.sentiment_factors import SentimentFactors
from app.utils.data_access import DataBroker


class MockDataBroker:
    """Mock DataBroker for testing."""
    
    def get_news_data(
        self,
        ts_code: str,
        trade_date: str,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """模拟新闻数据"""
        if ts_code == "000001.SZ":
            return [
                {
                    "sentiment": 0.8,
                    "heat": 0.6,
                    "entities": "公司A,行业B,概念C"
                },
                {
                    "sentiment": 0.6,
                    "heat": 0.4,
                    "entities": "公司A,概念D"
                }
            ]
        return []
        
    def get_stock_data(
        self,
        ts_code: str,
        trade_date: str,
        fields: List[str],
        limit: int = 1
    ) -> List[Dict[str, Any]]:
        """模拟股票数据"""
        if ts_code == "000001.SZ":
            return [
                {"daily_basic.volume_ratio": 1.2},
                {"daily_basic.volume_ratio": 1.1}
            ]
        return []
        
    def _lookup_industry(self, ts_code: str) -> str:
        """模拟行业查询"""
        if ts_code == "000001.SZ":
            return "银行"
        return ""
        
    def _derived_industry_sentiment(
        self,
        industry: str,
        trade_date: str
    ) -> float:
        """模拟行业情绪"""
        if industry == "银行":
            return 0.5
        return 0.0
        
    def get_industry_stocks(self, industry: str) -> List[str]:
        """模拟行业成分股"""
        if industry == "银行":
            return ["000001.SZ", "600000.SH"]
        return []


def test_compute_stock_factors():
    """测试股票情绪因子计算"""
    calculator = SentimentFactors()
    broker = MockDataBroker()
    
    # 测试有数据的情况
    factors = calculator.compute_stock_factors(
        broker,
        "000001.SZ",
        "20251001"
    )
    
    assert all(name in factors for name in ("sent_momentum", "sent_impact", "sent_market", "sent_divergence"))
    assert all(value is None for value in factors.values())
    
    # 测试无数据的情况
    factors = calculator.compute_stock_factors(
        broker,
        "000002.SZ",
        "20251001"
    )
    
    assert all(v is None for v in factors.values())
    
def test_compute_batch(tmp_path):
    """测试批量计算功能"""
    from app.data.schema import initialize_database
    from app.utils.config import get_config
    
    # 配置测试数据库
    config = get_config()
    config.db_path = tmp_path / "test.db"
    
    # 初始化数据库
    initialize_database()
    
    calculator = SentimentFactors()
    broker = MockDataBroker()
    
    # 测试批量计算
    ts_codes = ["000001.SZ", "000002.SZ", "600000.SH"]
    calculator.compute_batch(broker, ts_codes, "20251001")
    
    # 验证未写入任何情绪因子数据
    from app.utils.db import db_session
    with db_session() as conn:
        rows = conn.execute(
            "SELECT * FROM factors WHERE trade_date = ?",
            ("20251001",)
        ).fetchall()
        
    assert rows == []
