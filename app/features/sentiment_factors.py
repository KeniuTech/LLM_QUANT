"""Extended sentiment factor implementations."""
from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np

from app.core.sentiment import (
    news_sentiment_momentum,
    news_impact_score,
    market_sentiment_index,
    industry_sentiment_divergence
)
from dataclasses import dataclass
from datetime import datetime, timezone
from app.utils.data_access import DataBroker
from app.utils.db import db_session
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "sentiment_factors"}


class SentimentFactors:
    """情绪因子计算实现类。
    
    实现了一组基于新闻、市场和行业情绪的因子：
    1. 新闻情感动量 (sent_momentum)
    2. 新闻影响力 (sent_impact)
    3. 市场情绪指数 (sent_market)
    4. 行业情绪背离度 (sent_divergence)
    
    使用示例:
        calculator = SentimentFactors()
        broker = DataBroker()
        
        factors = calculator.compute_stock_factors(
            broker,
            "000001.SZ",
            "20251001"
        )
    """
    
    def __init__(self):
        """初始化情绪因子计算器"""
        self.factor_specs = {
            "sent_momentum": 20,    # 情感动量窗口
            "sent_impact": 0,       # 新闻影响力
            "sent_market": 20,      # 市场情绪窗口
            "sent_divergence": 0,   # 情绪背离度
        }
        
    def compute_stock_factors(
        self,
        broker: DataBroker,
        ts_code: str,
        trade_date: str,
    ) -> Dict[str, Optional[float]]:
        """计算单个股票的情绪因子
        
        Args:
            broker: 数据访问器
            ts_code: 股票代码
            trade_date: 交易日期
            
        Returns:
            因子名称到因子值的映射字典
        """
        results = {}
        
        try:
            # 获取历史新闻数据
            news_data = broker.get_news_data(
                ts_code,
                trade_date,
                limit=30  # 保留足够历史以计算动量
            )
            
            if not news_data:
                LOGGER.debug(
                    "无新闻数据 code=%s date=%s",
                    ts_code,
                    trade_date,
                    extra=LOG_EXTRA
                )
                return {name: None for name in self.factor_specs}
                
            # 提取序列数据
            sentiment_series = [row["sentiment"] for row in news_data]
            heat_series = [row["heat"] for row in news_data]
            entity_counts = [
                len(row["entities"].split(",")) if row["entities"] else 0
                for row in news_data
            ]
            
            # 1. 计算新闻情感动量
            results["sent_momentum"] = news_sentiment_momentum(
                sentiment_series,
                window=self.factor_specs["sent_momentum"]
            )
            
            # 2. 计算新闻影响力
            # 使用最新一条新闻的数据
            results["sent_impact"] = news_impact_score(
                sentiment=sentiment_series[0],
                heat=heat_series[0],
                entity_count=entity_counts[0]
            )
            
            # 3. 计算市场情绪指数
            # 获取成交量数据
            volume_data = broker.fetch_latest(
                ts_code,
                trade_date,
                fields=["daily_basic.volume_ratio"]
            )
            if volume_data and "daily_basic.volume_ratio" in volume_data:
                volume_ratio = volume_data["daily_basic.volume_ratio"]
                # 使用单个成交量比率值
                results["sent_market"] = market_sentiment_index(
                    sentiment_series,
                    heat_series,
                    [volume_ratio],  # 转换为列表
                    window=self.factor_specs["sent_market"]
                )
            else:
                results["sent_market"] = None
                
            # 4. 计算行业情绪背离度
            industry = broker._lookup_industry(ts_code)
            if industry:
                industry_sent = broker._derived_industry_sentiment(
                    industry,
                    trade_date
                )
                if industry_sent is not None:
                    # 获取同行业股票的情感得分
                    peer_sents = []
                    for peer in broker.get_industry_stocks(industry):
                        if peer != ts_code:
                            peer_data = broker.get_news_data(
                                peer,
                                trade_date,
                                limit=1
                            )
                            if peer_data:
                                peer_sents.append(peer_data[0]["sentiment"])
                                
                    results["sent_divergence"] = industry_sentiment_divergence(
                        industry_sent,
                        peer_sents
                    )
                else:
                    results["sent_divergence"] = None
            else:
                results["sent_divergence"] = None
                
        except Exception as e:
            LOGGER.error(
                "计算情绪因子出错 code=%s date=%s error=%s",
                ts_code,
                trade_date,
                str(e),
                exc_info=True,
                extra=LOG_EXTRA
            )
            return {name: None for name in self.factor_specs}
            
        return results
            
    def compute_batch(
        self,
        broker: DataBroker,
        ts_codes: list[str],
        trade_date: str,
        batch_size: int = 100
    ) -> None:
        """批量计算多个股票的情绪因子并保存
        
        Args:
            broker: 数据访问器
            ts_codes: 股票代码列表
            trade_date: 交易日期
            batch_size: 批处理大小
        """
        # 准备SQL语句
        columns = list(self.factor_specs.keys())
        insert_columns = ["ts_code", "trade_date", "updated_at"] + columns
        
        placeholders = ",".join("?" * len(insert_columns))
        update_clause = ", ".join(
            f"{column}=excluded.{column}" 
            for column in ["updated_at"] + columns
        )
        
        sql = (
            f"INSERT INTO factors ({','.join(insert_columns)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT(ts_code, trade_date) DO UPDATE SET {update_clause}"
        )
        
        # 获取当前时间戳
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # 分批处理
        total_processed = 0
        rows_to_persist = []
        
        for ts_code in ts_codes:
            # 计算因子
            values = self.compute_stock_factors(broker, ts_code, trade_date)
            
            # 准备数据
            if any(v is not None for v in values.values()):
                payload = [ts_code, trade_date, timestamp]
                payload.extend(values.get(col) for col in columns)
                rows_to_persist.append(payload)
            
            total_processed += 1
            if total_processed % batch_size == 0:
                LOGGER.info(
                    "情绪因子计算进度: %d/%d (%.1f%%)",
                    total_processed,
                    len(ts_codes),
                    (total_processed / len(ts_codes)) * 100,
                    extra=LOG_EXTRA
                )
        
        # 执行批量写入
        if rows_to_persist:
            with db_session() as conn:
                try:
                    conn.executemany(sql, rows_to_persist)
                    LOGGER.info(
                        "情绪因子持久化完成 total=%d",
                        len(rows_to_persist),
                        extra=LOG_EXTRA
                    )
                except Exception as e:
                    LOGGER.error(
                        "情绪因子持久化失败 error=%s",
                        str(e),
                        exc_info=True,
                        extra=LOG_EXTRA
                    )
