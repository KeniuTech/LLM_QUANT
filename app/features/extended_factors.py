"""Extended factor implementations for the quant system.

This module contains additional high-quality factors that extend the default factor set.
All factors are designed to be lightweight and programmatically generated to meet 
end-to-end automated decision-making requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from app.core.indicators import momentum, volatility, rolling_mean, normalize


@dataclass
class FactorSpec:
    """Specification for a factor computation.
    
    Attributes:
        name: Factor name identifier
        window: Required lookback window (0 for snapshot-only factors)
    """
    name: str
    window: int


# Extended factors focusing on momentum, value, and liquidity signals
EXTENDED_FACTORS: List[FactorSpec] = [
    # 增强动量因子
    FactorSpec("mom_10_30", 0),  # 10日与30日动量差
    FactorSpec("mom_5_20_rank", 0),  # 相对排名动量因子
    FactorSpec("mom_dynamic", 0),  # 动态窗口动量因子
    # 波动率相关因子
    FactorSpec("volat_5", 5),  # 短期波动率
    FactorSpec("volat_ratio", 0),  # 长短期波动率比率
    # 换手率扩展因子
    FactorSpec("turn_60", 60),  # 长期换手率
    FactorSpec("turn_rank", 0),  # 换手率相对排名
    # 价格均线比率因子
    FactorSpec("price_ma_10_ratio", 0),  # 当前价格与10日均线比率
    FactorSpec("price_ma_20_ratio", 0),  # 当前价格与20日均线比率
    FactorSpec("price_ma_60_ratio", 0),  # 当前价格与60日均线比率
    # 成交量均线比率因子
    FactorSpec("volume_ma_5_ratio", 0),  # 当前成交量与5日均线比率
    FactorSpec("volume_ma_20_ratio", 0),  # 当前成交量与20日均线比率
    # 高级估值因子
    FactorSpec("val_ps_score", 0),  # PS估值评分
    FactorSpec("val_multiscore", 0),  # 综合估值评分
    FactorSpec("val_dividend_score", 0),  # 股息率估值评分
    # 市场状态因子
    FactorSpec("market_regime", 0),  # 市场状态因子
    FactorSpec("trend_strength", 0),  # 趋势强度因子
]


def compute_extended_factor_values(
    close_series: Sequence[float],
    volume_series: Sequence[float],
    turnover_series: Sequence[float],
    latest_fields: Dict[str, float],
) -> Dict[str, float]:
    """Compute values for extended factors.
    
    Args:
        close_series: Closing prices series (most recent first)
        volume_series: Trading volume series (most recent first)
        turnover_series: Turnover rate series (most recent first)
        latest_fields: Latest available fields including valuation ratios
        
    Returns:
        Dictionary mapping factor names to computed values
    """
    if not close_series:
        return {}
        
    results: Dict[str, float] = {}
    
    # 增强动量因子
    # 10日与30日动量差
    if len(close_series) >= 30:
        mom_10 = momentum(close_series, 10)
        mom_30 = momentum(close_series, 30)
        if mom_10 is not None and mom_30 is not None:
            results["mom_10_30"] = mom_10 - mom_30
    
    # 相对排名动量因子
    # 这里需要市场数据来计算相对排名，暂时使用简化版本
    if len(close_series) >= 20:
        mom_20 = momentum(close_series, 20)
        if mom_20 is not None:
            # 简化处理：将动量标准化
            results["mom_5_20_rank"] = min(1.0, max(0.0, (mom_20 + 0.2) / 0.4))
    
    # 动态窗口动量因子
    # 根据波动率动态调整窗口
    if len(close_series) >= 20:
        volat_20 = volatility(close_series, 20)
        mom_20 = momentum(close_series, 20)
        if volat_20 is not None and mom_20 is not None and volat_20 > 0:
            # 波动率调整后的动量
            results["mom_dynamic"] = mom_20 / volat_20
    
    # 波动率相关因子
    # 短期波动率
    if len(close_series) >= 5:
        results["volat_5"] = volatility(close_series, 5)
    
    # 长短期波动率比率
    if len(close_series) >= 20 and len(close_series) >= 5:
        volat_5 = volatility(close_series, 5)
        volat_20 = volatility(close_series, 20)
        if volat_5 is not None and volat_20 is not None and volat_20 > 0:
            results["volat_ratio"] = volat_5 / volat_20
    
    # 换手率扩展因子
    # 长期换手率
    if len(turnover_series) >= 60:
        results["turn_60"] = rolling_mean(turnover_series, 60)
    
    # 换手率相对排名
    if len(turnover_series) >= 20:
        turn_20 = rolling_mean(turnover_series, 20)
        if turn_20 is not None:
            # 简化处理：将换手率标准化
            results["turn_rank"] = min(1.0, max(0.0, turn_20 / 5.0))  # 假设5%为高换手率
    
    # 价格均线比率因子
    if len(close_series) >= 10:
        ma_10 = rolling_mean(close_series, 10)
        if ma_10 is not None and ma_10 > 0:
            results["price_ma_10_ratio"] = close_series[0] / ma_10
    
    if len(close_series) >= 20:
        ma_20 = rolling_mean(close_series, 20)
        if ma_20 is not None and ma_20 > 0:
            results["price_ma_20_ratio"] = close_series[0] / ma_20
    
    if len(close_series) >= 60:
        ma_60 = rolling_mean(close_series, 60)
        if ma_60 is not None and ma_60 > 0:
            results["price_ma_60_ratio"] = close_series[0] / ma_60
    
    # 成交量均线比率因子
    if len(volume_series) >= 5:
        vol_ma_5 = rolling_mean(volume_series, 5)
        if vol_ma_5 is not None and vol_ma_5 > 0:
            results["volume_ma_5_ratio"] = volume_series[0] / vol_ma_5
    
    if len(volume_series) >= 20:
        vol_ma_20 = rolling_mean(volume_series, 20)
        if vol_ma_20 is not None and vol_ma_20 > 0:
            results["volume_ma_20_ratio"] = volume_series[0] / vol_ma_20
    
    # 高级估值因子
    ps = latest_fields.get("daily_basic.ps")
    if ps is not None and ps > 0:
        # PS估值评分
        results["val_ps_score"] = 2.5 / (2.5 + ps)  # 使用2.5作为scale参数
    
    pe = latest_fields.get("daily_basic.pe")
    pb = latest_fields.get("daily_basic.pb")
    # 综合估值评分
    scores = []
    if pe is not None and pe > 0:
        scores.append(12.0 / (12.0 + pe))  # PE评分
    if pb is not None and pb > 0:
        scores.append(2.5 / (2.5 + pb))   # PB评分
    if ps is not None and ps > 0:
        scores.append(2.5 / (2.5 + ps))   # PS评分
    
    if scores:
        results["val_multiscore"] = sum(scores) / len(scores)
    
    dv_ratio = latest_fields.get("daily_basic.dv_ratio")
    if dv_ratio is not None:
        # 股息率估值评分
        results["val_dividend_score"] = min(1.0, max(0.0, dv_ratio / 5.0))  # 假设5%为高股息率
    
    # 市场状态因子
    # 简单的市场状态指标：基于价格位置
    if len(close_series) >= 60:
        ma_60 = rolling_mean(close_series, 60)
        if ma_60 is not None and ma_60 > 0:
            results["market_regime"] = close_series[0] / ma_60
    
    # 趋势强度因子
    if len(close_series) >= 20:
        mom_20 = momentum(close_series, 20)
        volat_20 = volatility(close_series, 20)
        if mom_20 is not None and volat_20 is not None and volat_20 > 0:
            # 趋势强度：动量与波动率的比率
            results["trend_strength"] = abs(mom_20) / volat_20
    
    return results