"""Extended factor implementations for the quant system.

This module contains additional high-quality factors that extend the default factor set.
All factors are designed to be lightweight and programmatically generated to meet 
end-to-end automated decision-making requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional, Any
import functools

import numpy as np

from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "extended_factors"}


def handle_factor_errors(func: Any) -> Any:
    """装饰器：处理因子计算过程中的错误
    
    Args:
        func: 要装饰的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[float]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 获取因子名称（如果可能）
            factor_name = "unknown"
            if len(args) > 2 and isinstance(args[1], str):
                factor_name = args[1]
            elif "factor_name" in kwargs:
                factor_name = kwargs["factor_name"]
                
            LOGGER.error(
                "计算因子出错 name=%s error=%s",
                factor_name,
                str(e),
                exc_info=True,
                extra=LOG_EXTRA
            )
            return None
    return wrapper

from app.core.indicators import momentum, rolling_mean, normalize
from app.core.technical import (
    rsi, macd, bollinger_bands, obv_momentum, price_volume_trend
)
from app.core.momentum import (
    adaptive_momentum, momentum_quality, momentum_regime
)
from app.core.volatility import (
    volatility, garch_volatility, volatility_regime,
    volume_price_correlation
)


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
    # 技术分析因子
    FactorSpec("tech_rsi_14", 14),           # 14日RSI
    FactorSpec("tech_macd_signal", 26),       # MACD信号
    FactorSpec("tech_bb_position", 20),       # 布林带位置
    FactorSpec("tech_obv_momentum", 20),      # OBV动量
    FactorSpec("tech_pv_trend", 20),          # 价量趋势
    
    # 趋势跟踪因子
    FactorSpec("trend_ma_cross", 20),         # 均线交叉信号
    FactorSpec("trend_price_channel", 20),    # 价格通道突破
    FactorSpec("trend_adx", 14),              # 平均趋向指标
    
    # 市场微观结构因子
    FactorSpec("micro_tick_direction", 5),    # 逐笔方向
    FactorSpec("micro_trade_imbalance", 10),  # 交易失衡
    
    # 波动率预测因子
    FactorSpec("vol_garch", 20),             # GARCH波动率
    FactorSpec("vol_range_pred", 10),        # 波动区间预测
    FactorSpec("vol_regime", 20),            # 波动率状态
    
    # 量价联合因子
    FactorSpec("volume_price_corr", 20),     # 量价相关性
    FactorSpec("volume_price_diverge", 10),   # 量价背离
    FactorSpec("volume_intensity", 5),        # 成交强度
    
    # 增强动量因子
    FactorSpec("momentum_adaptive", 20),      # 自适应动量
    FactorSpec("momentum_regime", 20),        # 动量区间
    FactorSpec("momentum_quality", 20),       # 动量质量,
    
    # 价格均线比率因子
    FactorSpec("price_ma_10_ratio", 10),     # 当前价格与10日均线比率
    FactorSpec("price_ma_20_ratio", 20),     # 当前价格与20日均线比率
    FactorSpec("price_ma_60_ratio", 60),     # 当前价格与60日均线比率
    
    # 成交量均线比率因子 
    FactorSpec("volume_ma_5_ratio", 5),      # 当前成交量与5日均线比率
    FactorSpec("volume_ma_20_ratio", 20),    # 当前成交量与20日均线比率
]


class ExtendedFactors:
    """扩展因子计算实现类。
    
    该类实现了一组用于量化交易的扩展因子计算。包括:
    1. 技术分析因子 (RSI, MACD, 布林带等)
    2. 趋势跟踪因子 (均线交叉等)
    3. 波动率预测因子 (GARCH, 波动率状态等)
    4. 量价联合因子 (量价相关性等)
    5. 动量强化因子 (自适应动量等)
    6. 均线比率因子 (价格/成交量均线比率)
    
    使用示例:
        calculator = ExtendedFactors()
        factor_value = calculator.compute_factor(
            "tech_rsi_14", 
            close_series,
            volume_series
        )
        all_factors = calculator.compute_all_factors(close_series, volume_series)
        normalized = calculator.normalize_factors(all_factors)
    
    属性:
        factor_specs: Dict[str, FactorSpec], 因子名称到因子规格的映射
    """
    
    def __init__(self):
        """初始化因子计算器，构建因子规格映射"""
        self.factor_specs = {spec.name: spec for spec in EXTENDED_FACTORS}
        LOGGER.info(
            "初始化因子计算器，加载因子数量: %d",
            len(self.factor_specs),
            extra=LOG_EXTRA
        )
        
    @handle_factor_errors
    def compute_factor(self, 
                      factor_name: str,
                      close_series: Sequence[float],
                      volume_series: Sequence[float]) -> Optional[float]:
        """计算单个因子值
        
        Args:
            factor_name: 因子名称，必须是已注册的因子
            close_series: 收盘价序列，从新到旧排序
            volume_series: 成交量序列，从新到旧排序
            
        Returns:
            factor_value: Optional[float], 计算得到的因子值，失败时返回None
            
        Raises:
            ValueError: 当因子名称未知或数据不足时抛出
        """
        spec = self.factor_specs.get(factor_name)
        if spec is None:
            raise ValueError(f"未知的因子名称: {factor_name}")
            
        if len(close_series) < spec.window:
            raise ValueError(
                f"数据长度不足: 需要{spec.window}，实际{len(close_series)}"
            )
                
        # 技术分析因子
        if factor_name == "tech_rsi_14":
            return rsi(close_series, 14)
            
        elif factor_name == "tech_macd_signal":
            _, signal = macd(close_series)
            return signal
            
        elif factor_name == "tech_bb_position":
            upper, lower = bollinger_bands(close_series, 20)
            pos = (close_series[0] - lower) / (upper - lower + 1e-8)
            return pos
            
        elif factor_name == "tech_obv_momentum":
            return obv_momentum(close_series, volume_series, 20)
            
        elif factor_name == "tech_pv_trend":
            return price_volume_trend(close_series, volume_series, 20)
        
        # 趋势跟踪因子
        elif factor_name == "trend_ma_cross":
            ma_5 = rolling_mean(close_series, 5)
            ma_20 = rolling_mean(close_series, 20)
            return ma_5 - ma_20
        
        # 波动率预测因子
        elif factor_name == "vol_garch":
            return garch_volatility(close_series, 20)
            
        elif factor_name == "vol_regime":
            regime, _ = volatility_regime(close_series, volume_series, 20)
            return regime
            
        # 量价联合因子
        elif factor_name == "volume_price_corr":
            return volume_price_correlation(close_series, volume_series, 20)
            
        # 增强动量因子
        elif factor_name == "momentum_adaptive":
            return adaptive_momentum(close_series, volume_series, 20)
            
        elif factor_name == "momentum_regime":
            return momentum_regime(close_series, volume_series, 20)
            
        elif factor_name == "momentum_quality":
            return momentum_quality(close_series, 20)
            
        # 均线比率因子
        elif factor_name.endswith("_ratio"):
            if "price_ma" in factor_name:
                window = int(factor_name.split("_")[2]) 
                ma = rolling_mean(close_series, window)
                return close_series[0] / ma if ma > 0 else None
                
            elif "volume_ma" in factor_name:
                window = int(factor_name.split("_")[2])
                ma = rolling_mean(volume_series, window) 
                return volume_series[0] / ma if ma > 0 else None
        
        raise ValueError(f"因子 {factor_name} 没有对应的计算实现")

    def compute_all_factors(self,
                          close_series: Sequence[float],
                          volume_series: Sequence[float]) -> Dict[str, float]:
        """计算所有已注册的扩展因子值
        
        Args:
            close_series: 收盘价序列，从新到旧排序
            volume_series: 成交量序列，从新到旧排序
            
        Returns:
            Dict[str, float]: 因子名称到因子值的映射字典，
            只包含成功计算的因子值
            
        Note:
            该方法会尝试计算所有已注册的因子，失败的因子将被忽略。
            如果所有因子计算都失败，将返回空字典。
        """
        results = {}
        success_count = 0
        total_count = len(self.factor_specs)
        
        for factor_name in self.factor_specs:
            value = self.compute_factor(factor_name, close_series, volume_series)
            if value is not None:
                results[factor_name] = value
                success_count += 1
                
        LOGGER.info(
            "因子计算完成 total=%d success=%d failed=%d",
            total_count,
            success_count,
            total_count - success_count,
            extra=LOG_EXTRA
        )
        
        return results
        
    def normalize_factors(self, 
                        factors: Dict[str, float], 
                        clip_threshold: float = 3.0) -> Dict[str, float]:
        """标准化因子值到[-1,1]区间
        
        Args:
            factors: 原始因子值字典
            clip_threshold: float, 标准化时的截断阈值，默认为3.0
            
        Returns:
            Dict[str, float]: 标准化后的因子值字典，
            只包含成功标准化的因子值
            
        Note:
            标准化过程包括:
            1. Z-score标准化
            2. 使用tanh压缩到[-1,1]区间
            3. 异常值处理（截断）
        """
        results = {}
        success_count = 0
        
        for name, value in factors.items():
            if value is not None:
                try:
                    normalized = normalize(value, clip_threshold)
                    if not np.isnan(normalized):
                        results[name] = normalized
                        success_count += 1
                except Exception as e:
                    LOGGER.warning(
                        "因子标准化失败 name=%s error=%s",
                        name,
                        str(e),
                        extra=LOG_EXTRA
                    )
        
        LOGGER.info(
            "因子标准化完成 total=%d success=%d failed=%d",
            len(factors),
            success_count,
            len(factors) - success_count,
            extra=LOG_EXTRA
        )
        
        return results