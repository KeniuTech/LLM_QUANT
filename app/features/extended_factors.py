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
from app.features.validation import validate_factor_value


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
    
    # 估值因子
    FactorSpec("val_ps_score", 0),           # PS估值分数
    FactorSpec("val_multiscore", 0),         # 多维估值分数
    FactorSpec("val_dividend_score", 0),     # 股息评分
    
    # 市场状态因子
    FactorSpec("market_regime", 20),         # 市场状态
    FactorSpec("trend_strength", 20),        # 趋势强度
    
    # 风险因子
    FactorSpec("risk_penalty", 20),          # 风险惩罚因子
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
        # 关闭初始化日志打印
        # LOGGER.info(
        #     "初始化因子计算器，加载因子数量: %d",
        #     len(self.factor_specs),
        #     extra=LOG_EXTRA
        # )
        
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
            return macd(close_series, 12, 26, 9)
            
        elif factor_name == "tech_bb_position":
            return bollinger_bands(close_series, 20)
            
        elif factor_name == "tech_obv_momentum":
            return obv_momentum(close_series, volume_series, 20)
            
        elif factor_name == "tech_pv_trend":
            return price_volume_trend(close_series, volume_series, 20)
        
        # 趋势跟踪因子
        elif factor_name == "trend_ma_cross":
            ma_5 = rolling_mean(close_series, 5)
            ma_20 = rolling_mean(close_series, 20)
            return ma_5 - ma_20
            
        elif factor_name == "trend_price_channel":
            # 价格通道突破因子：当前价格相对于通道的位置
            window = 20
            high_channel = max(close_series[:window])
            low_channel = min(close_series[:window])
            if high_channel != low_channel:
                return (close_series[0] - low_channel) / (high_channel - low_channel)
            return 0.0
            
        elif factor_name == "trend_adx":
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
        
        # 市场微观结构因子
        elif factor_name == "micro_tick_direction":
            # 简化的逐笔方向：基于最近价格变动
            window = 5
            if len(close_series) < window + 1:
                return None
            
            # 计算价格变动方向
            directions = [1 if close_series[i] > close_series[i+1] else -1 for i in range(window)]
            return sum(directions) / window
            
        elif factor_name == "micro_trade_imbalance":
            # 交易失衡：基于价格和成交量的联合分析
            window = 10
            if len(close_series) < window + 1 or len(volume_series) < window + 1:
                return None
            
            # 计算价格变动和成交量变动
            price_changes = [close_series[i] - close_series[i+1] for i in range(window)]
            volume_changes = [volume_series[i] - volume_series[i+1] for i in range(window)]
            
            # 计算交易失衡指标
            imbalance = sum(price_changes[i] * volume_changes[i] for i in range(window))
            return imbalance / (window * np.mean(volume_series[:window]) + 1e-8)
        
        # 波动率预测因子
        elif factor_name == "vol_garch":
            return garch_volatility(close_series, 20)
            
        elif factor_name == "vol_range_pred":
            # 波动区间预测：基于历史价格区间
            window = 10
            if len(close_series) < window + 5:
                return None
            
            # 计算历史价格区间
            ranges = []
            for i in range(5):  # 使用最近5个窗口
                if i + window < len(close_series):
                    price_range = max(close_series[i:i+window]) - min(close_series[i:i+window])
                    ranges.append(price_range / close_series[i])
            
            if ranges:
                # 使用历史区间的75分位数作为预测
                return np.percentile(ranges, 75)
            return None
            
        elif factor_name == "vol_regime":
            return volatility_regime(close_series, volume_series, 20)
            
        # 量价联合因子
        elif factor_name == "volume_price_corr":
            return volume_price_correlation(close_series, volume_series, 20)
            
        elif factor_name == "volume_price_diverge":
            # 量价背离：价格和成交量趋势的背离程度
            window = 10
            if len(close_series) < window or len(volume_series) < window:
                return None
            
            # 计算价格和成交量趋势
            price_trend = sum(1 if close_series[i] > close_series[i+1] else -1 for i in range(window-1))
            volume_trend = sum(1 if volume_series[i] > volume_series[i+1] else -1 for i in range(window-1))
            
            # 计算背离程度
            divergence = price_trend * volume_trend * -1  # 反向为背离
            return np.clip(divergence / (window - 1), -1, 1)
            
        elif factor_name == "volume_intensity":
            # 成交强度：基于成交量和价格变动的加权指标
            window = 5
            if len(close_series) < window + 1 or len(volume_series) < window + 1:
                return None
            
            # 计算价格变动
            price_changes = [abs(close_series[i] - close_series[i+1]) for i in range(window)]
            
            # 计算成交量加权的价格变动
            weighted_changes = sum(price_changes[i] * volume_series[i] for i in range(window))
            total_volume = sum(volume_series[:window])
            
            if total_volume > 0:
                intensity = weighted_changes / (total_volume * np.mean(close_series[:window]) + 1e-8)
                return np.clip(intensity * 100, -100, 100)  # 归一化到合理范围
            return None
            
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
        
        # 估值因子
        elif factor_name == "val_ps_score":
            # PS估值分数：基于PS比率的估值指标
            # 假设PS比率越低，估值越有吸引力
            if len(close_series) < 10:
                return None
            
            # 简化的PS估值：基于价格与历史均值的比较
            current_price = close_series[0]
            avg_price = np.mean(close_series[:10])
            
            if avg_price > 0:
                # 当前价格相对于历史均值的偏离程度
                ps_ratio = current_price / avg_price
                # 标准化到[-1, 1]区间
                return np.clip((1.0 - ps_ratio) / 2.0, -1, 1)
            return None
            
        elif factor_name == "val_multiscore":
            # 多维估值分数：综合多个估值维度的评分
            if len(close_series) < 20:
                return None
                
            # 使用价格动量、波动率和相对强度作为估值代理
            momentum_5 = momentum(close_series, 5)
            momentum_20 = momentum(close_series, 20)
            
            # 计算波动率（手动实现，避免依赖外部函数）
            if len(close_series) >= 20:
                returns = [close_series[i] / close_series[i+1] - 1 for i in range(19)]  # 修正索引范围
                volatility_20 = np.std(returns) if returns else 0
            else:
                volatility_20 = 0
            
            # 综合评分：动量正向，波动率负向
            if volatility_20 > 0:
                score = (momentum_5 + momentum_20) / (2 * volatility_20)
                return np.clip(score, -1, 1)
            return None
            
        elif factor_name == "val_dividend_score":
            # 股息评分：基于价格稳定性和趋势的股息吸引力评分
            if len(close_series) < 20:
                return None
                
            # 计算价格稳定性（低波动率）
            if len(close_series) >= 20:
                returns = [close_series[i] / close_series[i+1] - 1 for i in range(19)]  # 修正索引范围
                vol = np.std(returns) if returns else 0
            else:
                vol = 0
            
            # 计算趋势强度
            trend = momentum(close_series, 20)
            
            # 股息吸引力：稳定性正向，趋势正向
            stability_score = 1.0 - np.clip(vol, 0, 1)
            trend_score = np.clip(trend, -1, 1)
            
            return (stability_score + trend_score) / 2.0
        
        # 市场状态因子
        elif factor_name == "market_regime":
            # 市场状态：基于价格和成交量的市场状态判断
            if len(close_series) < 20 or len(volume_series) < 20:
                return None
                
            # 价格趋势
            price_trend = momentum(close_series, 20)
            # 成交量趋势
            volume_trend = momentum(volume_series, 20)
            
            # 市场状态：牛市（价格↑成交量↑）、熊市（价格↓成交量↓）、
            # 震荡市（价格平稳成交量平稳）、背离市（价格成交量反向）
            regime_score = price_trend * volume_trend
            return np.clip(regime_score, -1, 1)
            
        elif factor_name == "trend_strength":
            # 趋势强度：基于价格变动的趋势强度度量
            if len(close_series) < 20:
                return None
                
            # 计算不同时间窗口的动量
            momentum_5 = momentum(close_series, 5)
            momentum_10 = momentum(close_series, 10)
            momentum_20 = momentum(close_series, 20)
            
            # 趋势强度：短期、中期、长期动量的一致性
            trend_strength = (momentum_5 + momentum_10 + momentum_20) / 3.0
            return np.clip(trend_strength, -1, 1)
        
        # 风险因子
        elif factor_name == "risk_penalty":
            # 风险惩罚因子：基于波动率和异常价格的综合风险度量
            if len(close_series) < 20:
                return None
                
            # 波动率风险
            if len(close_series) >= 20:
                returns = [close_series[i] / close_series[i+1] - 1 for i in range(19)]  # 修正索引范围
                vol_risk = np.std(returns) if returns else 0
            else:
                vol_risk = 0
            
            # 价格异常风险（相对于均值的偏离）
            avg_price = np.mean(close_series[:20])
            if avg_price > 0:
                price_deviation = abs(close_series[0] / avg_price - 1.0)
            else:
                price_deviation = 0
            
            # 综合风险评分
            risk_score = (vol_risk + price_deviation) / 2.0
            return np.clip(risk_score, 0, 1)  # 风险因子范围为[0, 1]
        
        raise ValueError(f"因子 {factor_name} 没有对应的计算实现")

    def compute_all_factors(self,
                          close_series: List[float],
                          volume_series: List[float],
                          ts_code: str,
                          trade_date: str) -> Dict[str, float | None]:
        """计算所有扩展因子
        
        Args:
            close_series: 收盘价序列
            volume_series: 成交量序列
            ts_code: 股票代码
            trade_date: 交易日期
            
        Returns:
            因子名称到因子值的映射
        """
        results = {}
        
        for factor_spec in EXTENDED_FACTORS:
            try:
                factor_name = factor_spec.name
                factor_value = self.compute_factor(factor_name, close_series, volume_series)
                
                # 验证因子值
                if factor_value is not None:
                    # 使用真实的 ts_code 和 trade_date 进行验证
                    validate_factor_value(factor_name, factor_value, ts_code, trade_date)
                
                results[factor_name] = factor_value
                
            except Exception as e:
                LOGGER.debug(
                    "因子计算失败 factor=%s ts_code=%s date=%s err=%s",
                    factor_spec.name,
                    ts_code,
                    trade_date,
                    str(e),
                    extra=LOG_EXTRA,
                )
                results[factor_spec.name] = None
        
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