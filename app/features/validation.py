"""Factor validation and quality control utilities."""
from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "factor_validation"}

# 因子值范围限制配置 - 基于物理规律和实际数据特征
FACTOR_LIMITS = {
    # 动量类因子：收益率相关，限制在 ±50% (实际收益率很少超过这个范围)
    "mom_": (-0.5, 0.5),
    "momentum_": (-0.5, 0.5),
    
    # 波动率类因子：年化波动率，限制在 0-200% (考虑极端市场情况)
    "volat_": (0, 2.0),
    "vol_": (0, 2.0),
    
    # 换手率类因子：日换手率，限制在 0-100% (实际换手率通常在这个范围内)
    "turn_": (0, 1.0),
    
    # 估值评分类因子：标准化评分，限制在 -3到3 (Z-score标准化范围)
    "val_": (-3.0, 3.0),
    
    # 量价类因子：成交量比率，限制在 0-10倍
    "volume_": (0, 10.0),
    "volume_ratio": (0, 10.0),
    
    # 市场状态类因子：标准化状态指标，限制在 -3到3
    "market_": (-3.0, 3.0),
    
    # 技术指标类因子：具体技术指标的范围限制
    "tech_rsi": (0, 100.0),           # RSI指标范围 0-100
    "tech_macd": (-0.5, 0.5),         # MACD信号范围
    "tech_bb": (-3.0, 3.0),            # 布林带位置，标准差倍数
    "tech_obv": (-10.0, 10.0),         # OBV动量标准化
    "tech_pv": (-1.0, 1.0),            # 量价趋势相关性
    
    # 趋势类因子：趋势强度指标
    "trend_": (-3.0, 3.0),
    "trend_ma": (-0.5, 0.5),           # 均线交叉
    "trend_adx": (0, 100.0),           # ADX趋势强度 0-100
    
    # 微观结构类因子：标准化微观指标
    "micro_": (-1.0, 1.0),
    
    # 情绪类因子：情绪指标标准化
    "sent_": (-1.0, 1.0),
    
    # 风险类因子：风险惩罚因子
    "risk_": (0, 1.0),
    
    # 价格比率类因子：价格与均线比率，限制在 0.5-2.0 (50%-200%)
    "price_ma_": (0.5, 2.0),
    
    # 成交量比率类因子：成交量与均线比率，限制在 0.1-10.0
    "volume_ma_": (0.1, 10.0),
}

def validate_factor_value(
    name: str, 
    value: float,
    ts_code: str,
    trade_date: str
) -> Optional[float]:
    """验证单个因子值是否在合理范围内。
    
    Args:
        name: 因子名称
        value: 因子值
        ts_code: 股票代码
        trade_date: 交易日期
    
    Returns:
        如果因子值有效则返回原值，否则返回 None
    """
    if value is None:
        return None
        
    # 检查是否为有限数值
    if not np.isfinite(value):
        LOGGER.warning(
            "因子值非有限数值 factor=%s value=%f ts_code=%s date=%s",
            name, value, ts_code, trade_date, 
            extra=LOG_EXTRA
        )
        return None
        
    # 优先检查精确匹配的因子名称
    exact_matches = {
        # 技术指标精确范围
        "tech_rsi_14": (0, 100.0),           # RSI指标范围 0-100
        "tech_macd_signal": (-0.5, 0.5),     # MACD信号范围
        "tech_bb_position": (-3.0, 3.0),     # 布林带位置，标准差倍数
        "tech_obv_momentum": (-10.0, 10.0),  # OBV动量标准化
        "tech_pv_trend": (-1.0, 1.0),       # 量价趋势相关性
        
        # 趋势指标精确范围
        "trend_adx": (0, 100.0),             # ADX趋势强度 0-100
        "trend_ma_cross": (-1.0, 1.0),       # 均线交叉
        "trend_price_channel": (0, 1.0),     # 价格通道位置
        
        # 波动率指标精确范围
        "vol_garch": (0, 0.5),               # GARCH波动率预测，限制在50%以内
        "vol_range_pred": (0, 0.2),          # 波动率范围预测，限制在20%以内
        "vol_regime": (0, 1.0),              # 波动率状态，0-1之间
        
        # 微观结构精确范围
        "micro_tick_direction": (0, 1.0),    # 买卖方向比例
        "micro_trade_imbalance": (-1.0, 1.0), # 交易不平衡度
        
        # 情绪指标精确范围
        "sent_impact": (0, 1.0),             # 情绪影响度
        "sent_divergence": (-1.0, 1.0),      # 情绪分歧度
    }
    
    # 检查精确匹配
    if name in exact_matches:
        min_val, max_val = exact_matches[name]
        if min_val <= value <= max_val:
            return value
        else:
            LOGGER.warning(
                "因子值超出精确范围 factor=%s value=%f range=[%f,%f] ts_code=%s date=%s",
                name, value, min_val, max_val, ts_code, trade_date,
                extra=LOG_EXTRA
            )
            return None
    
    # 检查前缀模式匹配
    for prefix, (min_val, max_val) in FACTOR_LIMITS.items():
        if name.startswith(prefix):
            if min_val <= value <= max_val:
                return value
            else:
                LOGGER.warning(
                    "因子值超出前缀范围 factor=%s value=%f range=[%f,%f] ts_code=%s date=%s",
                    name, value, min_val, max_val, ts_code, trade_date,
                    extra=LOG_EXTRA
                )
                return None
    
    # 如果没有匹配，使用更严格的默认范围
    default_min, default_max = -5.0, 5.0
    if default_min <= value <= default_max:
        LOGGER.debug(
            "因子使用默认范围验证通过 factor=%s value=%f ts_code=%s date=%s",
            name, value, ts_code, trade_date,
            extra=LOG_EXTRA
        )
        return value
    else:
        LOGGER.warning(
            "因子值超出默认范围 factor=%s value=%f range=[%f,%f] ts_code=%s date=%s",
            name, value, default_min, default_max, ts_code, trade_date,
            extra=LOG_EXTRA
        )
        return None

def detect_outliers(
    values: Dict[str, float],
    ts_code: str,
    trade_date: str
) -> Dict[str, float]:
    """检测和处理因子值中的异常值。
    
    Args:
        values: 因子值字典
        ts_code: 股票代码
        trade_date: 交易日期
    
    Returns:
        处理后的因子值字典
    """
    result = {}
    
    for name, value in values.items():
        validated = validate_factor_value(name, value, ts_code, trade_date)
        if validated is not None:
            result[name] = validated
            
    return result

def check_data_sufficiency(
    ts_code: str,
    trade_date: str,
    min_days: int = 60
) -> bool:
    """验证因子计算所需数据是否充分。
    
    Args:
        ts_code: 股票代码
        trade_date: 交易日期
        min_days: 最少需要的历史数据天数
        
    Returns:
        数据是否充分
    """
    from app.utils.data_access import DataBroker
    
    broker = DataBroker()
    
    # 检查历史收盘价数据
    close_series = broker.fetch_series("daily", "close", ts_code, trade_date, min_days)
    # 计算有效值的数量
    valid_values = [val for _, val in close_series if val is not None and isinstance(val, (int, float))]
    if len(valid_values) < min_days:
        LOGGER.warning(
            "历史数据不足 ts_code=%s date=%s min_days=%d actual=%d", 
            ts_code, trade_date, min_days, len(valid_values),
            extra=LOG_EXTRA
        )
        return False

    # 检查日期点数据完整性
    latest_fields = broker.fetch_latest(
        ts_code,
        trade_date,
        ["daily.close", "daily_basic.turnover_rate", "daily_basic.pe", "daily_basic.pb"]
    )
    required_fields = {"daily.close", "daily_basic.turnover_rate"}
    
    for field in required_fields:
        if latest_fields.get(field) is None:
            LOGGER.warning(
                "缺少必需字段 field=%s ts_code=%s date=%s",
                field, ts_code, trade_date,
                extra=LOG_EXTRA
            )
            return False
            
    return True


def check_series_sufficiency(
    data: Sequence,
    required_length: int,
    field_name: str,
    ts_code: str,
    trade_date: str
) -> bool:
    """检查数据序列是否满足计算要求。
    
    Args:
        data: 数据序列
        required_length: 所需最小长度
        field_name: 字段名称
        ts_code: 股票代码
        trade_date: 交易日期
    
    Returns:
        数据是否足够
    """
    if len(data) < required_length:
        LOGGER.warning(
            "数据长度不足 field=%s required=%d actual=%d ts_code=%s date=%s",
            field_name, required_length, len(data), ts_code, trade_date,
            extra=LOG_EXTRA
        )
        return False
        
    # 检查数据有效性
    valid_count = sum(1 for x in data if x is not None and np.isfinite(x))
    if valid_count < required_length:
        LOGGER.warning(
            "有效数据不足 field=%s required=%d valid=%d ts_code=%s date=%s",
            field_name, required_length, valid_count, ts_code, trade_date,
            extra=LOG_EXTRA
        )
        return False
        
    return True
