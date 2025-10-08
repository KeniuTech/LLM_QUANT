"""Factor validation and quality control utilities."""
from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "factor_validation"}

# 因子值范围限制配置
FACTOR_LIMITS = {
    # 动量类因子限制在 ±50%
    "mom_": (-0.5, 0.5),
    # 波动率类因子限制在 0-30%
    "volat_": (0, 0.3),
    # 换手率类因子限制在 0-100%
    "turn_": (0, 1.0),
    # 估值评分类因子限制在 0-1
    "val_": (0, 1.0),
    # 量价类因子
    "volume_": (0, 5.0),
    # 市场状态类因子
    "market_": (-1.0, 1.0),
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
        
    # 根据因子类型应用不同的限制
    for prefix, (min_val, max_val) in FACTOR_LIMITS.items():
        if name.startswith(prefix):
            if value < min_val or value > max_val:
                LOGGER.warning(
                    "因子值超出范围 factor=%s value=%f range=[%f,%f] ts_code=%s date=%s",
                    name, value, min_val, max_val, ts_code, trade_date,
                    extra=LOG_EXTRA
                )
                return None
            break
            
    return value

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
    
    # 检查历史收盘价数据
    close_series = DataBroker.get_daily_price(ts_code, end_date=trade_date)
    if len(close_series) < min_days:
        LOGGER.warning(
            "历史数据不足 ts_code=%s date=%s min_days=%d actual=%d", 
            ts_code, trade_date, min_days, len(close_series),
            extra=LOG_EXTRA
        )
        return False

    # 检查日期点数据完整性
    latest_fields = DataBroker.get_latest_fields(
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
