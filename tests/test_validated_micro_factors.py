import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.features.extended_factors import ExtendedFactors
from app.features.validation import validate_factor_value

def test_validated_micro_factors():
    """测试修正后的微观结构因子验证"""
    # 创建因子引擎
    engine = ExtendedFactors()
    
    # 测试场景：极端行情 - 大幅波动（会产生超出原配置范围的值）
    close_prices = np.array([100, 110, 90, 120, 80, 130, 70, 140, 60, 150, 50], dtype=float)
    volume_prices = np.array([1000, 2000, 500, 2500, 300, 3000, 200, 3500, 100, 4000, 50], dtype=float)
    
    # 计算因子值
    tick_result = engine.compute_factor("micro_tick_direction", close_prices, volume_prices)
    imbalance_result = engine.compute_factor("micro_trade_imbalance", close_prices, volume_prices)
    
    print(f"micro_tick_direction: {tick_result}")
    print(f"micro_trade_imbalance: {imbalance_result}")
    
    # 验证因子值是否在新配置的范围内
    ts_code = "000001.SZ"
    trade_date = "20230101"
    
    validated_tick = validate_factor_value("micro_tick_direction", tick_result, ts_code, trade_date)
    validated_imbalance = validate_factor_value("micro_trade_imbalance", imbalance_result, ts_code, trade_date)
    
    print(f"\n验证结果:")
    print(f"micro_tick_direction验证: {validated_tick} (原始值: {tick_result})")
    print(f"micro_trade_imbalance验证: {validated_imbalance} (原始值: {imbalance_result})")
    
    # 测试各种边界值
    print(f"\n边界值测试:")
    test_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 100.0, 150.0]
    
    for value in test_values:
        validated = validate_factor_value("micro_tick_direction", value, ts_code, trade_date)
        print(f"micro_tick_direction值 {value}: {'通过' if validated is not None else '拒绝'}")
    
    for value in test_values:
        validated = validate_factor_value("micro_trade_imbalance", value, ts_code, trade_date)
        print(f"micro_trade_imbalance值 {value}: {'通过' if validated is not None else '拒绝'}")

if __name__ == "__main__":
    test_validated_micro_factors()