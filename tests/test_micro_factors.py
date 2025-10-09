import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.features.extended_factors import ExtendedFactors

def test_micro_factors():
    """测试微观结构因子的实际取值范围"""
    # 创建因子引擎
    engine = ExtendedFactors()
    
    # 测试场景1：持续上涨行情
    print("测试场景1：持续上涨行情")
    close_prices1 = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], dtype=float)
    volume_prices1 = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000], dtype=float)
    result1_tick = engine.compute_factor("micro_tick_direction", close_prices1, volume_prices1)
    result1_imbalance = engine.compute_factor("micro_trade_imbalance", close_prices1, volume_prices1)
    print(f"micro_tick_direction: {result1_tick}")
    print(f"micro_trade_imbalance: {result1_imbalance}")
    
    # 测试场景2：持续下跌行情
    print("\n测试场景2：持续下跌行情")
    close_prices2 = np.array([110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100], dtype=float)
    volume_prices2 = np.array([2000, 1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000], dtype=float)
    result2_tick = engine.compute_factor("micro_tick_direction", close_prices2, volume_prices2)
    result2_imbalance = engine.compute_factor("micro_trade_imbalance", close_prices2, volume_prices2)
    print(f"micro_tick_direction: {result2_tick}")
    print(f"micro_trade_imbalance: {result2_imbalance}")
    
    # 测试场景3：震荡行情
    print("\n测试场景3：震荡行情")
    close_prices3 = np.array([100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100], dtype=float)
    volume_prices3 = np.array([1000, 1100, 1000, 1100, 1000, 1100, 1000, 1100, 1000, 1100, 1000], dtype=float)
    result3_tick = engine.compute_factor("micro_tick_direction", close_prices3, volume_prices3)
    result3_imbalance = engine.compute_factor("micro_trade_imbalance", close_prices3, volume_prices3)
    print(f"micro_tick_direction: {result3_tick}")
    print(f"micro_trade_imbalance: {result3_imbalance}")
    
    # 测试场景4：极端行情 - 大幅波动
    print("\n测试场景4：极端行情 - 大幅波动")
    close_prices4 = np.array([100, 110, 90, 120, 80, 130, 70, 140, 60, 150, 50], dtype=float)
    volume_prices4 = np.array([1000, 2000, 500, 2500, 300, 3000, 200, 3500, 100, 4000, 50], dtype=float)
    result4_tick = engine.compute_factor("micro_tick_direction", close_prices4, volume_prices4)
    result4_imbalance = engine.compute_factor("micro_trade_imbalance", close_prices4, volume_prices4)
    print(f"micro_tick_direction: {result4_tick}")
    print(f"micro_trade_imbalance: {result4_imbalance}")
    
    # 测试场景5：极端行情 - 大量小额波动
    print("\n测试场景5：极端行情 - 大量小额波动")
    close_prices5 = np.array([100, 100.1, 99.9, 100.2, 99.8, 100.3, 99.7, 100.4, 99.6, 100.5, 99.5], dtype=float)
    volume_prices5 = np.array([1000, 1100, 900, 1200, 800, 1300, 700, 1400, 600, 1500, 500], dtype=float)
    result5_tick = engine.compute_factor("micro_tick_direction", close_prices5, volume_prices5)
    result5_imbalance = engine.compute_factor("micro_trade_imbalance", close_prices5, volume_prices5)
    print(f"micro_tick_direction: {result5_tick}")
    print(f"micro_trade_imbalance: {result5_imbalance}")
    
    # 收集所有结果
    all_tick_results = [result1_tick, result2_tick, result3_tick, result4_tick, result5_tick]
    all_imbalance_results = [result1_imbalance, result2_imbalance, result3_imbalance, result4_imbalance, result5_imbalance]
    
    print("\n=== 结果分析 ===")
    print("micro_tick_direction因子结果范围:")
    valid_tick_results = [r for r in all_tick_results if r is not None]
    if valid_tick_results:
        print(f"  最小值: {min(valid_tick_results)}")
        print(f"  最大值: {max(valid_tick_results)}")
        print(f"  平均值: {np.mean(valid_tick_results)}")
    
    print("\nmicro_trade_imbalance因子结果范围:")
    valid_imbalance_results = [r for r in all_imbalance_results if r is not None]
    if valid_imbalance_results:
        print(f"  最小值: {min(valid_imbalance_results)}")
        print(f"  最大值: {max(valid_imbalance_results)}")
        print(f"  平均值: {np.mean(valid_imbalance_results)}")

if __name__ == "__main__":
    test_micro_factors()