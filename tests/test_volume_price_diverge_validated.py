import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.features.extended_factors import ExtendedFactorEngine

def test_volume_price_diverge_validated():
    """测试修复后的volume_price_diverge因子"""
    # 创建因子引擎
    engine = ExtendedFactorEngine()
    
    # 测试场景1：价格和成交量同向变动（强正相关）
    print("测试场景1：价格和成交量同向变动")
    close_prices1 = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], dtype=float)
    volume_prices1 = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000], dtype=float)
    result1 = engine.compute_factor("volume_price_diverge", close_prices1, volume_prices1)
    print(f"同向变动结果: {result1}")
    
    # 测试场景2：价格和成交量反向变动（强负相关）
    print("\n测试场景2：价格和成交量反向变动")
    close_prices2 = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], dtype=float)
    volume_prices2 = np.array([2000, 1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000], dtype=float)
    result2 = engine.compute_factor("volume_price_diverge", close_prices2, volume_prices2)
    print(f"反向变动结果: {result2}")
    
    # 测试场景3：价格上升，成交量下降
    print("\n测试场景3：价格上升，成交量下降")
    close_prices3 = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120], dtype=float)
    volume_prices3 = np.array([1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500], dtype=float)
    result3 = engine.compute_factor("volume_price_diverge", close_prices3, volume_prices3)
    print(f"价格上涨成交量下降结果: {result3}")
    
    # 测试场景4：价格下降，成交量上升
    print("\n测试场景4：价格下降，成交量上升")
    close_prices4 = np.array([120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100], dtype=float)
    volume_prices4 = np.array([500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000], dtype=float)
    result4 = engine.compute_factor("volume_price_diverge", close_prices4, volume_prices4)
    print(f"价格下降成交量上升结果: {result4}")
    
    # 测试场景5：震荡市场
    print("\n测试场景5：震荡市场")
    close_prices5 = np.array([100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100], dtype=float)
    volume_prices5 = np.array([1000, 1100, 1000, 1100, 1000, 1100, 1000, 1100, 1000, 1100, 1000], dtype=float)
    result5 = engine.compute_factor("volume_price_diverge", close_prices5, volume_prices5)
    print(f"震荡市场结果: {result5}")
    
    # 验证所有结果都在合理范围内
    print("\n验证结果范围:")
    all_results = [result1, result2, result3, result4, result5]
    for i, result in enumerate(all_results, 1):
        if result is not None:
            assert -1.0 <= result <= 1.0, f"测试场景{i}的结果超出范围: {result}"
            print(f"测试场景{i}结果 {result} 在合理范围内")
        else:
            print(f"测试场景{i}结果为 None")
    
    print("\n所有测试通过！")

if __name__ == "__main__":
    test_volume_price_diverge_validated()