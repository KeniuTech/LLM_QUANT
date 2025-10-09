import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.features.extended_factors import ExtendedFactorEngine


class TestTrendAdx(unittest.TestCase):
    """测试trend_adx因子计算"""

    def setUp(self):
        """初始化测试环境"""
        self.engine = ExtendedFactorEngine()

    def test_trend_adx_positive_values(self):
        """测试trend_adx因子返回正值"""
        # 模拟一个上涨趋势的数据
        close_prices = [115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
        volume_prices = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        
        result = self.engine.compute_factor("trend_adx", close_prices, volume_prices)
        
        # 验证结果不为None
        self.assertIsNotNone(result)
        # 验证结果在0-100范围内
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 100)
        print(f"上涨趋势trend_adx值: {result}")

    def test_trend_adx_negative_values(self):
        """测试trend_adx因子处理下跌趋势"""
        # 模拟一个下跌趋势的数据
        close_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
        volume_prices = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        
        result = self.engine.compute_factor("trend_adx", close_prices, volume_prices)
        
        # 验证结果不为None
        self.assertIsNotNone(result)
        # 验证结果在0-100范围内
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 100)
        print(f"下跌趋势trend_adx值: {result}")

    def test_trend_adx_sideways_market(self):
        """测试trend_adx因子处理震荡市场"""
        # 模拟一个震荡市场的数据
        close_prices = [100, 100.5, 99.5, 101, 99, 100.5, 99.5, 100, 100.5, 99.5, 100, 100.5, 99.5, 100, 100.5, 99.5]
        volume_prices = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        
        result = self.engine.compute_factor("trend_adx", close_prices, volume_prices)
        
        # 验证结果不为None
        self.assertIsNotNone(result)
        # 验证结果在0-100范围内
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 100)
        print(f"震荡市场trend_adx值: {result}")

    def test_trend_adx_insufficient_data(self):
        """测试数据不足时返回None"""
        # 提供少于15个数据点
        close_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        volume_prices = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        
        result = self.engine.compute_factor("trend_adx", close_prices, volume_prices)
        
        # 验证结果为None
        self.assertIsNone(result)
        print("数据不足时正确返回None")

    def test_trend_adx_flat_market(self):
        """测试trend_adx因子处理平盘市场"""
        # 模拟一个价格保持不变的市场
        close_prices = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        volume_prices = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        
        result = self.engine.compute_factor("trend_adx", close_prices, volume_prices)
        
        # 验证结果不为None
        self.assertIsNotNone(result)
        # 验证结果在0-100范围内
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 100)
        print(f"平盘市场trend_adx值: {result}")


if __name__ == '__main__':
    unittest.main()