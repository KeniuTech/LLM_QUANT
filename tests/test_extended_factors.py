"""Test extended factor calculations."""

import numpy as np
import pytest

from app.features.extended_factors import ExtendedFactors, FactorSpec


def test_technical_factors():
    """测试技术分析因子计算。"""
    
    # 准备测试数据
    close = np.array([100, 102, 98, 103, 97, 105, 102, 98, 103, 99,
                     100, 102, 98, 103, 97, 105, 102, 98, 103, 99])
    volume = np.array([1000, 1200, 800, 1500, 600, 2000, 1100, 900, 1300, 700,
                      1000, 1200, 800, 1500, 600, 2000, 1100, 900, 1300, 700])
                      
    factors = ExtendedFactors()
    
    # 测试RSI计算
    rsi_spec = FactorSpec("tech_rsi_14", 14)
    rsi = factors.compute_factor(rsi_spec, close, volume)
    assert rsi is not None
    assert 0 <= rsi <= 100
    
    # 测试MACD计算
    macd_spec = FactorSpec("tech_macd_signal", 26)
    macd = factors.compute_factor(macd_spec, close, volume)
    assert macd is not None
    
    # 测试布林带位置
    bb_spec = FactorSpec("tech_bb_position", 20)
    bb = factors.compute_factor(bb_spec, close, volume)
    assert bb is not None
    assert 0 <= bb <= 1
    
    # 测试OBV动量
    obv_spec = FactorSpec("tech_obv_momentum", 20)
    obv = factors.compute_factor(obv_spec, close, volume)
    assert obv is not None

def test_momentum_factors():
    """测试动量因子计算。"""
    
    # 准备测试数据
    close = np.array([100, 102, 98, 103, 97, 105, 102, 98, 103, 99,
                     100, 102, 98, 103, 97, 105, 102, 98, 103, 99])
    volume = np.array([1000, 1200, 800, 1500, 600, 2000, 1100, 900, 1300, 700,
                      1000, 1200, 800, 1500, 600, 2000, 1100, 900, 1300, 700])
                      
    factors = ExtendedFactors()
    
    # 测试自适应动量
    adaptive_spec = FactorSpec("momentum_adaptive", 20)
    adaptive = factors.compute_factor(adaptive_spec, close, volume)
    assert adaptive is not None
    
    # 测试动量质量
    quality_spec = FactorSpec("momentum_quality", 20) 
    quality = factors.compute_factor(quality_spec, close, volume)
    assert quality is not None
    assert -1 <= quality <= 1
    
    # 测试动量状态
    regime_spec = FactorSpec("momentum_regime", 20)
    regime = factors.compute_factor(regime_spec, close, volume)
    assert regime is not None
    assert -1 <= regime <= 1
    
def test_volatility_factors():
    """测试波动率因子计算。"""
    
    # 准备测试数据
    close = np.array([100, 102, 98, 103, 97, 105, 102, 98, 103, 99,
                     100, 102, 98, 103, 97, 105, 102, 98, 103, 99])
    volume = np.array([1000, 1200, 800, 1500, 600, 2000, 1100, 900, 1300, 700,
                      1000, 1200, 800, 1500, 600, 2000, 1100, 900, 1300, 700])
                      
    factors = ExtendedFactors()
    
    # 测试GARCH波动率
    garch_spec = FactorSpec("vol_garch", 20)
    garch = factors.compute_factor(garch_spec, close, volume)
    assert garch is not None
    assert garch >= 0
    
    # 测试波动率状态
    regime_spec = FactorSpec("vol_regime", 20)
    regime = factors.compute_factor(regime_spec, close, volume)
    assert regime is not None
    
def test_compute_all_factors():
    """测试计算所有因子。"""
    
    # 准备测试数据
    close = np.array([100, 102, 98, 103, 97, 105, 102, 98, 103, 99,
                     100, 102, 98, 103, 97, 105, 102, 98, 103, 99])
    volume = np.array([1000, 1200, 800, 1500, 600, 2000, 1100, 900, 1300, 700,
                      1000, 1200, 800, 1500, 600, 2000, 1100, 900, 1300, 700])
                      
    factors = ExtendedFactors()
    
    # 计算所有因子
    results = factors.compute_all_factors(close, volume)
    
    # 验证结果
    assert len(results) > 0
    for value in results.values():
        assert isinstance(value, (int, float))
        assert not np.isnan(value)
        
def test_data_validation():
    """测试数据验证。"""
    
    # 构造无效数据
    close = np.array([100, np.nan, 98, 103])  # 包含NaN
    volume = np.array([1000, 1200, -800, 1500])  # 包含负值
    
    factors = ExtendedFactors()
    
    # 测试单个因子计算
    rsi_spec = FactorSpec("tech_rsi_14", 14)
    rsi = factors.compute_factor(rsi_spec, close, volume)
    assert rsi is None  # 数据无效应返回None
    
    # 测试整体计算
    results = factors.compute_all_factors(close, volume)
    assert len(results) == 0  # 无效数据应返回空字典
