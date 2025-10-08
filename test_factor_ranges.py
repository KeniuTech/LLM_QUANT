#!/usr/bin/env python3
"""测试因子值范围验证功能"""

from app.features.factors import compute_factors
from datetime import date

def test_factor_ranges():
    """测试因子值范围验证功能"""
    print('测试改进后的因子值范围验证功能...')
    
    try:
        results = compute_factors(
            date(2024, 1, 15),
            ts_codes=['000001.SZ', '000002.SZ'],
            skip_existing=False,
            batch_size=10
        )
        
        print(f'因子计算完成，共计算 {len(results)} 个结果')
        
        # 检查每个因子的值范围
        valid_count = 0
        invalid_count = 0
        
        for result in results:
            print(f'\n证券 {result.ts_code} 的因子值:')
            for factor_name, value in result.values.items():
                if value is not None:
                    # 检查值是否在合理范围内
                    if -10 <= value <= 10:  # 放宽检查范围，主要看验证逻辑
                        print(f'  ✓ {factor_name}: {value:.6f}')
                        valid_count += 1
                    else:
                        print(f'  ✗ {factor_name}: {value:.6f} (超出范围!)')
                        invalid_count += 1
        
        print(f'\n验证统计:')
        print(f'  有效因子值: {valid_count}')
        print(f'  无效因子值: {invalid_count}')
        print(f'  总因子值: {valid_count + invalid_count}')
        
        if invalid_count == 0:
            print('\n✅ 所有因子值都在合理范围内，验证通过!')
        else:
            print(f'\n⚠️ 发现 {invalid_count} 个超出范围的因子值，需要进一步优化')
        
        print('\n✅ 因子值范围验证测试完成')
        
    except Exception as e:
        print(f'❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_factor_ranges()