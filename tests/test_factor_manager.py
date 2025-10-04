"""
因子管理器测试
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_manager import FactorLibrary, CustomFactorManager, FactorCalculator

class TestFactorManager(unittest.TestCase):
    """因子管理器测试类"""

    def setUp(self):
        """测试初始化"""
        self.factor_lib = FactorLibrary()
        self.custom_manager = CustomFactorManager()
        self.calculator = FactorCalculator()

        # 创建测试数据
        self.test_data = self._create_test_data()

    def _create_test_data(self):
        """创建测试数据"""
        symbols = ['000001', '000002', '600036']
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')

        data = []
        for symbol in symbols:
            for date in dates:
                base_price = 10.0
                price = base_price + np.random.randn() * 0.5

                data.append({
                    'symbol': symbol,
                    'date': date,
                    'open': price * (1 + np.random.randn() * 0.01),
                    'high': price * (1 + abs(np.random.randn()) * 0.02),
                    'low': price * (1 - abs(np.random.randn()) * 0.02),
                    'close': price,
                    'volume': max(1000000, int(np.random.randn() * 1000000 + 5000000)),
                })

        df = pd.DataFrame(data)
        # 添加收益率
        df['return_1d'] = df.groupby('symbol')['close'].pct_change()
        return df

    def test_factor_library(self):
        """测试因子库"""
        # 测试获取因子
        rsi_factor = self.factor_lib.get_factor('RSI')
        self.assertIsNotNone(rsi_factor)
        self.assertEqual(rsi_factor.name, 'RSI')

        # 测试列出因子
        all_factors = self.factor_lib.list_factors()
        self.assertGreater(len(all_factors), 0)
        self.assertIn('RSI', all_factors)

        # 测试因子计算
        result = self.factor_lib.calculate_factor('RSI', self.test_data)
        self.assertIn('RSI', result.columns)

        # 测试因子统计
        stats = self.factor_lib.get_factor_statistics('RSI', self.test_data)
        self.assertIn('count', stats)
        self.assertIn('mean', stats)

    def test_custom_factor_manager(self):
        """测试自定义因子管理器"""
        # 测试添加因子
        test_code = """
def calculate_factor(data):
    return data.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
"""
        success = self.custom_manager.add_factor_from_code(
            'TEST_MA5', '测试5日均线', 'test', test_code
        )
        self.assertTrue(success)

        # 测试获取因子
        factor = self.custom_manager.get_factor('TEST_MA5')
        self.assertIsNotNone(factor)
        self.assertEqual(factor.name, 'TEST_MA5')

        # 测试执行因子
        result = self.custom_manager.execute_factor('TEST_MA5', self.test_data)
        self.assertIn('TEST_MA5', result.columns)

        # 测试因子验证
        validation = self.custom_manager.validate_factor_code(test_code)
        self.assertTrue(validation['valid'])

    def test_factor_calculator(self):
        """测试因子计算器"""
        # 测试单因子计算
        result = self.calculator.calculate_single_factor('RSI', self.test_data)
        self.assertIn('RSI', result.columns)

        # 测试批量因子计算
        factor_names = ['RSI', 'MA5']
        result = self.calculator.calculate_multiple_factors(factor_names, self.test_data)
        for factor_name in factor_names:
            self.assertIn(factor_name, result.columns)

        # 测试因子相关性
        correlation = self.calculator.get_factor_correlation(result, factor_names)
        self.assertEqual(correlation.shape[0], len(factor_names))
        self.assertEqual(correlation.shape[1], len(factor_names))

        # 测试因子排序
        ranked_result = self.calculator.rank_factor_values(result, 'RSI')
        self.assertIn('RSI_rank', ranked_result.columns)

        # 测试因子摘要
        summary = self.calculator.calculate_factor_summary(result, 'RSI')
        self.assertIn('factor_name', summary)
        self.assertEqual(summary['factor_name'], 'RSI')

if __name__ == '__main__':
    unittest.main()