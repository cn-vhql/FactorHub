"""
数据处理器测试
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import AKShareDataProvider, DataPreprocessor

class TestDataProcessor(unittest.TestCase):
    """数据处理器测试类"""

    def setUp(self):
        """测试初始化"""
        self.data_provider = AKShareDataProvider()
        self.preprocessor = DataPreprocessor()

        # 创建测试数据
        self.test_data = self._create_test_data()

    def _create_test_data(self):
        """创建测试数据"""
        symbols = ['000001', '000002', '600036']
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')

        data = []
        for symbol in symbols:
            for date in dates:
                # 模拟股价数据
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
                    'amount': price * max(1000000, int(np.random.randn() * 1000000 + 5000000))
                })

        return pd.DataFrame(data)

    def test_data_preprocessor(self):
        """测试数据预处理器"""
        # 测试缺失值填充
        data_with_nan = self.test_data.copy()
        data_with_nan.loc[0, 'close'] = np.nan

        filled_data = self.preprocessor.fill_missing_values(data_with_nan)
        self.assertFalse(filled_data['close'].isna().any())

        # 测试异常值处理
        data_with_outliers = self.test_data.copy()
        data_with_outliers.loc[0, 'close'] = 1000  # 异常值

        cleaned_data = self.preprocessor.remove_outliers(data_with_outliers)
        self.assertLess(cleaned_data.loc[0, 'close'], 1000)

        # 测试技术指标计算
        data_with_indicators = self.preprocessor.calculate_technical_indicators(self.test_data)
        self.assertIn('ma5', data_with_indicators.columns)
        self.assertIn('rsi', data_with_indicators.columns)

        # 测试收益率计算
        data_with_returns = self.preprocessor.add_returns(self.test_data)
        self.assertIn('return_1d', data_with_returns.columns)

    def test_preprocessor_pipeline(self):
        """测试预处理管道"""
        processed_data = self.preprocessor.preprocess_pipeline(
            self.test_data,
            adjust_type="none",
            fill_method="ffill",
            outlier_method="3sigma",
            add_returns=True,
            add_indicators=True
        )

        # 验证处理后的数据
        self.assertGreater(len(processed_data), 0)
        self.assertIn('return_1d', processed_data.columns)
        self.assertFalse(processed_data['close'].isna().any())

    def test_data_summary(self):
        """测试数据摘要功能"""
        summary = self.preprocessor.get_data_summary(self.test_data)

        self.assertIn('total_records', summary)
        self.assertIn('unique_symbols', summary)
        self.assertIn('date_range', summary)
        self.assertEqual(summary['total_records'], len(self.test_data))
        self.assertEqual(summary['unique_symbols'], 3)

if __name__ == '__main__':
    unittest.main()