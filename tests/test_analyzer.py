"""
因子分析器测试
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import FactorAnalyzer, PortfolioAnalyzer
from factor_manager import FactorCalculator

class TestAnalyzer(unittest.TestCase):
    """因子分析器测试类"""

    def setUp(self):
        """测试初始化"""
        self.factor_analyzer = FactorAnalyzer()
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.calculator = FactorCalculator()

        # 创建测试数据
        self.test_data = self._create_test_data_with_factors()

    def _create_test_data_with_factors(self):
        """创建带因子的测试数据"""
        symbols = ['000001', '000002', '600036', '000858', '002415']
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')

        data = []
        for symbol in symbols:
            for date in dates:
                base_price = 10.0 + hash(symbol) % 100 / 10
                price = base_price + np.random.randn() * 0.5

                # 模拟因子值
                rsi_value = 50 + np.random.randn() * 15
                ma5_value = price * (1 + np.random.randn() * 0.02)

                data.append({
                    'symbol': symbol,
                    'date': date,
                    'close': price,
                    'volume': max(1000000, int(np.random.randn() * 1000000 + 5000000)),
                    'RSI': rsi_value,
                    'MA5': ma5_value,
                    'return_1d': np.random.randn() * 0.02  # 模拟收益率
                })

        return pd.DataFrame(data)

    def test_ic_analysis(self):
        """测试IC分析"""
        # 测试Spearman IC
        ic_results = self.factor_analyzer.calculate_ic_analysis(
            self.test_data, 'RSI', 'return_1d', 'spearman'
        )
        self.assertNotIn('error', ic_results)
        self.assertIn('ic_mean', ic_results)
        self.assertIn('ir', ic_results)
        self.assertIn('ic_win_rate', ic_results)

        # 测试Pearson IC
        ic_results_pearson = self.factor_analyzer.calculate_ic_analysis(
            self.test_data, 'RSI', 'return_1d', 'pearson'
        )
        self.assertNotIn('error', ic_results_pearson)

    def test_layered_returns(self):
        """测试分层收益分析"""
        layer_results = self.factor_analyzer.calculate_layered_returns(
            self.test_data, 'RSI', 'return_1d', layers=5
        )
        self.assertNotIn('error', layer_results)
        self.assertIn('layer_data', layer_results)
        self.assertIn('summary', layer_results)
        self.assertEqual(len(layer_results['summary']), 5)  # 5层

    def test_turnover_analysis(self):
        """测试换手率分析"""
        turnover_results = self.factor_analyzer.calculate_turnover_analysis(
            self.test_data, 'RSI', top_percent=0.2
        )
        self.assertNotIn('error', turnover_results)
        self.assertIn('turnover_data', turnover_results)
        self.assertIn('avg_turnover', turnover_results)

    def test_stability_analysis(self):
        """测试稳定性分析"""
        stability_results = self.factor_analyzer.calculate_stability_analysis(
            self.test_data, 'RSI', window=30
        )
        self.assertNotIn('error', stability_results)
        self.assertIn('daily_stats', stability_results)
        self.assertIn('stability_metrics', stability_results)

    def test_factor_report(self):
        """测试因子报告生成"""
        report = self.factor_analyzer.generate_factor_report(self.test_data, 'RSI')
        self.assertNotIn('error', report)
        self.assertIn('ic_analysis', report)
        self.assertIn('layer_analysis', report)
        self.assertIn('turnover_analysis', report)
        self.assertIn('stability_analysis', report)
        self.assertIn('overall_score', report)

    def test_factor_comparison(self):
        """测试因子比较"""
        comparison = self.factor_analyzer.compare_factors(
            self.test_data, ['RSI', 'MA5']
        )
        self.assertNotIn('error', comparison)
        self.assertIn('comparison_results', comparison)
        self.assertIn('ranking', comparison)

    def test_portfolio_metrics(self):
        """测试组合绩效计算"""
        # 创建组合价值序列
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
        values = 1000000 * (1 + np.random.randn(len(dates)) * 0.01).cumprod()
        portfolio_data = pd.DataFrame({
            'date': dates,
            'value': values
        })

        metrics = self.portfolio_analyzer.calculate_portfolio_metrics(portfolio_data)
        self.assertIn('return_metrics', metrics)
        self.assertIn('risk_metrics', metrics)
        self.assertIn('other_metrics', metrics)

    def test_concentration_risk(self):
        """测试集中度风险分析"""
        # 创建持仓数据
        portfolio_data = pd.DataFrame({
            'symbol': ['000001', '000002', '600036', '000858', '002415'],
            'weight': [0.3, 0.25, 0.2, 0.15, 0.1]
        })

        concentration = self.portfolio_analyzer.calculate_concentration_risk(portfolio_data)
        self.assertNotIn('error', concentration)
        self.assertIn('top_n_holdings', concentration)
        self.assertIn('gini_coefficient', concentration)

if __name__ == '__main__':
    unittest.main()