"""
回测器测试
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import Backtester, FactorStrategy
from factor_manager import FactorCalculator

class TestBacktester(unittest.TestCase):
    """回测器测试类"""

    def setUp(self):
        """测试初始化"""
        self.backtester = Backtester(
            initial_capital=1000000,
            commission=0.0003,
            slippage=0.001
        )

        # 创建测试数据
        self.test_data = self._create_test_data_with_factors()

    def _create_test_data_with_factors(self):
        """创建带因子的测试数据"""
        symbols = ['000001', '000002', '600036', '000858', '002415']
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')

        data = []
        for symbol in symbols:
            for date in dates:
                base_price = 10.0 + hash(symbol) % 100 / 10
                price = base_price + np.random.randn() * 0.5

                # 模拟因子值（与价格有一定相关性）
                factor_score = np.random.randn()  # 简化的因子得分

                data.append({
                    'symbol': symbol,
                    'date': date,
                    'open': price * (1 + np.random.randn() * 0.01),
                    'high': price * (1 + abs(np.random.randn()) * 0.02),
                    'low': price * (1 - abs(np.random.randn()) * 0.02),
                    'close': price,
                    'volume': max(1000000, int(np.random.randn() * 1000000 + 5000000)),
                    'amount': price * max(1000000, int(np.random.randn() * 1000000 + 5000000)),
                    'TEST_FACTOR': factor_score
                })

        return pd.DataFrame(data)

    def test_backtester_initialization(self):
        """测试回测器初始化"""
        self.assertEqual(self.backtester.initial_capital, 1000000)
        self.assertEqual(self.backtester.current_capital, 1000000)
        self.assertEqual(self.backtester.commission, 0.0003)
        self.assertEqual(self.backtester.slippage, 0.001)
        self.assertEqual(len(self.backtester.positions), 0)

    def test_reset_function(self):
        """测试重置功能"""
        # 修改状态
        self.backtester.current_capital = 500000
        self.backtester.positions['000001'] = 100

        # 重置
        self.backtester.reset()

        # 验证重置
        self.assertEqual(self.backtester.current_capital, 1000000)
        self.assertEqual(len(self.backtester.positions), 0)

    def test_portfolio_value_calculation(self):
        """测试组合价值计算"""
        date = pd.Timestamp('2023-01-15')

        # 添加持仓
        self.backtester.positions['000001'] = 1000

        # 创建价格数据
        price_data = pd.DataFrame({
            'symbol': ['000001'],
            'close': [10.5]
        })

        # 计算组合价值
        portfolio_value = self.backtester.calculate_portfolio_value(date, self.test_data)
        self.assertGreater(portfolio_value, self.backtester.current_capital)

    def test_trade_execution(self):
        """测试交易执行"""
        date = pd.Timestamp('2023-01-15')
        current_price = 10.0

        # 测试买入
        trade_result = self.backtester.execute_trade('000001', 0.1, current_price, date)
        self.assertIn('action', trade_result)
        self.assertEqual(trade_result['symbol'], '000001')

        # 测试卖出
        self.backtester.positions['000001'] = 1000
        trade_result = self.backtester.execute_trade('000001', 0.0, current_price, date)
        self.assertEqual(trade_result['action'], 'sell')

    def test_factor_strategy(self):
        """测试因子策略"""
        strategy = FactorStrategy(
            name='测试因子策略',
            factor_names=['TEST_FACTOR'],
            top_percent=0.2,
            long_only=True
        )

        # 测试信号生成
        date = self.test_data['date'].max()
        signals = strategy.generate_signals(self.test_data, date)

        self.assertIsInstance(signals, dict)
        self.assertGreater(len(signals), 0)

        # 验证信号格式
        for symbol, weight in signals.items():
            self.assertIsInstance(symbol, str)
            self.assertIsInstance(weight, (int, float))
            self.assertGreaterEqual(weight, 0)  # 仅多头策略

    def test_full_backtest(self):
        """测试完整回测"""
        # 创建策略
        strategy = FactorStrategy(
            name='测试因子策略',
            factor_names=['TEST_FACTOR'],
            top_percent=0.2,
            long_only=True
        )

        # 执行回测
        results = self.backtester.run_backtest(
            strategy,
            self.test_data,
            '2023-01-01',
            '2023-03-31',
            'monthly'
        )

        # 验证结果
        self.assertNotIn('error', results)
        self.assertIn('performance_metrics', results)
        self.assertIn('trading_statistics', results)
        self.assertIn('portfolio_statistics', results)
        self.assertIn('detailed_data', results)

        # 验证绩效指标
        metrics = results['performance_metrics']
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)

    def test_rebalance_dates(self):
        """测试调仓日期计算"""
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')

        # 测试月度调仓
        monthly_dates = self.backtester._get_rebalance_dates(dates, 'monthly')
        self.assertGreater(len(monthly_dates), 0)

        # 测试周度调仓
        weekly_dates = self.backtester._get_rebalance_dates(dates, 'weekly')
        self.assertGreater(len(weekly_dates), 0)

        # 测试日度调仓
        daily_dates = self.backtester._get_rebalance_dates(dates, 'daily')
        self.assertEqual(len(daily_dates), len(dates))

    def test_strategy_comparison(self):
        """测试策略比较"""
        # 创建多个策略
        strategy1 = FactorStrategy('策略1', ['TEST_FACTOR'], top_percent=0.2)
        strategy2 = FactorStrategy('策略2', ['TEST_FACTOR'], top_percent=0.3)

        strategies = [strategy1, strategy2]

        # 比较策略
        comparison = self.backtester.compare_strategies(
            strategies, self.test_data, '2023-01-01', '2023-03-31'
        )

        # 验证比较结果
        self.assertNotIn('error', comparison)
        self.assertIn('individual_results', comparison)
        self.assertIn('ranking', comparison)
        self.assertEqual(len(comparison['individual_results']), 2)

if __name__ == '__main__':
    unittest.main()