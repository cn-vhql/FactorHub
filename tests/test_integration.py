"""
集成测试
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import DataPreprocessor
from factor_manager import FactorCalculator, FactorLibrary
from analyzer import FactorAnalyzer
from backtester import Backtester, FactorStrategy
from factor_miner import FactorGenerator

class TestIntegration(unittest.TestCase):
    """集成测试类"""

    def setUp(self):
        """测试初始化"""
        self.preprocessor = DataPreprocessor()
        self.calculator = FactorCalculator()
        self.analyzer = FactorAnalyzer()
        self.backtester = Backtester()
        self.factor_lib = FactorLibrary()
        self.generator = FactorGenerator()

        # 创建基础测试数据
        self.raw_data = self._create_raw_test_data()

    def _create_raw_test_data(self):
        """创建原始测试数据"""
        symbols = ['000001', '000002', '600036', '000858', '002415']
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')

        data = []
        for symbol in symbols:
            for date in dates:
                base_price = 10.0 + hash(symbol) % 100 / 10
                price = base_price + np.random.randn() * 0.5

                data.append({
                    'symbol': symbol,
                    'date': date,
                    'open': price * (1 + np.random.randn() * 0.01),
                    'high': price * (1 + abs(np.random.randn()) * 0.02),
                    'low': price * (1 - abs(np.random.randn()) * 0.02),
                    'close': price,
                    'volume': max(1000000, int(np.random.randn() * 1000000 + 5000000)),
                    'amount': price * max(1000000, int(np.random.randn() * 1000000 + 5000000)),
                    'pct_change': np.random.randn() * 0.02,
                    'turnover': np.random.randn() * 0.05
                })

        return pd.DataFrame(data)

    def test_full_workflow(self):
        """测试完整工作流程"""
        print("开始完整工作流程测试...")

        # 1. 数据预处理
        print("1. 数据预处理...")
        processed_data = self.preprocessor.preprocess_pipeline(
            self.raw_data,
            adjust_type="none",
            fill_method="ffill",
            add_returns=True,
            add_indicators=True
        )
        self.assertGreater(len(processed_data), 0)
        self.assertIn('return_1d', processed_data.columns)

        # 2. 因子计算
        print("2. 因子计算...")
        factor_names = ['MA5', 'MA20', 'RSI']
        factor_data = self.calculator.calculate_multiple_factors(
            factor_names, processed_data
        )

        for factor_name in factor_names:
            self.assertIn(factor_name, factor_data.columns)

        # 3. 因子分析
        print("3. 因子分析...")
        # 选择一个因子进行分析
        factor_to_analyze = 'RSI'
        if factor_to_analyze in factor_data.columns:
            ic_results = self.analyzer.calculate_ic_analysis(
                factor_data, factor_to_analyze, 'return_1d'
            )
            self.assertNotIn('error', ic_results)

            layer_results = self.analyzer.calculate_layered_returns(
                factor_data, factor_to_analyze, 'return_1d'
            )
            self.assertNotIn('error', layer_results)

        # 4. 策略回测
        print("4. 策略回测...")
        available_factors = [f for f in factor_names if f in factor_data.columns]
        if available_factors:
            strategy = FactorStrategy(
                name='集成测试策略',
                factor_names=available_factors,
                top_percent=0.2,
                long_only=True
            )

            backtest_results = self.backtester.run_backtest(
                strategy,
                factor_data,
                '2023-02-01',
                '2023-06-30',
                'monthly'
            )
            self.assertNotIn('error', backtest_results)
            self.assertIn('performance_metrics', backtest_results)

        print("完整工作流程测试完成！")

    def test_factor_mining_integration(self):
        """测试因子挖掘集成"""
        print("开始因子挖掘集成测试...")

        # 1. 数据预处理
        processed_data = self.preprocessor.preprocess_pipeline(
            self.raw_data,
            add_returns=True,
            add_indicators=True
        )

        # 2. 生成因子
        print("生成新因子...")
        generated_factors = self.generator.generate_all_factors(
            processed_data,
            include_simple=True,
            include_combination=True,
            max_combinations=20
        )

        if generated_factors:
            print(f"生成了 {len(generated_factors)} 个因子")

            # 3. 评估因子
            returns = processed_data['return_1d']
            evaluation_results = self.generator.evaluate_factors(
                generated_factors, returns, 'ic'
            )

            if evaluation_results:
                print(f"评估了 {len(evaluation_results)} 个因子")

                # 4. 选择最佳因子添加到数据中
                best_factor_name, best_score = evaluation_results[0]
                if best_factor_name in generated_factors:
                    processed_data[best_factor_name] = generated_factors[best_factor_name]

                    # 5. 分析新因子
                    ic_results = self.analyzer.calculate_ic_analysis(
                        processed_data, best_factor_name, 'return_1d'
                    )
                    self.assertNotIn('error', ic_results)

                    print(f"最佳因子 {best_factor_name} 的IC均值: {ic_results.get('ic_mean', 'N/A')}")

        print("因子挖掘集成测试完成！")

    def test_performance_analysis(self):
        """测试性能分析"""
        print("开始性能分析测试...")

        import time

        # 1. 测试数据预处理性能
        start_time = time.time()
        processed_data = self.preprocessor.preprocess_pipeline(
            self.raw_data,
            add_returns=True,
            add_indicators=True
        )
        preprocess_time = time.time() - start_time
        print(f"数据预处理耗时: {preprocess_time:.2f}秒")

        # 2. 测试因子计算性能
        factor_names = ['MA5', 'MA20', 'RSI']
        start_time = time.time()
        factor_data = self.calculator.calculate_multiple_factors(
            factor_names, processed_data
        )
        factor_calc_time = time.time() - start_time
        print(f"因子计算耗时: {factor_calc_time:.2f}秒")

        # 3. 测试因子分析性能
        if 'RSI' in factor_data.columns:
            start_time = time.time()
            ic_results = self.analyzer.calculate_ic_analysis(
                factor_data, 'RSI', 'return_1d'
            )
            analysis_time = time.time() - start_time
            print(f"因子分析耗时: {analysis_time:.2f}秒")

        # 4. 测试回测性能
        if 'RSI' in factor_data.columns:
            strategy = FactorStrategy(
                name='性能测试策略',
                factor_names=['RSI'],
                top_percent=0.2
            )

            start_time = time.time()
            backtest_results = self.backtester.run_backtest(
                strategy,
                factor_data,
                '2023-02-01',
                '2023-06-30',
                'monthly'
            )
            backtest_time = time.time() - start_time
            print(f"策略回测耗时: {backtest_time:.2f}秒")

        print("性能分析测试完成！")

    def test_error_handling(self):
        """测试错误处理"""
        print("开始错误处理测试...")

        # 1. 测试空数据处理
        empty_data = pd.DataFrame()
        try:
            processed_data = self.preprocessor.preprocess_pipeline(empty_data)
            # 应该返回空DataFrame而不报错
            self.assertTrue(processed_data.empty)
        except Exception as e:
            self.fail(f"处理空数据时抛出异常: {e}")

        # 2. 测试无效因子名称
        try:
            result = self.calculator.calculate_single_factor('INVALID_FACTOR', self.raw_data)
            # 应该返回空DataFrame而不报错
            self.assertTrue(result.empty)
        except Exception as e:
            self.fail(f"处理无效因子时抛出异常: {e}")

        # 3. 测试时间范围错误
        if not self.raw_data.empty:
            strategy = FactorStrategy('测试策略', ['MA5'])
            try:
                # 使用无效的时间范围
                results = self.backtester.run_backtest(
                    strategy,
                    self.raw_data,
                    '2025-01-01',  # 未来日期
                    '2025-12-31'
                )
                # 应该返回错误信息而不报错
                self.assertIn('error', results)
            except Exception as e:
                self.fail(f"处理无效时间范围时抛出异常: {e}")

        print("错误处理测试完成！")

    def test_data_consistency(self):
        """测试数据一致性"""
        print("开始数据一致性测试...")

        # 1. 测试数据长度一致性
        processed_data = self.preprocessor.preprocess_pipeline(
            self.raw_data,
            add_returns=True
        )

        original_length = len(self.raw_data)
        processed_length = len(processed_data)
        self.assertEqual(original_length, processed_length)

        # 2. 测试因子数据长度一致性
        factor_data = self.calculator.calculate_multiple_factors(
            ['MA5', 'MA20'], processed_data
        )

        for factor_name in ['MA5', 'MA20']:
            if factor_name in factor_data.columns:
                self.assertEqual(len(factor_data), len(factor_data[factor_name].dropna()) +
                               factor_data[factor_name].isna().sum())

        # 3. 测试收益率计算一致性
        returns = processed_data.groupby('symbol')['close'].pct_change()
        self.assertTrue((returns >= -1).all())  # 收益率不应小于-100%

        print("数据一致性测试完成！")

if __name__ == '__main__':
    # 运行集成测试
    unittest.main(verbosity=2)