"""
FactorHub 使用示例
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def example_basic_workflow():
    """基础工作流程示例"""
    print("=" * 60)
    print("FactorHub 基础工作流程示例")
    print("=" * 60)

    # 1. 数据获取和预处理
    print("1. 数据获取和预处理...")
    from data_processor import DataPreprocessor

    # 创建模拟数据
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    symbols = ['000001', '000002', '600036']

    data = []
    for symbol in symbols:
        for date in dates:
            price = 10.0 + np.random.randn() * 0.5
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

    df = pd.DataFrame(data)

    # 数据预处理
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(
        df,
        add_returns=True,
        add_indicators=True
    )

    print(f"预处理完成，数据量: {len(processed_data)}")
    print(f"列名: {list(processed_data.columns)}")

    # 2. 因子计算
    print("\n2. 因子计算...")
    from factor_manager import FactorCalculator

    calculator = FactorCalculator()
    factor_names = ['RSI', 'MA5', 'MA20']
    factor_data = calculator.calculate_multiple_factors(
        factor_names, processed_data
    )

    print(f"计算因子: {factor_names}")
    for factor_name in factor_names:
        if factor_name in factor_data.columns:
            stats = calculator.get_factor_summary(factor_data, factor_name)
            print(f"{factor_name}: 均值={stats.get('mean', 'N/A'):.4f}, "
                  f"标准差={stats.get('std', 'N/A'):.4f}")

    # 3. 因子分析
    print("\n3. 因子分析...")
    from analyzer import FactorAnalyzer

    analyzer = FactorAnalyzer()

    # 分析RSI因子
    if 'RSI' in factor_data.columns:
        ic_results = analyzer.calculate_ic_analysis(
            factor_data, 'RSI', 'return_1d'
        )

        print(f"RSI因子分析结果:")
        print(f"  IC均值: {ic_results.get('ic_mean', 'N/A'):.4f}")
        print(f"  IR值: {ic_results.get('ir', 'N/A'):.4f}")
        print(f"  IC胜率: {ic_results.get('ic_win_rate', 'N/A'):.2%}")

        # 分层收益分析
        layer_results = analyzer.calculate_layered_returns(
            factor_data, 'RSI', 'return_1d', layers=3
        )

        if 'summary' in layer_results:
            print(f"分层收益结果:")
            for layer, stats in layer_results['summary'].items():
                print(f"  {layer}: 累计收益={stats.get('cumulative_return', 'N/A'):.2%}")

    # 4. 策略回测
    print("\n4. 策略回测...")
    from backtester import Backtester, FactorStrategy

    # 创建因子策略
    available_factors = [f for f in factor_names if f in factor_data.columns]
    if available_factors:
        strategy = FactorStrategy(
            name='示例因子策略',
            factor_names=available_factors,
            top_percent=0.2,
            long_only=True
        )

        # 执行回测
        backtester = Backtester(initial_capital=1000000)
        backtest_results = backtester.run_backtest(
            strategy,
            factor_data,
            '2023-02-01',
            '2023-06-30',
            'monthly'
        )

        if 'error' not in backtest_results:
            metrics = backtest_results['performance_metrics']
            print(f"回测结果:")
            print(f"  总收益率: {metrics.get('total_return', 'N/A'):.2%}")
            print(f"  年化收益率: {metrics.get('annual_return', 'N/A'):.2%}")
            print(f"  夏普比率: {metrics.get('sharpe_ratio', 'N/A'):.2f}")
            print(f"  最大回撤: {metrics.get('max_drawdown', 'N/A'):.2%}")

    print("\n基础工作流程示例完成！")

def example_custom_factor():
    """自定义因子示例"""
    print("\n" + "=" * 60)
    print("自定义因子示例")
    print("=" * 60)

    from factor_manager import CustomFactorManager

    # 创建自定义因子
    custom_manager = CustomFactorManager()

    # 定义自定义因子代码
    custom_factor_code = """
def calculate_factor(data):
    # 自定义因子：价格动量与成交量的组合
    # 20日价格动量
    momentum = data.groupby('symbol')['close'].transform(lambda x: x.pct_change(20))

    # 成交量比率
    volume_ratio = data.groupby('symbol')['volume'].transform(
        lambda x: x / x.rolling(20).mean()
    )

    # 组合因子
    return momentum * volume_ratio
"""

    # 添加自定义因子
    success = custom_manager.add_factor_from_code(
        'MOMENTUM_VOLUME',
        '价格动量与成交量组合因子',
        'custom',
        custom_factor_code
    )

    if success:
        print("自定义因子添加成功！")

        # 创建测试数据并计算自定义因子
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        symbols = ['000001', '000002']

        data = []
        for symbol in symbols:
            for date in dates:
                price = 10.0 + np.random.randn() * 0.5
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'close': price,
                    'volume': max(1000000, int(np.random.randn() * 1000000 + 5000000))
                })

        df = pd.DataFrame(data)

        # 执行自定义因子
        result = custom_manager.execute_factor('MOMENTUM_VOLUME', df)
        if 'MOMENTUM_VOLUME' in result.columns:
            print(f"自定义因子计算完成，统计信息:")
            print(f"  均值: {result['MOMENTUM_VOLUME'].mean():.6f}")
            print(f"  标准差: {result['MOMENTUM_VOLUME'].std():.6f}")
            print(f"  最大值: {result['MOMENTUM_VOLUME'].max():.6f}")
            print(f"  最小值: {result['MOMENTUM_VOLUME'].min():.6f}")

    print("自定义因子示例完成！")

def example_factor_mining():
    """因子挖掘示例"""
    print("\n" + "=" * 60)
    print("因子挖掘示例")
    print("=" * 60)

    from factor_miner import FactorGenerator

    # 创建测试数据
    dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
    symbols = ['000001', '000002', '600036']

    data = []
    for symbol in symbols:
        for date in dates:
            price = 10.0 + np.random.randn() * 0.5
            data.append({
                'symbol': symbol,
                'date': date,
                'open': price * (1 + np.random.randn() * 0.01),
                'high': price * (1 + abs(np.random.randn()) * 0.02),
                'low': price * (1 - abs(np.random.randn()) * 0.02),
                'close': price,
                'volume': max(1000000, int(np.random.randn() * 1000000 + 5000000))
            })

    df = pd.DataFrame(data)
    df['return_1d'] = df.groupby('symbol')['close'].pct_change()

    # 创建因子生成器
    generator = FactorGenerator()

    # 生成因子
    print("生成新因子...")
    generated_factors = generator.generate_all_factors(
        df,
        include_simple=True,
        include_combination=True,
        max_combinations=20
    )

    print(f"生成了 {len(generated_factors)} 个因子")

    # 评估因子
    if generated_factors:
        returns = df['return_1d']
        evaluation_results = generator.evaluate_factors(
            generated_factors, returns, 'ic'
        )

        print(f"评估了 {len(evaluation_results)} 个因子")
        print("前5个最佳因子:")
        for i, (factor_name, score) in enumerate(evaluation_results[:5]):
            print(f"  {i+1}. {factor_name}: IC = {score:.4f}")

    print("因子挖掘示例完成！")

def example_performance_comparison():
    """策略比较示例"""
    print("\n" + "=" * 60)
    print("策略比较示例")
    print("=" * 60)

    from backtester import Backtester, FactorStrategy

    # 创建测试数据
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    symbols = ['000001', '000002', '600036', '000858', '002415']

    data = []
    for symbol in symbols:
        for date in dates:
            price = 10.0 + np.random.randn() * 0.5
            # 模拟不同的因子值
            factor1 = np.random.randn()
            factor2 = np.random.randn()
            data.append({
                'symbol': symbol,
                'date': date,
                'open': price * (1 + np.random.randn() * 0.01),
                'high': price * (1 + abs(np.random.randn()) * 0.02),
                'low': price * (1 - abs(np.random.randn()) * 0.02),
                'close': price,
                'volume': max(1000000, int(np.random.randn() * 1000000 + 5000000)),
                'FACTOR1': factor1,
                'FACTOR2': factor2
            })

    df = pd.DataFrame(data)

    # 创建多个策略
    strategies = [
        FactorStrategy('因子1策略', ['FACTOR1'], top_percent=0.2),
        FactorStrategy('因子2策略', ['FACTOR2'], top_percent=0.2),
        FactorStrategy('多因子策略', ['FACTOR1', 'FACTOR2'], top_percent=0.2)
    ]

    # 比较策略
    backtester = Backtester()
    comparison = backtester.compare_strategies(
        strategies, df, '2023-02-01', '2023-06-30'
    )

    if 'error' not in comparison:
        print("策略比较结果:")
        print(f"最佳策略: {comparison['best_strategy']}")

        metrics_comparison = comparison.get('metrics_comparison', {})
        if 'total_return' in metrics_comparison:
            print("\n总收益率排名:")
            for name, value in metrics_comparison['total_return']['values']:
                print(f"  {name}: {value:.2%}")

        if 'sharpe_ratio' in metrics_comparison:
            print("\n夏普比率排名:")
            for name, value in metrics_comparison['sharpe_ratio']['values']:
                print(f"  {name}: {value:.2f}")

    print("策略比较示例完成！")

if __name__ == "__main__":
    """运行所有示例"""
    print("FactorHub 使用示例")
    print("=" * 60)

    try:
        example_basic_workflow()
        example_custom_factor()
        example_factor_mining()
        example_performance_comparison()

        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"运行示例时发生错误: {e}")
        import traceback
        traceback.print_exc()