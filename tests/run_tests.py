"""
运行所有测试
"""
import unittest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("FactorHub 系统测试")
    print("=" * 60)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 导入测试模块
    try:
        from tests.test_data_processor import TestDataProcessor
        from tests.test_factor_manager import TestFactorManager
        from tests.test_analyzer import TestAnalyzer
        from tests.test_backtester import TestBacktester
        from tests.test_integration import TestIntegration

        # 添加测试到套件
        suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
        suite.addTests(loader.loadTestsFromTestCase(TestFactorManager))
        suite.addTests(loader.loadTestsFromTestCase(TestAnalyzer))
        suite.addTests(loader.loadTestsFromTestCase(TestBacktester))
        suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # 输出测试结果
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        print(f"运行测试: {result.testsRun}")
        print(f"失败: {len(result.failures)}")
        print(f"错误: {len(result.errors)}")
        print(f"跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

        if result.failures:
            print("\n失败的测试:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")

        if result.errors:
            print("\n错误的测试:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")

        # 返回测试是否成功
        return len(result.failures) == 0 and len(result.errors) == 0

    except ImportError as e:
        print(f"导入测试模块失败: {e}")
        return False

def run_unit_tests():
    """运行单元测试"""
    print("运行单元测试...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 只运行单元测试（不包括集成测试）
    from tests.test_data_processor import TestDataProcessor
    from tests.test_factor_manager import TestFactorManager
    from tests.test_analyzer import TestAnalyzer
    from tests.test_backtester import TestBacktester

    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestFactorManager))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktester))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return len(result.failures) == 0 and len(result.errors) == 0

def run_integration_tests():
    """运行集成测试"""
    print("运行集成测试...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    from tests.test_integration import TestIntegration
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FactorHub 测试运行器")
    parser.add_argument("--unit", action="store_true", help="只运行单元测试")
    parser.add_argument("--integration", action="store_true", help="只运行集成测试")
    parser.add_argument("--all", action="store_true", help="运行所有测试（默认）")

    args = parser.parse_args()

    if args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    else:
        success = run_all_tests()

    # 退出码
    sys.exit(0 if success else 1)