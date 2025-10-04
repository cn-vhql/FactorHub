"""
FactorHub 导入路径配置
"""
import os
import sys

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 确保项目根目录在Python路径中
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入核心模块
try:
    from utils.config import DEFAULT_CONFIG
    from utils.logger import logger
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        import utils.config as config_module
        import utils.logger as logger_module
        DEFAULT_CONFIG = config_module.DEFAULT_CONFIG
        logger = logger_module.logger
    except ImportError:
        # 如果还是失败，创建默认配置和日志器
        import logging
        DEFAULT_CONFIG = {}
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

def safe_import(module_path, fallback=None):
    """
    安全导入模块

    Args:
        module_path: 模块路径，如 'data_processor.AKShareDataProvider'
        fallback: 导入失败时的备用对象

    Returns:
        导入的模块或备用对象
    """
    try:
        module_parts = module_path.split('.')
        module = __import__(module_parts[0])

        for part in module_parts[1:]:
            module = getattr(module, part)

        return module

    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to import {module_path}: {e}")
        return fallback

# 常用模块的安全导入
AKShareDataProvider = safe_import('data_processor.AKShareDataProvider')
DataPreprocessor = safe_import('data_processor.DataPreprocessor')
FactorCalculator = safe_import('factor_manager.FactorCalculator')
FactorLibrary = safe_import('factor_manager.FactorLibrary')
CustomFactorManager = safe_import('factor_manager.CustomFactorManager')
FactorAnalyzer = safe_import('analyzer.FactorAnalyzer')
PortfolioAnalyzer = safe_import('analyzer.PortfolioAnalyzer')
Backtester = safe_import('backtester.Backtester')
FactorStrategy = safe_import('backtester.FactorStrategy')
FactorGenerator = safe_import('factor_miner.FactorGenerator')
GeneticFactorMiner = safe_import('factor_miner.GeneticFactorMiner')

# 导出所有符号
__all__ = [
    'PROJECT_ROOT',
    'DEFAULT_CONFIG',
    'logger',
    'safe_import',
    'AKShareDataProvider',
    'DataPreprocessor',
    'FactorCalculator',
    'FactorLibrary',
    'CustomFactorManager',
    'FactorAnalyzer',
    'PortfolioAnalyzer',
    'Backtester',
    'FactorStrategy',
    'FactorGenerator',
    'GeneticFactorMiner'
]