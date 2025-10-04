"""
策略回测模块
"""
from .backtester import Backtester
from .strategy import FactorStrategy

__all__ = ['Backtester', 'FactorStrategy']