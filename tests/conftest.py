"""
全局测试配置：在 talib 未安装时注入 mock，避免导入失败。
所有 talib 函数 mock 均动态匹配输入长度，防止形状不匹配错误。
"""
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_talib_mock() -> MagicMock:
    """构造一个能正确处理任意长度输入的 talib mock。"""

    def _zeros_1d(arr, *args, **kwargs):
        n = len(arr) if hasattr(arr, "__len__") else 100
        return np.full(n, np.nan)

    def _zeros_3out(arr, *args, **kwargs):
        n = len(arr) if hasattr(arr, "__len__") else 100
        z = np.full(n, np.nan)
        return z, z.copy(), z.copy()

    def _zeros_2out(arr, *args, **kwargs):
        n = len(arr) if hasattr(arr, "__len__") else 100
        z = np.full(n, np.nan)
        return z, z.copy()

    mock = MagicMock()
    for fn in ("EMA", "SMA", "RSI", "ADX", "CCI", "ATR", "OBV", "WILLR", "KAMA", "ROC", "MOM"):
        setattr(mock, fn, MagicMock(side_effect=_zeros_1d))
    mock.MACD = MagicMock(side_effect=_zeros_3out)
    mock.BBANDS = MagicMock(side_effect=_zeros_3out)
    mock.STOCH = MagicMock(side_effect=_zeros_2out)
    mock.STOCHRSI = MagicMock(side_effect=_zeros_2out)
    return mock


# 检测 talib 是否真实可用
try:
    import talib as _real_talib  # noqa: F401
    TALIB_AVAILABLE = True
except ModuleNotFoundError:
    TALIB_AVAILABLE = False
    sys.modules["talib"] = _make_talib_mock()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_talib: 需要真实 talib 安装，无 talib 时自动 skip",
    )


def pytest_collection_modifyitems(config, items):
    """自动为依赖真实 talib 的测试加 skip marker。"""
    if TALIB_AVAILABLE:
        return
    skip_talib = pytest.mark.skip(reason="talib 未安装，跳过需要真实 talib 的测试")
    for item in items:
        if item.get_closest_marker("requires_talib"):
            item.add_marker(skip_talib)
