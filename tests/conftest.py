import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_module(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


talib_stub = types.ModuleType("talib")


def _series_result(values, index=None):
    array = np.asarray(values, dtype=float)
    if index is None:
        return pd.Series(array)
    return pd.Series(array, index=index)


def _single_input_stub(series, timeperiod=14, **kwargs):
    if isinstance(series, pd.Series):
        return _series_result(series.values, series.index)
    return np.asarray(series, dtype=float)


def _triple_input_stub(high, low, close, timeperiod=14, **kwargs):
    if isinstance(close, pd.Series):
        return _series_result(close.values, close.index)
    return np.asarray(close, dtype=float)


talib_stub.EMA = _single_input_stub
talib_stub.RSI = _single_input_stub
talib_stub.KAMA = _single_input_stub
talib_stub.ROC = _single_input_stub
talib_stub.MOM = _single_input_stub
talib_stub.SMA = _single_input_stub
talib_stub.ADX = _triple_input_stub
talib_stub.CCI = _triple_input_stub
talib_stub.ATR = _triple_input_stub
talib_stub.WILLR = _triple_input_stub
talib_stub.OBV = lambda close, volume, **kwargs: _series_result(close.values if isinstance(close, pd.Series) else close, close.index if isinstance(close, pd.Series) else None)
talib_stub.MACD = lambda close, **kwargs: (
    _series_result(close.values if isinstance(close, pd.Series) else close, close.index if isinstance(close, pd.Series) else None),
    _series_result(close.values if isinstance(close, pd.Series) else close, close.index if isinstance(close, pd.Series) else None),
    _series_result(close.values if isinstance(close, pd.Series) else close, close.index if isinstance(close, pd.Series) else None),
)
talib_stub.BBANDS = lambda close, **kwargs: (
    _series_result(close.values if isinstance(close, pd.Series) else close, close.index if isinstance(close, pd.Series) else None),
    _series_result(close.values if isinstance(close, pd.Series) else close, close.index if isinstance(close, pd.Series) else None),
    _series_result(close.values if isinstance(close, pd.Series) else close, close.index if isinstance(close, pd.Series) else None),
)
talib_stub.STOCH = lambda high, low, close, **kwargs: (
    _series_result(close.values if isinstance(close, pd.Series) else close, close.index if isinstance(close, pd.Series) else None),
    _series_result(close.values if isinstance(close, pd.Series) else close, close.index if isinstance(close, pd.Series) else None),
)
talib_stub.STOCHRSI = lambda close, **kwargs: (
    _series_result(close.values if isinstance(close, pd.Series) else close, close.index if isinstance(close, pd.Series) else None),
    _series_result(close.values if isinstance(close, pd.Series) else close, close.index if isinstance(close, pd.Series) else None),
)
_ensure_module("talib", talib_stub)

akshare_stub = types.ModuleType("akshare")
akshare_stub.stock_zh_a_daily = lambda *args, **kwargs: pd.DataFrame()
_ensure_module("akshare", akshare_stub)

xgboost_stub = types.ModuleType("xgboost")
_ensure_module("xgboost", xgboost_stub)
