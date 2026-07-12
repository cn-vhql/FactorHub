"""
公式执行公共运行时。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import talib


class FormulaExecutionError(ValueError):
    """公式执行失败。"""


FORMULA_TYPE_ALIASES = {
    "auto": "auto",
    "expression": "auto",
    "function": "python",
    "python": "python",
    "python_expression": "python",
    "python_function": "python",
    "mylanguage": "mylanguage",
    "my_language": "mylanguage",
    "tdx": "mylanguage",
    "tongdaxin": "mylanguage",
}


PYTHON_MARKERS = (
    "def calculate_factor",
    "df[",
    "np.",
    "pd.",
    ".rolling(",
    ".expanding(",
    ".shift(",
    ".pct_change(",
    ".astype(",
    ".clip(",
    ".quantile(",
    ".skew(",
    ".kurt(",
    ".kurtosis(",
    "lambda ",
    "return ",
)


def normalize_formula_type(formula_type: Optional[str], code: str) -> str:
    """归一化公式类型。"""
    if formula_type:
        normalized = FORMULA_TYPE_ALIASES.get(formula_type.strip().lower())
        if normalized:
            if normalized != "auto":
                return normalized
        else:
            raise FormulaExecutionError(f"不支持的公式类型: {formula_type}")

    return infer_formula_type(code)


def infer_formula_type(code: str) -> str:
    """根据代码内容推断公式类型。"""
    stripped = code.strip()
    lowered = stripped.lower()

    if not stripped:
        return "mylanguage"

    if lowered.startswith("def calculate_factor"):
        return "python"

    if any(marker in lowered for marker in PYTHON_MARKERS):
        return "python"

    if ":=" in stripped:
        return "mylanguage"

    for line in stripped.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith(("#", "//")):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", candidate):
            return "mylanguage"

    return "mylanguage"


def normalize_result(value: Any, index: pd.Index) -> pd.Series:
    """将执行结果规范化为 Series。"""
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            raise FormulaExecutionError("公式返回了空 DataFrame")
        value = value.iloc[:, 0]

    if isinstance(value, pd.Series):
        if not value.index.equals(index):
            return value.reindex(index)
        return value

    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.ndim != 1:
            raise FormulaExecutionError("仅支持一维公式结果")
        if len(array) == len(index):
            return pd.Series(array, index=index)
        if len(array) == 1:
            return pd.Series(np.repeat(array[0], len(index)), index=index)
        raise FormulaExecutionError("公式结果长度与行情数据长度不一致")

    return pd.Series(np.repeat(value, len(index)), index=index)


def ensure_bool_scalar(value: Any) -> bool:
    """确保控制流条件是标量布尔值。"""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int, np.floating, float)):
        return bool(value)
    raise FormulaExecutionError("控制流条件必须是标量，不能是序列")


def ensure_scalar(value: Any, *, default: Optional[float] = None) -> float:
    """将参数收敛为标量。"""
    if isinstance(value, pd.Series):
        valid = value.dropna()
        if valid.empty:
            if default is None:
                raise FormulaExecutionError("无法从空序列中提取标量参数")
            return default
        return float(valid.iloc[-1])
    if isinstance(value, np.ndarray):
        if value.size == 0:
            if default is None:
                raise FormulaExecutionError("无法从空数组中提取标量参数")
            return default
        return float(value.reshape(-1)[-1])
    if value is None:
        if default is None:
            raise FormulaExecutionError("标量参数不能为空")
        return default
    return float(value)


def ensure_int(value: Any, *, default: Optional[int] = None) -> int:
    """将参数收敛为整数。"""
    return int(round(ensure_scalar(value, default=default)))


def _align(left: Any, right: Any, index: pd.Index) -> tuple[Any, Any]:
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        return left.reindex(index), right.reindex(index)
    if isinstance(left, pd.Series):
        return left.reindex(index), pd.Series(np.repeat(right, len(index)), index=index)
    if isinstance(right, pd.Series):
        return pd.Series(np.repeat(left, len(index)), index=index), right.reindex(index)
    return left, right


def elementwise_binary(op: str, left: Any, right: Any, index: pd.Index) -> Any:
    """执行按元素二元运算。"""
    left, right = _align(left, right, index)

    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return left / right
    if op == "%":
        return left % right
    if op == "^":
        return left ** right
    if op in {"AND", "&"}:
        return left.astype(bool) & right.astype(bool) if isinstance(left, pd.Series) else bool(left) and bool(right)
    if op in {"OR", "|"}:
        return left.astype(bool) | right.astype(bool) if isinstance(left, pd.Series) else bool(left) or bool(right)
    if op in {">", ">=", "<", "<=", "=", "==", "<>", "!="}:
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op in {"=", "=="}:
            return left == right
        return left != right
    raise FormulaExecutionError(f"不支持的运算符: {op}")


def elementwise_unary(op: str, value: Any) -> Any:
    """执行一元运算。"""
    if op == "+":
        return value
    if op == "-":
        return -value
    if op == "NOT":
        if isinstance(value, pd.Series):
            return ~value.astype(bool)
        return not bool(value)
    raise FormulaExecutionError(f"不支持的一元运算符: {op}")


def _series(values: Iterable[Any], index: pd.Index) -> pd.Series:
    return pd.Series(list(values), index=index)


def _talib_single_arg(func: Callable[..., Any], series: Any, index: pd.Index, **kwargs: Any) -> pd.Series:
    if isinstance(series, pd.Series):
        result = func(series.values, **kwargs)
        return pd.Series(result, index=series.index)
    series = normalize_result(series, index)
    result = func(series.values, **kwargs)
    return pd.Series(result, index=index)


def _talib_triple_arg(
    func: Callable[..., Any],
    high: Any,
    low: Any,
    close: Any,
    index: pd.Index,
    **kwargs: Any,
) -> pd.Series:
    high = normalize_result(high, index)
    low = normalize_result(low, index)
    close = normalize_result(close, index)
    result = func(high.values, low.values, close.values, **kwargs)
    return pd.Series(result, index=index)


class RollingWindow:
    """安全滚动窗口包装器。"""

    def __init__(self, series: pd.Series, window: int, min_periods: Optional[int] = None):
        self.series = series
        self.window = max(1, int(window))
        self.min_periods = max(1, int(min_periods or 1))
        self._rolling = series.rolling(window=self.window, min_periods=self.min_periods)

    def mean(self) -> pd.Series:
        return self._rolling.mean()

    def std(self) -> pd.Series:
        return self._rolling.std()

    def max(self) -> pd.Series:
        return self._rolling.max()

    def min(self) -> pd.Series:
        return self._rolling.min()

    def median(self) -> pd.Series:
        return self._rolling.median()

    def sum(self) -> pd.Series:
        return self._rolling.sum()

    def skew(self) -> pd.Series:
        return self._rolling.skew()

    def kurt(self) -> pd.Series:
        return self._rolling.kurt()

    def kurtosis(self) -> pd.Series:
        return self._rolling.kurt()

    def quantile(self, q: Any) -> pd.Series:
        return self._rolling.quantile(ensure_scalar(q))


class ExpandingWindow:
    """安全扩展窗口包装器。"""

    def __init__(self, series: pd.Series, min_periods: Optional[int] = None):
        self.series = series
        self.min_periods = max(1, int(min_periods or 1))
        self._expanding = series.expanding(min_periods=self.min_periods)

    def mean(self) -> pd.Series:
        return self._expanding.mean()

    def std(self) -> pd.Series:
        return self._expanding.std()

    def max(self) -> pd.Series:
        return self._expanding.max()

    def min(self) -> pd.Series:
        return self._expanding.min()

    def sum(self) -> pd.Series:
        return self._expanding.sum()


@dataclass(frozen=True)
class SafeModule:
    """安全模块代理。"""

    name: str
    attributes: Mapping[str, Any]

    def resolve(self, attribute: str) -> Any:
        if attribute not in self.attributes:
            raise FormulaExecutionError(f"{self.name} 不允许访问属性 {attribute}")
        return self.attributes[attribute]


class FormulaFunctions:
    """统一函数库。"""

    def __init__(self, index: pd.Index):
        self.index = index

    def _as_series(self, value: Any) -> pd.Series:
        return normalize_result(value, self.index)

    def MA(self, series: Any, period: Any = 5, timeperiod: Any = None, **_: Any) -> pd.Series:
        window = ensure_int(timeperiod if timeperiod is not None else period, default=5)
        data = self._as_series(series)
        return data.rolling(window=window, min_periods=1).mean()

    def SMA(
        self,
        series: Any,
        period: Any = 5,
        weight: Any = None,
        timeperiod: Any = None,
        **_: Any,
    ) -> pd.Series:
        data = self._as_series(series)

        if timeperiod is not None:
            return self.MA(data, timeperiod=timeperiod)

        window = ensure_int(period, default=5)
        if weight is None:
            return self.MA(data, period=window)

        m = ensure_scalar(weight, default=1.0)
        if window <= 0:
            raise FormulaExecutionError("SMA 的 N 参数必须大于 0")
        if m <= 0 or m > window:
            raise FormulaExecutionError("SMA 的 M 参数必须满足 0 < M <= N")

        result = pd.Series(np.nan, index=data.index, dtype=float)
        prev = np.nan
        for i, current in enumerate(data.astype(float)):
            if pd.isna(current):
                result.iloc[i] = prev
                continue
            if math.isnan(prev):
                prev = current
            else:
                prev = (m * current + (window - m) * prev) / window
            result.iloc[i] = prev
        return result

    def EMA(self, series: Any, period: Any = 12, timeperiod: Any = None, **kwargs: Any) -> pd.Series:
        window = ensure_int(timeperiod if timeperiod is not None else period, default=12)
        return _talib_single_arg(talib.EMA, self._as_series(series), self.index, timeperiod=window, **kwargs)

    def RSI(self, series: Any, period: Any = 14, timeperiod: Any = None, **kwargs: Any) -> pd.Series:
        window = ensure_int(timeperiod if timeperiod is not None else period, default=14)
        return _talib_single_arg(talib.RSI, self._as_series(series), self.index, timeperiod=window, **kwargs)

    def KAMA(self, series: Any, period: Any = 30, timeperiod: Any = None, **kwargs: Any) -> pd.Series:
        window = ensure_int(timeperiod if timeperiod is not None else period, default=30)
        return _talib_single_arg(talib.KAMA, self._as_series(series), self.index, timeperiod=window, **kwargs)

    def ROC(self, series: Any, period: Any = 10, timeperiod: Any = None, **kwargs: Any) -> pd.Series:
        window = ensure_int(timeperiod if timeperiod is not None else period, default=10)
        return _talib_single_arg(talib.ROC, self._as_series(series), self.index, timeperiod=window, **kwargs)

    def MOM(self, series: Any, period: Any = 10, timeperiod: Any = None, **kwargs: Any) -> pd.Series:
        window = ensure_int(timeperiod if timeperiod is not None else period, default=10)
        return _talib_single_arg(talib.MOM, self._as_series(series), self.index, timeperiod=window, **kwargs)

    def MACD(
        self,
        series: Any,
        fastperiod: Any = 12,
        slowperiod: Any = 26,
        signalperiod: Any = 9,
        **kwargs: Any,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        data = self._as_series(series)
        macd, signal, hist = talib.MACD(
            data.values,
            fastperiod=ensure_int(fastperiod, default=12),
            slowperiod=ensure_int(slowperiod, default=26),
            signalperiod=ensure_int(signalperiod, default=9),
            **kwargs,
        )
        return (
            pd.Series(macd, index=self.index),
            pd.Series(signal, index=self.index),
            pd.Series(hist, index=self.index),
        )

    def ADX(self, high: Any, low: Any, close: Any, period: Any = 14, timeperiod: Any = None, **kwargs: Any) -> pd.Series:
        window = ensure_int(timeperiod if timeperiod is not None else period, default=14)
        return _talib_triple_arg(talib.ADX, high, low, close, self.index, timeperiod=window, **kwargs)

    def CCI(self, high: Any, low: Any, close: Any, period: Any = 14, timeperiod: Any = None, **kwargs: Any) -> pd.Series:
        window = ensure_int(timeperiod if timeperiod is not None else period, default=14)
        return _talib_triple_arg(talib.CCI, high, low, close, self.index, timeperiod=window, **kwargs)

    def ATR(self, high: Any, low: Any, close: Any, period: Any = 14, timeperiod: Any = None, **kwargs: Any) -> pd.Series:
        window = ensure_int(timeperiod if timeperiod is not None else period, default=14)
        return _talib_triple_arg(talib.ATR, high, low, close, self.index, timeperiod=window, **kwargs)

    def WILLR(self, high: Any, low: Any, close: Any, period: Any = 14, timeperiod: Any = None, **kwargs: Any) -> pd.Series:
        window = ensure_int(timeperiod if timeperiod is not None else period, default=14)
        return _talib_triple_arg(talib.WILLR, high, low, close, self.index, timeperiod=window, **kwargs)

    def BBANDS(
        self,
        series: Any,
        period: Any = 20,
        timeperiod: Any = None,
        nbdevup: Any = 2,
        nbdevdn: Any = 2,
        **kwargs: Any,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        data = self._as_series(series)
        upper, middle, lower = talib.BBANDS(
            data.values,
            timeperiod=ensure_int(timeperiod if timeperiod is not None else period, default=20),
            nbdevup=ensure_scalar(nbdevup, default=2.0),
            nbdevdn=ensure_scalar(nbdevdn, default=2.0),
            **kwargs,
        )
        return (
            pd.Series(upper, index=self.index),
            pd.Series(middle, index=self.index),
            pd.Series(lower, index=self.index),
        )

    def OBV(self, close: Any, volume: Any, **kwargs: Any) -> pd.Series:
        close = self._as_series(close)
        volume = self._as_series(volume)
        result = talib.OBV(close.values, volume.values, **kwargs)
        return pd.Series(result, index=self.index)

    def STOCH(self, high: Any, low: Any, close: Any, **kwargs: Any) -> tuple[pd.Series, pd.Series]:
        high = self._as_series(high)
        low = self._as_series(low)
        close = self._as_series(close)
        slowk, slowd = talib.STOCH(high.values, low.values, close.values, **kwargs)
        return pd.Series(slowk, index=self.index), pd.Series(slowd, index=self.index)

    def STOCHRSI(self, series: Any, **kwargs: Any) -> tuple[pd.Series, pd.Series]:
        data = self._as_series(series)
        fastk, fastd = talib.STOCHRSI(data.values, **kwargs)
        return pd.Series(fastk, index=self.index), pd.Series(fastd, index=self.index)

    def REF(self, series: Any, periods: Any = 1) -> pd.Series:
        return self._as_series(series).shift(ensure_int(periods, default=1))

    def HHV(self, series: Any, periods: Any = 5) -> pd.Series:
        data = self._as_series(series)
        window = ensure_int(periods, default=5)
        return data.expanding().max() if window <= 0 else data.rolling(window=window, min_periods=1).max()

    def LLV(self, series: Any, periods: Any = 5) -> pd.Series:
        data = self._as_series(series)
        window = ensure_int(periods, default=5)
        return data.expanding().min() if window <= 0 else data.rolling(window=window, min_periods=1).min()

    def SUM(self, series: Any, periods: Any = 5) -> pd.Series:
        data = self._as_series(series)
        window = ensure_int(periods, default=5)
        return data.cumsum() if window <= 0 else data.rolling(window=window, min_periods=1).sum()

    def AVE(self, series: Any, periods: Any = 5) -> pd.Series:
        data = self._as_series(series)
        window = ensure_int(periods, default=5)
        return data.expanding().mean() if window <= 0 else data.rolling(window=window, min_periods=1).mean()

    def STD(self, series: Any, periods: Any = 5) -> pd.Series:
        data = self._as_series(series)
        window = ensure_int(periods, default=5)
        return data.expanding().std(ddof=0) if window <= 0 else data.rolling(window=window, min_periods=1).std(ddof=0)

    def ABS(self, series: Any) -> Any:
        return series.abs() if isinstance(series, pd.Series) else abs(series)

    def LOG(self, series: Any) -> Any:
        return np.log(series)

    def LN(self, series: Any) -> Any:
        return np.log(series)

    def EXP(self, series: Any) -> Any:
        return np.exp(series)

    def SQRT(self, series: Any) -> Any:
        return np.sqrt(series)

    def POW(self, left: Any, right: Any) -> Any:
        return elementwise_binary("^", left, right, self.index)

    def ROUND(self, value: Any, digits: Any = 0) -> Any:
        decimals = ensure_int(digits, default=0)
        if isinstance(value, pd.Series):
            return value.round(decimals)
        return round(value, decimals)

    def COUNT(self, condition: Any, periods: Any = 5) -> pd.Series:
        data = self._as_series(condition).astype(bool).astype(int)
        window = ensure_int(periods, default=5)
        return data.cumsum() if window <= 0 else data.rolling(window=window, min_periods=1).sum()

    def EVERY(self, condition: Any, periods: Any = 5) -> pd.Series:
        data = self._as_series(condition).astype(bool)
        window = ensure_int(periods, default=5)
        if window <= 0:
            return data.expanding().min().astype(bool)
        return data.rolling(window=window, min_periods=1).min().astype(bool)

    def EXIST(self, condition: Any, periods: Any = 5) -> pd.Series:
        data = self._as_series(condition).astype(bool)
        window = ensure_int(periods, default=5)
        if window <= 0:
            return data.expanding().max().astype(bool)
        return data.rolling(window=window, min_periods=1).max().astype(bool)

    def UPNDAY(self, series: Any, periods: Any = 1) -> pd.Series:
        data = self._as_series(series)
        window = ensure_int(periods, default=1)
        return self.EVERY(data.diff() > 0, window)

    def DOWNNDAY(self, series: Any, periods: Any = 1) -> pd.Series:
        data = self._as_series(series)
        window = ensure_int(periods, default=1)
        return self.EVERY(data.diff() < 0, window)

    def CROSS(self, left: Any, right: Any) -> pd.Series:
        left = self._as_series(left)
        right = self._as_series(right)
        return (left > right) & (left.shift(1) <= right.shift(1))

    def LONGCROSS(self, left: Any, right: Any, periods: Any = 5) -> pd.Series:
        left = self._as_series(left)
        right = self._as_series(right)
        window = ensure_int(periods, default=5)
        below = (left < right).shift(1).fillna(False)
        must_below = below.rolling(window=window, min_periods=window).min().fillna(0).astype(bool)
        return self.CROSS(left, right) & must_below

    def UP(self, series: Any, periods: Any = 1) -> pd.Series:
        data = self._as_series(series)
        return data > data.shift(ensure_int(periods, default=1))

    def DOWN(self, series: Any, periods: Any = 1) -> pd.Series:
        data = self._as_series(series)
        return data < data.shift(ensure_int(periods, default=1))

    def IF(self, condition: Any, true_value: Any, false_value: Any = 0) -> pd.Series:
        condition = self._as_series(condition).astype(bool)
        true_series = normalize_result(true_value, self.index)
        false_series = normalize_result(false_value, self.index)
        return pd.Series(np.where(condition, true_series, false_series), index=self.index)

    def BETWEEN(self, value: Any, lower: Any, upper: Any) -> Any:
        return elementwise_binary("<=", lower, value, self.index) & elementwise_binary("<=", value, upper, self.index)

    def MAX(self, left: Any, right: Any) -> Any:
        left, right = _align(left, right, self.index)
        return np.maximum(left, right)

    def MIN(self, left: Any, right: Any) -> Any:
        left, right = _align(left, right, self.index)
        return np.minimum(left, right)

    def BARSLAST(self, condition: Any) -> pd.Series:
        cond = self._as_series(condition).astype(bool).fillna(False)
        result = []
        last_true = None
        for i, flag in enumerate(cond):
            if flag:
                last_true = i
                result.append(0)
            elif last_true is None:
                result.append(i + 1)
            else:
                result.append(i - last_true)
        return _series(result, self.index)

    def BARSSINCE(self, condition: Any) -> pd.Series:
        cond = self._as_series(condition).astype(bool).fillna(False)
        result = []
        first_true = None
        for i, flag in enumerate(cond):
            if flag and first_true is None:
                first_true = i
            result.append(np.nan if first_true is None else i - first_true)
        return _series(result, self.index)

    def BARSCOUNT(self, value: Any) -> pd.Series:
        series = self._as_series(value)
        first_valid = series.first_valid_index()
        result = []
        started = first_valid is None
        count = 0
        for idx, current in series.items():
            if not started and idx == first_valid:
                started = True
            if started:
                count += 1
                result.append(count)
            else:
                result.append(0)
        return _series(result, self.index)

    def FILTER(self, condition: Any, periods: Any = 1) -> pd.Series:
        cond = self._as_series(condition).astype(bool).fillna(False)
        window = max(0, ensure_int(periods, default=1))
        cooldown = 0
        result = []
        for flag in cond:
            if cooldown > 0:
                result.append(False)
                cooldown -= 1
                continue
            if flag:
                result.append(True)
                cooldown = window
            else:
                result.append(False)
        return _series(result, self.index)

    def BACKSET(self, condition: Any, periods: Any = 1) -> pd.Series:
        cond = self._as_series(condition).astype(bool).fillna(False)
        window = max(0, ensure_int(periods, default=1))
        result = pd.Series(False, index=self.index)
        for i, flag in enumerate(cond):
            if flag:
                start = max(0, i - window)
                result.iloc[start : i + 1] = True
        return result

    def VALUEWHEN(self, condition: Any, value: Any) -> pd.Series:
        cond = self._as_series(condition).astype(bool)
        values = normalize_result(value, self.index)
        return values.where(cond).ffill()

    def CONST(self, value: Any) -> pd.Series:
        if isinstance(value, pd.Series):
            valid = value.dropna()
            constant = np.nan if valid.empty else valid.iloc[-1]
        else:
            constant = value
        return pd.Series(np.repeat(constant, len(self.index)), index=self.index)

    def TSRANK(self, value: Any, periods: Any = 20) -> pd.Series:
        series = self._as_series(value).astype(float)
        window = ensure_int(periods, default=20)
        if window <= 1:
            return pd.Series(1.0, index=self.index)
        return series.rolling(window=window, min_periods=1).apply(
            lambda s: pd.Series(s).rank(pct=True).iloc[-1],
            raw=False,
        )

    def PCTRANK(self, value: Any, periods: Any = 20) -> pd.Series:
        return self.TSRANK(value, periods)

    def DIFF(self, value: Any, periods: Any = 1) -> pd.Series:
        return self._as_series(value).diff(ensure_int(periods, default=1))

    def RANGEPOS(self, value: Any, periods: Any = 20) -> pd.Series:
        series = self._as_series(value).astype(float)
        window = ensure_int(periods, default=20)
        if window <= 0:
            raise FormulaExecutionError("RANGEPOS 的周期必须大于 0")
        lowest = series.rolling(window=window, min_periods=1).min()
        highest = series.rolling(window=window, min_periods=1).max()
        spread = (highest - lowest).replace(0, np.nan)
        return (series - lowest) / spread

    def MAXDRAWDOWN(self, value: Any, periods: Any = 20) -> pd.Series:
        series = self._as_series(value).astype(float)
        window = ensure_int(periods, default=20)
        if window <= 0:
            raise FormulaExecutionError("MAXDRAWDOWN 的周期必须大于 0")
        return series.rolling(window=window, min_periods=1).apply(
            lambda s: ((pd.Series(s) / pd.Series(s).cummax()) - 1.0).min(),
            raw=False,
        )

    def registry(self) -> Dict[str, Callable[..., Any]]:
        return {
            "MA": self.MA,
            "SMA": self.SMA,
            "EMA": self.EMA,
            "RSI": self.RSI,
            "MACD": self.MACD,
            "ADX": self.ADX,
            "CCI": self.CCI,
            "ATR": self.ATR,
            "BBANDS": self.BBANDS,
            "OBV": self.OBV,
            "STOCH": self.STOCH,
            "STOCHRSI": self.STOCHRSI,
            "WILLR": self.WILLR,
            "KAMA": self.KAMA,
            "ROC": self.ROC,
            "MOM": self.MOM,
            "REF": self.REF,
            "HHV": self.HHV,
            "LLV": self.LLV,
            "SUM": self.SUM,
            "AVE": self.AVE,
            "STD": self.STD,
            "ABS": self.ABS,
            "LOG": self.LOG,
            "LN": self.LN,
            "EXP": self.EXP,
            "SQRT": self.SQRT,
            "POW": self.POW,
            "ROUND": self.ROUND,
            "COUNT": self.COUNT,
            "EVERY": self.EVERY,
            "EXIST": self.EXIST,
            "UPNDAY": self.UPNDAY,
            "DOWNNDAY": self.DOWNNDAY,
            "CROSS": self.CROSS,
            "LONGCROSS": self.LONGCROSS,
            "UP": self.UP,
            "DOWN": self.DOWN,
            "IF": self.IF,
            "BETWEEN": self.BETWEEN,
            "MAX": self.MAX,
            "MIN": self.MIN,
            "BARSLAST": self.BARSLAST,
            "BARSSINCE": self.BARSSINCE,
            "BARSCOUNT": self.BARSCOUNT,
            "FILTER": self.FILTER,
            "BACKSET": self.BACKSET,
            "VALUEWHEN": self.VALUEWHEN,
            "CONST": self.CONST,
            "TSRANK": self.TSRANK,
            "PCTRANK": self.PCTRANK,
            "DIFF": self.DIFF,
            "RANGEPOS": self.RANGEPOS,
            "MAXDRAWDOWN": self.MAXDRAWDOWN,
        }


@dataclass
class FormulaRuntime:
    """统一运行时上下文。"""

    df: pd.DataFrame

    def __post_init__(self) -> None:
        if "date" in self.df.columns:
            prepared = self.df.copy()
            prepared["date"] = pd.to_datetime(prepared["date"])
            prepared = prepared.set_index("date")
            self.df = prepared
        elif not isinstance(self.df.index, pd.DatetimeIndex):
            self.df = self.df.copy()
        self.df = self.df.sort_index()
        self.index = self.df.index
        self.functions = FormulaFunctions(self.index)
        self.function_registry = {name.upper(): func for name, func in self.functions.registry().items()}
        self.base_variables = self._build_base_variables()
        self.safe_modules = self._build_safe_modules()

    def _build_base_variables(self) -> Dict[str, Any]:
        amount = self.df["amount"] if "amount" in self.df.columns else pd.Series(np.nan, index=self.index)
        amplitude = self.df["amplitude"] if "amplitude" in self.df.columns else pd.Series(np.nan, index=self.index)
        pct_change = self.df["pct_change"] if "pct_change" in self.df.columns else pd.Series(np.nan, index=self.index)
        turnover = self.df["turnover"] if "turnover" in self.df.columns else pd.Series(np.nan, index=self.index)

        variables = {
            "df": self.df,
            "open": self.df["open"],
            "high": self.df["high"],
            "low": self.df["low"],
            "close": self.df["close"],
            "volume": self.df["volume"],
            "amount": amount,
            "amplitude": amplitude,
            "pct_change": pct_change,
            "turnover": turnover,
            "date": pd.Series(self.index.strftime("%Y%m%d").astype(int), index=self.index)
            if isinstance(self.index, pd.DatetimeIndex)
            else pd.Series(np.nan, index=self.index),
            "time": pd.Series(np.zeros(len(self.index), dtype=int), index=self.index),
            "year": pd.Series(self.index.year, index=self.index) if isinstance(self.index, pd.DatetimeIndex) else pd.Series(np.nan, index=self.index),
            "month": pd.Series(self.index.month, index=self.index) if isinstance(self.index, pd.DatetimeIndex) else pd.Series(np.nan, index=self.index),
            "day": pd.Series(self.index.day, index=self.index) if isinstance(self.index, pd.DatetimeIndex) else pd.Series(np.nan, index=self.index),
            "weekday": pd.Series(self.index.dayofweek + 1, index=self.index) if isinstance(self.index, pd.DatetimeIndex) else pd.Series(np.nan, index=self.index),
        }

        aliases = {
            "O": variables["open"],
            "H": variables["high"],
            "L": variables["low"],
            "C": variables["close"],
            "V": variables["volume"],
            "AMOUNT": variables["amount"],
            "OPEN": variables["open"],
            "HIGH": variables["high"],
            "LOW": variables["low"],
            "CLOSE": variables["close"],
            "VOL": variables["volume"],
            "VOLUME": variables["volume"],
            "DATE": variables["date"],
            "TIME": variables["time"],
            "YEAR": variables["year"],
            "MONTH": variables["month"],
            "DAY": variables["day"],
            "WEEKDAY": variables["weekday"],
            "PCT_CHG": variables["pct_change"],
            "TURNOVER": variables["turnover"],
            "AMPLITUDE": variables["amplitude"],
        }

        variables.update(aliases)
        return variables

    def _build_safe_modules(self) -> Dict[str, SafeModule]:
        return {
            "np": SafeModule(
                name="np",
                attributes={
                    "log": np.log,
                    "sqrt": np.sqrt,
                    "exp": np.exp,
                    "abs": np.abs,
                    "where": np.where,
                    "maximum": np.maximum,
                    "minimum": np.minimum,
                    "clip": np.clip,
                    "nan": np.nan,
                },
            ),
            "pd": SafeModule(
                name="pd",
                attributes={
                    "Series": pd.Series,
                    "DataFrame": pd.DataFrame,
                    "Timestamp": pd.Timestamp,
                    "to_datetime": pd.to_datetime,
                },
            ),
        }

    def resolve_name(self, name: str, variables: Optional[Dict[str, Any]] = None) -> Any:
        namespace = variables or {}
        if name in namespace:
            return namespace[name]
        if name in self.base_variables:
            return self.base_variables[name]
        upper = name.upper()
        if upper in namespace:
            return namespace[upper]
        if upper in self.base_variables:
            return self.base_variables[upper]
        if upper in self.function_registry:
            return self.function_registry[upper]
        if name in self.safe_modules:
            return self.safe_modules[name]
        if name in {"True", "False", "None"}:
            return {"True": True, "False": False, "None": None}[name]
        raise FormulaExecutionError(f"未定义的名称: {name}")


def resolve_attribute(target: Any, attribute: str) -> Any:
    """解析安全属性。"""
    if isinstance(target, SafeModule):
        return target.resolve(attribute)

    if isinstance(target, pd.DataFrame):
        if attribute == "index":
            return target.index
        raise FormulaExecutionError(f"DataFrame 不允许访问属性 {attribute}")

    if isinstance(target, pd.Series):
        if attribute == "index":
            return target.index
        safe_methods = {
            "shift": lambda *args, **kwargs: target.shift(periods=ensure_int(kwargs.get("periods", args[0] if args else 1), default=1)),
            "diff": lambda *args, **kwargs: target.diff(periods=ensure_int(kwargs.get("periods", args[0] if args else 1), default=1)),
            "pct_change": lambda *args, **kwargs: target.pct_change(periods=ensure_int(kwargs.get("periods", args[0] if args else 1), default=1)),
            "abs": lambda *args, **kwargs: target.abs(),
            "round": lambda *args, **kwargs: target.round(decimals=ensure_int(kwargs.get("decimals", args[0] if args else 0), default=0)),
            "fillna": lambda *args, **kwargs: target.fillna(kwargs.get("value", args[0] if args else np.nan)),
            "clip": lambda *args, **kwargs: target.clip(
                lower=kwargs.get("lower", args[0] if args else None),
                upper=kwargs.get("upper", args[1] if len(args) > 1 else None),
            ),
            "astype": lambda *args, **kwargs: target.astype((args[0] if args else kwargs.get("dtype"))),
            "mean": lambda *args, **kwargs: target.mean(),
            "std": lambda *args, **kwargs: target.std(ddof=0),
            "max": lambda *args, **kwargs: target.max(),
            "min": lambda *args, **kwargs: target.min(),
            "median": lambda *args, **kwargs: target.median(),
            "sum": lambda *args, **kwargs: target.sum(),
            "skew": lambda *args, **kwargs: target.skew(),
            "kurt": lambda *args, **kwargs: target.kurt(),
            "kurtosis": lambda *args, **kwargs: target.kurt(),
            "quantile": lambda *args, **kwargs: target.quantile(ensure_scalar(args[0] if args else kwargs.get("q"), default=0.5)),
            "rank": lambda *args, **kwargs: target.rank(**kwargs),
            "rolling": lambda *args, **kwargs: RollingWindow(
                target,
                window=ensure_int(kwargs.get("window", args[0] if args else None), default=1),
                min_periods=ensure_int(kwargs.get("min_periods", 1), default=1),
            ),
            "expanding": lambda *args, **kwargs: ExpandingWindow(
                target,
                min_periods=ensure_int(kwargs.get("min_periods", 1), default=1),
            ),
            "ffill": lambda *args, **kwargs: target.ffill(),
            "bfill": lambda *args, **kwargs: target.bfill(),
            "replace": lambda *args, **kwargs: target.replace(*args, **kwargs),
        }
        if attribute not in safe_methods:
            raise FormulaExecutionError(f"Series 不允许访问属性 {attribute}")
        return safe_methods[attribute]

    if isinstance(target, RollingWindow):
        if not hasattr(target, attribute):
            raise FormulaExecutionError(f"RollingWindow 不允许访问属性 {attribute}")
        return getattr(target, attribute)

    if isinstance(target, ExpandingWindow):
        if not hasattr(target, attribute):
            raise FormulaExecutionError(f"ExpandingWindow 不允许访问属性 {attribute}")
        return getattr(target, attribute)

    if isinstance(target, pd.Index):
        if attribute == "name":
            return target.name
        raise FormulaExecutionError(f"Index 不允许访问属性 {attribute}")

    raise FormulaExecutionError(f"对象 {type(target)} 不允许访问属性 {attribute}")
