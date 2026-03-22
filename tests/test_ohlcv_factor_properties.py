"""
OHLCV 因子扩展属性测试（Hypothesis）
Tasks 3.1–3.3: body_ratio, efficiency_ratio_20, close_location_value 值域属性验证

**Validates: Requirements 4.1, 4.3, 4.4**
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from backend.services.factor_service import FactorCalculator


# ── OHLCV DataFrame 生成策略 ──────────────────────────────────────────────────

def ohlcv_dataframe(min_rows: int = 1, max_rows: int = 60):
    """
    生成合法 OHLCV DataFrame 的 Hypothesis 策略。
    约束：high >= close >= low > 0, high >= open >= low, volume >= 0
    """
    @st.composite
    def _strategy(draw):
        n = draw(st.integers(min_value=min_rows, max_value=max_rows))

        # 生成 low > 0
        lows = draw(
            st.lists(
                st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )

        # 生成 high >= low（high = low + delta, delta >= 0）
        deltas = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )

        highs = [l + d for l, d in zip(lows, deltas)]

        # close ∈ [low, high]
        close_fracs = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )
        closes = [l + f * (h - l) for l, h, f in zip(lows, highs, close_fracs)]

        # open ∈ [low, high]
        open_fracs = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )
        opens = [l + f * (h - l) for l, h, f in zip(lows, highs, open_fracs)]

        # volume >= 0
        volumes = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )

        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
            index=dates,
        )
        return df

    return _strategy()


def ohlcv_dataframe_min21():
    """序列长度 ≥ 21 的 OHLCV DataFrame 策略（用于 efficiency_ratio_20）"""
    return ohlcv_dataframe(min_rows=21, max_rows=80)


# ── Task 3.1: body_ratio ∈ [-1, 1] ───────────────────────────────────────────

class TestBodyRatioProperty:
    """
    Task 3.1: 属性测试 — 对任意合法 OHLCV 序列，body_ratio ∈ [-1, 1]

    **Validates: Requirements 4.1**
    """

    @given(df=ohlcv_dataframe())
    @settings(max_examples=50)
    def test_body_ratio_in_minus1_to_1(self, df):
        """body_ratio 对任意合法 OHLCV 序列，有效值必须在 [-1, 1]"""
        code = "(close - open) / (high - low + 1e-12)"
        calc = FactorCalculator()
        result = calc.calculate(df, code)

        valid = result.dropna()
        # 如果没有有效值（全 NaN），跳过
        assume(len(valid) > 0)

        assert (valid >= -1.0).all(), (
            f"body_ratio 存在小于 -1 的值: min={valid.min():.6f}"
        )
        assert (valid <= 1.0).all(), (
            f"body_ratio 存在大于 1 的值: max={valid.max():.6f}"
        )


# ── Task 3.2: efficiency_ratio_20 ∈ [0, 1]（序列长度 ≥ 21）─────────────────

class TestEfficiencyRatio20Property:
    """
    Task 3.2: 属性测试 — 对任意合法 OHLCV 序列（长度 ≥ 21），efficiency_ratio_20 ∈ [0, 1]

    **Validates: Requirements 4.4**
    """

    @given(df=ohlcv_dataframe_min21())
    @settings(max_examples=50)
    def test_efficiency_ratio_20_in_0_to_1(self, df):
        """efficiency_ratio_20 对任意合法 OHLCV 序列（长度 ≥ 21），有效值必须在 [0, 1]"""
        code = "np.abs(close - REF(close, 20)) / (SUM(np.abs(close - REF(close, 1)), 20) + 1e-12)"
        calc = FactorCalculator()
        result = calc.calculate(df, code)

        valid = result.dropna()
        # 序列长度 ≥ 21 时应有有效值
        assume(len(valid) > 0)

        assert (valid >= 0.0).all(), (
            f"efficiency_ratio_20 存在小于 0 的值: min={valid.min():.6f}"
        )
        assert (valid <= 1.0).all(), (
            f"efficiency_ratio_20 存在大于 1 的值: max={valid.max():.6f}"
        )


# ── Task 3.3: close_location_value ∈ [0, 1] ──────────────────────────────────

class TestCloseLocationValueProperty:
    """
    Task 3.3: 属性测试 — 对任意合法 OHLCV 序列，close_location_value ∈ [0, 1]

    **Validates: Requirements 4.3**
    """

    @given(df=ohlcv_dataframe())
    @settings(max_examples=50)
    def test_close_location_value_in_0_to_1(self, df):
        """close_location_value 对任意合法 OHLCV 序列，有效值必须在 [0, 1]"""
        code = "(close - low) / (high - low + 1e-12)"
        calc = FactorCalculator()
        result = calc.calculate(df, code)

        valid = result.dropna()
        assume(len(valid) > 0)

        assert (valid >= 0.0).all(), (
            f"close_location_value 存在小于 0 的值: min={valid.min():.6f}"
        )
        assert (valid <= 1.0).all(), (
            f"close_location_value 存在大于 1 的值: max={valid.max():.6f}"
        )
