"""
OHLCV 因子扩展单元测试
Task 2.1: 为 12 个 P0 因子编写可计算性测试（250 日合成数据，无异常）
"""
import numpy as np
import pandas as pd
import pytest

from backend.services.factor_service import FactorCalculator


# ── 共享 fixture ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ohlcv_250():
    """250 日干净 OHLCV 数据：high >= close >= low > 0, open > 0, volume > 0"""
    rng = np.random.default_rng(42)
    n = 250
    low   = rng.uniform(9.0, 10.0, n)
    high  = low + rng.uniform(0.5, 2.0, n)   # high > low
    close = low + rng.uniform(0.0, 1.0, n) * (high - low)  # low <= close <= high
    open_ = low + rng.uniform(0.0, 1.0, n) * (high - low)  # low <= open <= high
    volume = rng.uniform(1e6, 2e6, n)

    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ── P0 因子代码表 ─────────────────────────────────────────────────────────────

P0_FACTORS = [
    ("overnight_return_1",   "open / REF(close, 1) - 1"),
    ("intraday_return_1",    "close / open - 1"),
    ("body_ratio",           "(close - open) / (high - low + 1e-12)"),
    ("upper_shadow_ratio",   "(high - MAX(open, close)) / (high - low + 1e-12)"),
    ("close_location_value", "(close - low) / (high - low + 1e-12)"),
    ("breakout_strength_20", "(close - HHV(high, 20).shift(1)) / (ATR(high, low, close, timeperiod=14) + 1e-12)"),
    ("efficiency_ratio_20",  "np.abs(close - REF(close, 20)) / (SUM(np.abs(close - REF(close, 1)), 20) + 1e-12)"),
    ("kama_ratio_20",        "close / KAMA(close, timeperiod=20)"),
    ("parkinson_vol_20",     "np.sqrt((np.log(high / low) ** 2).rolling(20).mean() / (4 * np.log(2)))"),
    ("garman_klass_vol_20",  "np.sqrt((0.5 * (np.log(high / low) ** 2) - (2 * np.log(2) - 1) * (np.log(close / open) ** 2)).rolling(20).mean())"),
    ("cmf_20",               "SUM((((close - low) - (high - close)) / (high - low + 1e-12)) * volume, 20) / (SUM(volume, 20) + 1e-12)"),
    ("volume_zscore_20",     "(volume - AVE(volume, 20)) / (STD(volume, 20) + 1e-12)"),
]

P0_FACTOR_NAMES = [name for name, _ in P0_FACTORS]
P0_FACTOR_CODES = {name: code for name, code in P0_FACTORS}


# ── Task 2.1: 可计算性测试 ────────────────────────────────────────────────────

class TestP0FactorComputability:
    """Task 2.1: 12 个 P0 因子可计算性测试（250 日合成数据，无异常）"""

    @pytest.mark.parametrize("factor_name", P0_FACTOR_NAMES)
    def test_no_exception(self, ohlcv_250, factor_name):
        """每个 P0 因子在 250 日干净数据上计算不抛出异常"""
        calc = FactorCalculator()
        code = P0_FACTOR_CODES[factor_name]
        # Should not raise
        result = calc.calculate(ohlcv_250, code)
        assert result is not None, f"{factor_name}: calculate() returned None"

    @pytest.mark.parametrize("factor_name", P0_FACTOR_NAMES)
    def test_returns_series_of_length_250(self, ohlcv_250, factor_name):
        """每个 P0 因子返回长度为 250 的 pd.Series"""
        calc = FactorCalculator()
        code = P0_FACTOR_CODES[factor_name]
        result = calc.calculate(ohlcv_250, code)
        assert isinstance(result, pd.Series), (
            f"{factor_name}: expected pd.Series, got {type(result)}"
        )
        assert len(result) == 250, (
            f"{factor_name}: expected length 250, got {len(result)}"
        )


# ── Task 2.2: warmup NaN 数量验证 ────────────────────────────────────────────

EXPECTED_WARMUP_NANS = {
    # overnight_return_1 uses REF(close, 1) = shift(1), so row 0 is NaN
    "overnight_return_1":   1,
    # intraday_return_1 = close/open - 1, no lookback needed
    "intraday_return_1":    0,
    # body_ratio, upper_shadow_ratio, close_location_value: pure same-bar arithmetic
    "body_ratio":           0,
    "upper_shadow_ratio":   0,
    "close_location_value": 0,
    # breakout_strength_20: ATR(timeperiod=14) drives warmup (TA-Lib strict min_periods)
    # HHV uses min_periods=1 so no extra NaNs; ATR produces 14 leading NaNs
    "breakout_strength_20": 14,
    # efficiency_ratio_20: REF(close, 20) = shift(20) → 20 leading NaNs
    "efficiency_ratio_20":  20,
    # parkinson_vol_20: rolling(20).mean() with default min_periods=20 → 19 leading NaNs
    "parkinson_vol_20":     19,
    # garman_klass_vol_20: rolling(20).mean() with default min_periods=20 → 19 leading NaNs
    "garman_klass_vol_20":  19,
    # cmf_20: SUM() uses min_periods=1, so no leading NaNs
    "cmf_20":               0,
    # volume_zscore_20: STD() uses min_periods=1, returns NaN for single-element window
    "volume_zscore_20":     1,
    # kama_ratio_20: KAMA(timeperiod=20) via TA-Lib → 20 leading NaNs
    "kama_ratio_20":        20,
}


@pytest.mark.requires_talib
class TestP0FactorWarmupNaN:
    """Task 2.2: 验证每个 P0 因子的 warmup NaN 数量（需要真实 talib）"""

    @pytest.mark.parametrize("factor_name,expected_nans", list(EXPECTED_WARMUP_NANS.items()))
    def test_warmup_nan_count(self, ohlcv_250, factor_name, expected_nans):
        """每个因子的前 N 行应为 NaN，N 等于预期 warmup 行数"""
        calc = FactorCalculator()
        code = P0_FACTOR_CODES[factor_name]
        result = calc.calculate(ohlcv_250, code)

        # Count leading NaNs
        leading_nans = 0
        for val in result:
            if pd.isna(val):
                leading_nans += 1
            else:
                break

        assert leading_nans == expected_nans, (
            f"{factor_name}: expected {expected_nans} leading NaNs, got {leading_nans}"
        )

    @pytest.mark.parametrize("factor_name,expected_nans", list(EXPECTED_WARMUP_NANS.items()))
    def test_non_nan_after_warmup(self, ohlcv_250, factor_name, expected_nans):
        """warmup 之后至少有一个有效值（非全 NaN）"""
        calc = FactorCalculator()
        code = P0_FACTOR_CODES[factor_name]
        result = calc.calculate(ohlcv_250, code)

        post_warmup = result.iloc[expected_nans:]
        assert post_warmup.notna().any(), (
            f"{factor_name}: all values after warmup are NaN"
        )


# ── Task 2.3: 有界因子值域约束测试 ───────────────────────────────────────────

class TestBoundedFactorRanges:
    """Task 2.3: 有界因子值域约束测试"""

    def _valid_values(self, ohlcv_250, factor_name):
        """返回非 NaN 的有效值"""
        calc = FactorCalculator()
        code = P0_FACTOR_CODES[factor_name]
        result = calc.calculate(ohlcv_250, code)
        return result.dropna()

    def test_body_ratio_in_minus1_to_1(self, ohlcv_250):
        """body_ratio 有效值必须在 [-1, 1]"""
        valid = self._valid_values(ohlcv_250, "body_ratio")
        assert len(valid) > 0, "body_ratio: no valid values"
        assert (valid >= -1).all() and (valid <= 1).all(), (
            f"body_ratio out of [-1, 1]: min={valid.min():.4f}, max={valid.max():.4f}"
        )

    def test_upper_shadow_ratio_in_0_to_1(self, ohlcv_250):
        """upper_shadow_ratio 有效值必须在 [0, 1]"""
        valid = self._valid_values(ohlcv_250, "upper_shadow_ratio")
        assert len(valid) > 0, "upper_shadow_ratio: no valid values"
        assert (valid >= 0).all() and (valid <= 1).all(), (
            f"upper_shadow_ratio out of [0, 1]: min={valid.min():.4f}, max={valid.max():.4f}"
        )

    def test_close_location_value_in_0_to_1(self, ohlcv_250):
        """close_location_value 有效值必须在 [0, 1]"""
        valid = self._valid_values(ohlcv_250, "close_location_value")
        assert len(valid) > 0, "close_location_value: no valid values"
        assert (valid >= 0).all() and (valid <= 1).all(), (
            f"close_location_value out of [0, 1]: min={valid.min():.4f}, max={valid.max():.4f}"
        )

    def test_efficiency_ratio_20_in_0_to_1(self, ohlcv_250):
        """efficiency_ratio_20 有效值必须在 [0, 1]"""
        valid = self._valid_values(ohlcv_250, "efficiency_ratio_20")
        assert len(valid) > 0, "efficiency_ratio_20: no valid values"
        assert (valid >= 0).all() and (valid <= 1).all(), (
            f"efficiency_ratio_20 out of [0, 1]: min={valid.min():.4f}, max={valid.max():.4f}"
        )

    def test_cmf_20_in_minus1_to_1(self, ohlcv_250):
        """cmf_20 有效值必须在 [-1, 1]"""
        valid = self._valid_values(ohlcv_250, "cmf_20")
        assert len(valid) > 0, "cmf_20: no valid values"
        assert (valid >= -1).all() and (valid <= 1).all(), (
            f"cmf_20 out of [-1, 1]: min={valid.min():.4f}, max={valid.max():.4f}"
        )


# ── Task 2.4: parkinson_vol_20 / garman_klass_vol_20 非负约束 ─────────────────

class TestVolatilityFactorsNonNegative:
    """Task 2.4: parkinson_vol_20 / garman_klass_vol_20 非负约束测试"""

    def test_parkinson_vol_20_non_negative(self, ohlcv_250):
        """parkinson_vol_20 有效值必须 >= 0"""
        calc = FactorCalculator()
        result = calc.calculate(ohlcv_250, P0_FACTOR_CODES["parkinson_vol_20"])
        valid = result.dropna()
        assert len(valid) > 0, "parkinson_vol_20: no valid values"
        assert (valid >= 0).all(), (
            f"parkinson_vol_20 has negative values: min={valid.min():.6f}"
        )

    def test_garman_klass_vol_20_non_negative(self, ohlcv_250):
        """garman_klass_vol_20 有效值必须 >= 0"""
        calc = FactorCalculator()
        result = calc.calculate(ohlcv_250, P0_FACTOR_CODES["garman_klass_vol_20"])
        valid = result.dropna()
        assert len(valid) > 0, "garman_klass_vol_20: no valid values"
        assert (valid >= 0).all(), (
            f"garman_klass_vol_20 has negative values: min={valid.min():.6f}"
        )


# ── Task 2.5: 边界条件测试：high == low（一字板）────────────────────────────

@pytest.fixture(scope="module")
def ohlcv_flat_price():
    """250 日一字板数据：high == low == close == open，volume > 0"""
    n = 250
    price = 10.0
    volume = np.full(n, 1e6)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open":   np.full(n, price),
            "high":   np.full(n, price),
            "low":    np.full(n, price),
            "close":  np.full(n, price),
            "volume": volume,
        },
        index=dates,
    )


class TestFlatPriceBoundary:
    """Task 2.5: high == low（一字板）边界条件测试"""

    FLAT_FACTORS = ["body_ratio", "upper_shadow_ratio", "close_location_value", "cmf_20"]

    @pytest.mark.parametrize("factor_name", FLAT_FACTORS)
    def test_finite_values_when_high_equals_low(self, ohlcv_flat_price, factor_name):
        """一字板时，因子返回有限数（非 inf，warmup 后非 NaN）"""
        calc = FactorCalculator()
        code = P0_FACTOR_CODES[factor_name]
        result = calc.calculate(ohlcv_flat_price, code)

        warmup = EXPECTED_WARMUP_NANS.get(factor_name, 0)
        post_warmup = result.iloc[warmup:]

        # No inf values in the entire series
        assert not np.isinf(result.replace(np.nan, 0)).any(), (
            f"{factor_name}: contains inf values in flat-price scenario"
        )
        # After warmup, values should be finite (not NaN)
        assert post_warmup.notna().all(), (
            f"{factor_name}: NaN values after warmup in flat-price scenario"
        )
        assert np.isfinite(post_warmup).all(), (
            f"{factor_name}: non-finite values after warmup in flat-price scenario"
        )


# ── Task 2.6: 边界条件测试：volume == 0 ──────────────────────────────────────

@pytest.fixture(scope="module")
def ohlcv_zero_volume():
    """250 日零成交量数据：volume == 0，OHLCV 正常"""
    rng = np.random.default_rng(99)
    n = 250
    low   = rng.uniform(9.0, 10.0, n)
    high  = low + rng.uniform(0.5, 2.0, n)
    close = low + rng.uniform(0.0, 1.0, n) * (high - low)
    open_ = low + rng.uniform(0.0, 1.0, n) * (high - low)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open":   open_,
            "high":   high,
            "low":    low,
            "close":  close,
            "volume": np.zeros(n),
        },
        index=dates,
    )


class TestZeroVolumeBoundary:
    """Task 2.6: volume == 0 边界条件测试"""

    @pytest.mark.parametrize("factor_name", ["volume_zscore_20", "cmf_20"])
    def test_finite_values_when_volume_zero(self, ohlcv_zero_volume, factor_name):
        """volume == 0 时，因子返回有限数（非 inf）"""
        calc = FactorCalculator()
        code = P0_FACTOR_CODES[factor_name]
        result = calc.calculate(ohlcv_zero_volume, code)

        # No inf values anywhere
        assert not np.isinf(result.replace(np.nan, 0)).any(), (
            f"{factor_name}: contains inf values when volume == 0"
        )


# ── Task 2.7: 验证 breakout_strength_20 使用 .shift(1) 不含未来函数 ──────────

@pytest.mark.requires_talib
class TestBreakoutStrengthNoLookahead:
    """Task 2.7: 验证 breakout_strength_20 使用 .shift(1) 不含未来函数（需要真实 talib）"""

    def test_shift1_differs_from_no_shift(self, ohlcv_250):
        """带 .shift(1) 的版本与不带 .shift(1) 的版本结果不同，证明 shift 有效"""
        calc = FactorCalculator()

        # Correct version: uses .shift(1) to avoid lookahead
        correct_code = (
            "(close - HHV(high, 20).shift(1)) / (ATR(high, low, close, timeperiod=14) + 1e-12)"
        )
        # Lookahead version: no .shift(1)
        lookahead_code = (
            "(close - HHV(high, 20)) / (ATR(high, low, close, timeperiod=14) + 1e-12)"
        )

        correct = calc.calculate(ohlcv_250, correct_code)
        lookahead = calc.calculate(ohlcv_250, lookahead_code)

        # The two series must differ (shift changes the result)
        assert not correct.equals(lookahead), (
            "breakout_strength_20: shift(1) version equals no-shift version — "
            "shift(1) is not having any effect!"
        )

        # Verify they differ in at least some non-NaN positions
        both_valid = correct.notna() & lookahead.notna()
        assert both_valid.any(), "No overlapping valid values to compare"
        assert not (correct[both_valid] == lookahead[both_valid]).all(), (
            "breakout_strength_20: shift(1) and no-shift produce identical valid values"
        )
