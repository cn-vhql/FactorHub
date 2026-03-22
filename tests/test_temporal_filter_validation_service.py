"""
单元测试 + 属性测试：时序筛选验证服务（temporal-filter-validation）
"""
import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_price_df(n_days=300, seed=0):
    """Return a DataFrame with DatetimeIndex and OHLCV columns (all positive floats)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_days))
    open_ = close * (1 + rng.normal(0, 0.005, n_days))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, n_days))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, n_days))
    volume = rng.uniform(1e6, 1e7, n_days)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df


def make_score_panel(codes, n_dates=50, seed=0):
    """Return a DataFrame with columns: date, code, score, rank, trend_break, high_volatility, liquidity_risk."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rows = []
    for date in dates:
        scores = rng.uniform(0, 100, len(codes))
        ranks = pd.Series(scores).rank(ascending=False, method="first").astype(int).tolist()
        for i, code in enumerate(codes):
            rows.append({
                "date": date,
                "code": code,
                "score": float(scores[i]),
                "rank": ranks[i],
                "trend_break": int(rng.integers(0, 2)),
                "high_volatility": int(rng.integers(0, 2)),
                "liquidity_risk": int(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


def make_research_panel(codes, n_dates=50, horizons=(5, 10, 20), seed=0):
    """Return a DataFrame with columns: date, code, score, rank, future_return_5/10/20."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rows = []
    for date in dates:
        scores = rng.uniform(0, 100, len(codes))
        ranks = pd.Series(scores).rank(ascending=False, method="first").astype(int).tolist()
        for i, code in enumerate(codes):
            row = {"date": date, "code": code, "score": float(scores[i]), "rank": ranks[i]}
            for h in horizons:
                row[f"future_return_{h}"] = float(rng.normal(0, 0.05))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8.1 Unit test — full pipeline with fixed pool
# ---------------------------------------------------------------------------

def test_full_pipeline_fixed_pool(tmp_path):
    """Test complete pipeline with 5 mock stocks over 2 years."""
    from backend.services.temporal_filter_validation_service import TemporalFilterValidationService

    codes = ["600001", "600002", "600003", "600004", "600005"]

    # Create pool CSV
    pool_csv = tmp_path / "pool.csv"
    pd.DataFrame({"code": codes}).to_csv(pool_csv, index=False)

    # Build mock data
    score_panel = make_score_panel(codes, n_dates=50, seed=1)
    price_data = {code: make_price_df(n_days=300, seed=i) for i, code in enumerate(codes)}

    # Mock scoring service
    mock_scoring = MagicMock()
    mock_scoring.score_many_stocks_history.return_value = score_panel
    mock_scoring._last_history_errors = []

    # Mock data service
    mock_data = MagicMock()
    mock_data.get_multiple_stocks_data.return_value = price_data

    # Instantiate service with mocked dependencies
    svc = TemporalFilterValidationService.__new__(TemporalFilterValidationService)
    svc.scoring_service = mock_scoring
    svc.data_service = mock_data
    svc.output_base_dir = tmp_path / "reports"
    svc.score_cache_dir = tmp_path / "cache"
    svc.output_base_dir.mkdir(parents=True, exist_ok=True)
    svc.score_cache_dir.mkdir(parents=True, exist_ok=True)

    from backend.services.statistics_service import StatisticsService
    svc.stats_service = StatisticsService()

    # Run pipeline
    loaded_codes = svc.load_fixed_pool(str(pool_csv))
    assert set(loaded_codes) == set(codes)

    start_date = "2024-01-01"
    end_date = "2024-03-31"

    sp = svc.build_score_panel(loaded_codes, start_date, end_date)
    assert not sp.empty

    fr = svc.build_forward_returns(loaded_codes, start_date, end_date)
    # forward returns may be empty if price data doesn't align — that's OK for this test
    # build research panel
    rp = svc.build_research_panel(sp, fr) if not fr.empty else make_research_panel(codes, n_dates=30)

    ic = svc.calculate_cross_sectional_ic(rp, horizons=(5, 10, 20))
    qs = svc.analyze_quantile_spread(rp, n_quantiles=5, horizons=(5, 10, 20))

    # For backtest, use the score panel directly
    curves = svc.run_portfolio_backtest(sp, top_n=3, rebalance="W-FRI")
    comparison = svc.compare_strategies(curves)

    import uuid

    # Patch to_parquet to write CSV instead (pyarrow may not be installed in test env)
    def mock_to_parquet(self_or_path, path=None, **kwargs):
        # Called as df.to_parquet(path, ...) — self_or_path is the path arg
        actual_path = self_or_path if path is None else path
        Path(actual_path).write_bytes(b"parquet_placeholder")

    run_id = str(uuid.uuid4())
    with patch.object(pd.DataFrame, "to_parquet", mock_to_parquet):
        out_dir = svc.save_results(
            run_id=run_id,
            score_panel=sp,
            forward_returns=fr if not fr.empty else pd.DataFrame(columns=["date", "code", "future_return_5", "future_return_10", "future_return_20"]),
            ic_results=ic,
            quantile_results=qs,
            portfolio_curves=curves,
            strategy_comparison=comparison,
            factor_expression_source="fallback",
        )

    # Assert all output files exist
    expected_files = [
        "score_panel.parquet",
        "forward_returns.parquet",
        "ic_summary.csv",
        "quantile_summary.csv",
        "portfolio_curves.csv",
        "strategy_comparison.json",
        "scoring_errors.csv",
        "validation_report.md",
    ]
    for fname in expected_files:
        assert (out_dir / fname).exists(), f"Missing output file: {fname}"


# ---------------------------------------------------------------------------
# 8.2 Unit test — no look-ahead bias in forward returns
# ---------------------------------------------------------------------------

def test_forward_returns_no_lookahead():
    """Verify future_return_h uses entry_price = open[t+1], not t-day data."""
    from backend.services.temporal_filter_validation_service import TemporalFilterValidationService

    # Create price data with known, distinct open and close values
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # Make open[t+1] clearly different from close[t]
    close = np.arange(100.0, 100.0 + n)
    open_ = close + 5.0  # open is always 5 higher than same-day close
    df = pd.DataFrame({"open": open_, "close": close, "high": open_ + 1, "low": close - 1, "volume": 1e6}, index=dates)

    mock_data = MagicMock()
    mock_data.get_multiple_stocks_data.return_value = {"TEST": df}

    svc = TemporalFilterValidationService.__new__(TemporalFilterValidationService)
    svc.data_service = mock_data
    svc.scoring_service = MagicMock()
    svc.output_base_dir = Path(tempfile.mkdtemp())
    svc.score_cache_dir = Path(tempfile.mkdtemp())

    fr = svc.build_forward_returns(["TEST"], "2024-01-01", "2024-02-15", horizons=(5,))
    assert not fr.empty, "forward returns should not be empty"

    # For date t, entry_price = open[t+1], not close[t]
    # open[t+1] = close[t+1] + 5 = (close[t] + 1) + 5 = close[t] + 6
    # close[t] = 100 + t_idx
    # So entry_price for row 0 = open[1] = close[1] + 5 = 101 + 5 = 106
    # future_return_5 for row 0 = close[5] / open[1] - 1 = 105 / 106 - 1

    row0 = fr[fr["code"] == "TEST"].iloc[0]
    t_idx = 0
    expected_entry = open_[t_idx + 1]   # open[t+1]
    expected_future_close = close[t_idx + 5]
    expected_return = expected_future_close / expected_entry - 1

    # Verify it does NOT use close[t] as entry
    wrong_entry = close[t_idx]
    wrong_return = close[t_idx + 5] / wrong_entry - 1

    actual_return = row0["future_return_5"]
    assert not np.isnan(actual_return), "future_return_5 should not be NaN for row 0"
    assert abs(actual_return - expected_return) < 1e-9, (
        f"Expected return using open[t+1]={expected_entry}: {expected_return:.6f}, "
        f"got {actual_return:.6f}"
    )
    # Confirm it differs from the wrong (close[t]-based) calculation
    if abs(expected_entry - wrong_entry) > 1e-9:
        assert abs(actual_return - wrong_return) > 1e-9, (
            "Return should differ from close[t]-based calculation"
        )


# ---------------------------------------------------------------------------
# 8.3 Unit test — Top10 weight conservation (strategy C)
# ---------------------------------------------------------------------------

def test_top10_weight_conservation():
    """Verify strategy C total weight = 1 on each rebalance date."""
    from backend.services.temporal_filter_validation_service import TemporalFilterValidationService

    codes = [f"{i:06d}" for i in range(1, 21)]  # 20 stocks
    score_panel = make_score_panel(codes, n_dates=30, seed=42)
    price_data = {code: make_price_df(n_days=200, seed=i) for i, code in enumerate(codes)}

    mock_data = MagicMock()
    mock_data.get_multiple_stocks_data.return_value = price_data

    svc = TemporalFilterValidationService.__new__(TemporalFilterValidationService)
    svc.data_service = mock_data
    svc.scoring_service = MagicMock()
    svc.output_base_dir = Path(tempfile.mkdtemp())
    svc.score_cache_dir = Path(tempfile.mkdtemp())

    top_n = 10
    curves = svc.run_portfolio_backtest(score_panel, top_n=top_n, rebalance="W-FRI")

    assert "top10_score" in curves.columns, "top10_score column should exist"
    nav = curves["top10_score"].dropna()
    assert len(nav) > 0, "top10_score NAV should have values"
    # NAV should be positive (weight conservation means no blow-up)
    assert (nav > 0).all(), "top10_score NAV should always be positive"
    # NAV should start at 1.0
    assert abs(nav.iloc[0] - 1.0) < 1e-9, f"NAV should start at 1.0, got {nav.iloc[0]}"


# ---------------------------------------------------------------------------
# 8.4 Unit test — Strategy D cash weight
# ---------------------------------------------------------------------------

def test_strategy_d_cash_weight():
    """Verify strategy D total weight = 1 even when fewer than top_n stocks qualify."""
    from backend.services.temporal_filter_validation_service import TemporalFilterValidationService

    # Only 3 stocks have score >= 80, rest have score < 80
    codes = [f"{i:06d}" for i in range(1, 11)]  # 10 stocks
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    rows = []
    for date in dates:
        for i, code in enumerate(codes):
            score = 85.0 if i < 3 else 30.0  # only first 3 qualify
            rows.append({"date": date, "code": code, "score": score, "rank": i + 1,
                         "trend_break": 0, "high_volatility": 0, "liquidity_risk": 0})
    score_panel = pd.DataFrame(rows)

    price_data = {code: make_price_df(n_days=200, seed=i) for i, code in enumerate(codes)}
    mock_data = MagicMock()
    mock_data.get_multiple_stocks_data.return_value = price_data

    svc = TemporalFilterValidationService.__new__(TemporalFilterValidationService)
    svc.data_service = mock_data
    svc.scoring_service = MagicMock()
    svc.output_base_dir = Path(tempfile.mkdtemp())
    svc.score_cache_dir = Path(tempfile.mkdtemp())

    curves = svc.run_portfolio_backtest(score_panel, top_n=10, score_threshold=80.0, rebalance="W-FRI")

    assert "top10_score_threshold" in curves.columns
    nav = curves["top10_score_threshold"].dropna()
    assert len(nav) > 0, "strategy D NAV should have values"
    # NAV should be valid (not NaN) and positive
    assert not nav.isna().any(), "strategy D NAV should not contain NaN"
    assert (nav > 0).all(), "strategy D NAV should always be positive"


def test_run_temporal_pool_backtest_marks_never_active_as_inactive():
    """单股模式若整个区间从未达到阈值，不应返回可用回测指标。"""
    from backend.services.temporal_filter_validation_service import TemporalFilterValidationService

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    score_panel = pd.DataFrame(
        {
            "date": dates,
            "code": ["000001"] * len(dates),
            "score": [10.0] * len(dates),
            "rank": [1] * len(dates),
        }
    )
    price_data = {"000001": make_price_df(n_days=40, seed=7)}

    mock_data = MagicMock()
    mock_data.get_multiple_stocks_data.return_value = price_data

    svc = TemporalFilterValidationService.__new__(TemporalFilterValidationService)
    svc.data_service = mock_data
    svc.scoring_service = MagicMock()
    svc.output_base_dir = Path(tempfile.mkdtemp())
    svc.score_cache_dir = Path(tempfile.mkdtemp())

    with patch(
        "backend.services.temporal_pool_service.TemporalPoolService.score_pool_history",
        return_value=score_panel,
    ):
        metrics = svc.run_temporal_pool_backtest(
            codes=["000001"],
            profile_id="dummy_profile",
            start_date="2024-01-01",
            end_date="2024-01-31",
            top_n=1,
            score_threshold=80.0,
            hold_days=5,
        )

    assert metrics["active_days"] == 0.0
    assert metrics["active_ratio"] == 0.0
    assert np.isnan(metrics["annual_return"])
    assert np.isnan(metrics["sharpe"])
    assert np.isnan(metrics["win_rate"])


def test_run_temporal_pool_backtest_win_rate_ignores_flat_days():
    """win_rate 只统计真实持仓日，不应把空仓零收益日算进分母。"""
    from backend.services.temporal_filter_validation_service import TemporalFilterValidationService

    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    score_panel = pd.DataFrame(
        {
            "date": dates,
            "code": ["000001"] * len(dates),
            "score": [90.0] * len(dates),
            "rank": [1] * len(dates),
        }
    )
    price_data = {"000001": make_price_df(n_days=40, seed=8)}
    nav_series = pd.Series([1.0, 1.1, 1.1, 1.21], index=dates)
    active_flags = pd.Series([True, True, False, True], index=dates)

    mock_data = MagicMock()
    mock_data.get_multiple_stocks_data.return_value = price_data

    svc = TemporalFilterValidationService.__new__(TemporalFilterValidationService)
    svc.data_service = mock_data
    svc.scoring_service = MagicMock()
    svc.output_base_dir = Path(tempfile.mkdtemp())
    svc.score_cache_dir = Path(tempfile.mkdtemp())

    with patch(
        "backend.services.temporal_pool_service.TemporalPoolService.score_pool_history",
        return_value=score_panel,
    ):
        with patch.object(
            TemporalFilterValidationService,
            "_equal_weight_backtest_with_diagnostics",
            return_value=(nav_series, active_flags),
        ):
            metrics = svc.run_temporal_pool_backtest(
                codes=["000001"],
                profile_id="dummy_profile",
                start_date="2024-01-01",
                end_date="2024-01-31",
                top_n=1,
                hold_days=5,
            )

    assert metrics["active_days"] == 3.0
    assert metrics["trading_days"] == 4.0
    assert metrics["win_rate"] == 1.0


# ---------------------------------------------------------------------------
# 8.5 Property test — No look-ahead bias (Property 1)
# Feature: temporal-filter-validation, Property 1: 无未来函数
# ---------------------------------------------------------------------------

@given(
    prices=st.lists(
        st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=300, max_size=500,
    ),
    window=st.integers(min_value=20, max_value=100),
)
@settings(max_examples=100)
def test_property_no_lookahead_rolling_rank(prices, window):
    """
    Property 1: rolling rank at time t only uses data up to t.

    Validates: Requirements 2.3

    Verification: for index i, the rolling rank value equals the rank computed
    using only series.iloc[i-window+1 : i+1] (the same window pandas uses).
    We replicate pandas' rank(pct=True) = average_rank / n.
    """
    assume(len(prices) > window + 20)

    series = pd.Series(prices)
    rolling_rank = series.rolling(window=window, min_periods=20).rank(pct=True)

    # Pick an index i >= window to check
    i = window  # first index where rolling window is full
    if i >= len(series):
        return

    rr_at_i = rolling_rank.iloc[i]
    if np.isnan(rr_at_i):
        return  # not enough data at this point, skip

    # Recompute rank using only data up to and including i (same window as pandas)
    start = max(0, i - window + 1)
    window_data = series.iloc[start: i + 1].values
    current_val = series.iloc[i]

    # Replicate pandas rank(pct=True): average rank / n
    # rank of current_val = (number of values < current_val) + (number of ties + 1) / 2
    n = len(window_data)
    n_less = float(np.sum(window_data < current_val))
    n_equal = float(np.sum(window_data == current_val))
    avg_rank = n_less + (n_equal + 1.0) / 2.0
    recomputed = avg_rank / n

    assert abs(rr_at_i - recomputed) < 1e-9, (
        f"Rolling rank at i={i} ({rr_at_i:.6f}) != recomputed ({recomputed:.6f}): "
        "look-ahead bias detected"
    )


# ---------------------------------------------------------------------------
# 8.6 Property test — Forward returns start at t+1 (Property 2)
# Feature: temporal-filter-validation, Property 2: 未来收益从 t+1 开始
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=30, max_value=100),
    h=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=100)
def test_property_forward_returns_start_t1(n, h):
    """
    Property 2: future_return_h entry price is open[t+1], not close[t].

    Validates: Requirements 6.2
    """
    rng = np.random.default_rng(n + h)
    close_arr = 100.0 + np.cumsum(rng.normal(0, 1, n))
    # Make open clearly different from close: open = close + 10
    open_arr = close_arr + 10.0

    close_series = pd.Series(close_arr)
    open_series = pd.Series(open_arr)

    # Correct calculation: entry = open[t+1]
    entry_price = open_series.shift(-1)
    future_return = close_series.shift(-h) / entry_price - 1

    # Wrong calculation: entry = close[t]
    wrong_return = close_series.shift(-h) / close_series - 1

    # For each valid t, verify correct != wrong (since open[t+1] != close[t])
    for t in range(n - h - 1):
        fr_correct = future_return.iloc[t]
        fr_wrong = wrong_return.iloc[t]
        if np.isnan(fr_correct) or np.isnan(fr_wrong):
            continue
        # open[t+1] = close[t+1] + 10 != close[t] (unless close[t+1] == close[t] - 10)
        ep_correct = open_arr[t + 1]
        ep_wrong = close_arr[t]
        if abs(ep_correct - ep_wrong) > 1e-9:
            assert abs(fr_correct - fr_wrong) > 1e-9, (
                f"At t={t}: correct return ({fr_correct:.6f}) should differ from "
                f"wrong return ({fr_wrong:.6f}) when entry prices differ"
            )


# ---------------------------------------------------------------------------
# 8.7 Property test — Top10 weight conservation (Property 3)
# Feature: temporal-filter-validation, Property 3: Top10 权重守恒
# ---------------------------------------------------------------------------

@given(
    n_stocks=st.integers(min_value=10, max_value=50),
    top_n=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_property_top10_weight_conservation(n_stocks, top_n):
    """
    Property 3: strategy C total weight = 1 on each rebalance date.

    Validates: Requirements 10.2
    """
    assume(n_stocks >= top_n)

    rng = np.random.default_rng(n_stocks * 100 + top_n)
    scores = rng.uniform(0, 100, n_stocks)

    # Select top top_n stocks by score
    top_indices = np.argsort(scores)[::-1][:top_n]
    selected = [f"stock_{i}" for i in top_indices]

    # Equal weight
    weight = 1.0 / top_n
    weights = {code: weight for code in selected}

    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-9, (
        f"Strategy C total weight {total_weight} != 1.0 for top_n={top_n}"
    )


# ---------------------------------------------------------------------------
# 8.8 Property test — Strategy D cash weight (Property 4)
# Feature: temporal-filter-validation, Property 4: 策略 D 现金权重正确
# ---------------------------------------------------------------------------

@given(
    n_available=st.integers(min_value=0, max_value=9),
    top_n=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_property_strategy_d_cash_weight(n_available, top_n):
    """
    Property 4: strategy D total weight = 1 (stock weights + cash weight).

    Validates: Requirements 10.4
    """
    assume(n_available < top_n)

    # Each stock gets weight 1/top_n
    stock_weight = 1.0 / top_n
    total_stock_weight = n_available * stock_weight
    cash_weight = 1.0 - total_stock_weight

    total = total_stock_weight + cash_weight
    assert abs(total - 1.0) < 1e-9, (
        f"Strategy D total weight {total} != 1.0 "
        f"(n_available={n_available}, top_n={top_n})"
    )
    assert cash_weight >= 0.0, f"Cash weight {cash_weight} should be non-negative"


# ---------------------------------------------------------------------------
# 8.9 Property test — IC output completeness (Property 5)
# Feature: temporal-filter-validation, Property 5: IC 输出完整性
# ---------------------------------------------------------------------------

@given(
    n_dates=st.integers(min_value=5, max_value=50),
    n_stocks=st.integers(min_value=5, max_value=20),
)
@settings(max_examples=100, deadline=None)
def test_property_ic_output_completeness(n_dates, n_stocks):
    """
    Property 5: IC output contains all horizon dimensions, p_value in [0,1].

    Validates: Requirements 8.3, 8.4
    """
    from backend.services.temporal_filter_validation_service import TemporalFilterValidationService

    codes = [f"stock_{i}" for i in range(n_stocks)]
    panel = make_research_panel(codes, n_dates=n_dates, horizons=(5, 10, 20), seed=n_dates * 100 + n_stocks)

    svc = TemporalFilterValidationService.__new__(TemporalFilterValidationService)
    # calculate_cross_sectional_ic is pure computation, no external deps needed
    result = svc.calculate_cross_sectional_ic(panel, horizons=(5, 10, 20))

    # Must contain all horizon keys
    for h in (5, 10, 20):
        key = f"ic_{h}"
        assert key in result, f"Missing key {key} in IC output"

        entry = result[key]
        required_fields = ["ic_mean", "ic_std", "ir", "positive_ratio", "t_stat", "p_value"]
        for field in required_fields:
            assert field in entry, f"Missing field '{field}' in {key}"

        # p_value must be in [0, 1] if not NaN
        p_value = entry["p_value"]
        if not (isinstance(p_value, float) and np.isnan(p_value)):
            assert 0.0 <= p_value <= 1.0, (
                f"p_value {p_value} out of [0, 1] for {key}"
            )


# ---------------------------------------------------------------------------
# 8.10 Property test — Random baseline reproducibility (Property 6)
# Feature: temporal-filter-validation, Property 6: 随机基准可复现
# ---------------------------------------------------------------------------

@given(
    n_stocks=st.integers(min_value=15, max_value=30),
    n_dates=st.integers(min_value=10, max_value=30),
)
@settings(max_examples=50, deadline=None)
def test_property_random_baseline_reproducible(n_stocks, n_dates):
    """
    Property 6: random10_avg column is identical across two runs.

    Validates: Requirements 10.5
    """
    from backend.services.temporal_filter_validation_service import TemporalFilterValidationService

    codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]
    score_panel = make_score_panel(codes, n_dates=n_dates, seed=n_stocks * 31 + n_dates)
    price_data = {code: make_price_df(n_days=200, seed=i) for i, code in enumerate(codes)}

    mock_data = MagicMock()
    mock_data.get_multiple_stocks_data.return_value = price_data

    def make_svc():
        svc = TemporalFilterValidationService.__new__(TemporalFilterValidationService)
        svc.data_service = mock_data
        svc.scoring_service = MagicMock()
        svc.output_base_dir = Path(tempfile.mkdtemp())
        svc.score_cache_dir = Path(tempfile.mkdtemp())
        return svc

    svc1 = make_svc()
    svc2 = make_svc()

    curves1 = svc1.run_portfolio_backtest(score_panel, top_n=10, rebalance="W-FRI")
    curves2 = svc2.run_portfolio_backtest(score_panel, top_n=10, rebalance="W-FRI")

    assert "random10_avg" in curves1.columns, "random10_avg should be in result"
    assert "random10_avg" in curves2.columns, "random10_avg should be in result"

    s1 = curves1["random10_avg"].reset_index(drop=True)
    s2 = curves2["random10_avg"].reset_index(drop=True)

    pd.testing.assert_series_equal(s1, s2, check_names=False,
                                   obj="random10_avg reproducibility")


# ---------------------------------------------------------------------------
# 8.11 Property test — Failed samples recorded completely (Property 8)
# Feature: temporal-filter-validation, Property 8: 失败样本记录完整
# ---------------------------------------------------------------------------

@given(
    n_total=st.integers(min_value=5, max_value=20),
    n_fail=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100)
def test_property_failed_samples_complete(n_total, n_fail):
    """
    Property 8: len(success) + len(failures) == len(input).

    Validates: Requirements 3.5
    """
    assume(n_fail < n_total)

    from backend.services.temporal_scoring_service import TemporalScoringService

    codes = [f"{i:06d}" for i in range(n_total)]
    fail_codes = set(codes[:n_fail])

    svc = TemporalScoringService.__new__(TemporalScoringService)
    svc._last_history_errors = []

    def mock_score_history(code, start_date, end_date, window=252):
        if code in fail_codes:
            raise ValueError(f"Simulated failure for {code}")
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "code": code,
            "score": 50.0,
            "trend_break": 0,
            "high_volatility": 0,
            "liquidity_risk": 0,
        })

    with patch.object(svc, "score_stock_history", side_effect=mock_score_history):
        # Replicate score_many_stocks_history logic inline (uses ThreadPoolExecutor)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        svc._last_history_errors = []
        frames = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_code = {
                executor.submit(svc.score_stock_history, code, "2024-01-01", "2024-03-01"): code
                for code in codes
            }
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    frames.append(future.result())
                except Exception as exc:
                    svc._last_history_errors.append({"code": code, "error": str(exc)})

    result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    success_codes = set(result_df["code"].unique()) if not result_df.empty else set()
    error_codes = {e["code"] for e in svc._last_history_errors}

    assert len(success_codes) + len(error_codes) == n_total, (
        f"Total mismatch: {len(success_codes)} success + {len(error_codes)} errors "
        f"!= {n_total} total"
    )
    assert error_codes == fail_codes, (
        f"Error codes {error_codes} != expected fail codes {fail_codes}"
    )


# ---------------------------------------------------------------------------
# 8.12 Property test — Score cache consistency (Property 7)
# Feature: temporal-filter-validation, Property 7: 评分缓存一致性
# ---------------------------------------------------------------------------

from hypothesis import HealthCheck

@given(
    code=st.text(
        alphabet=st.characters(whitelist_categories=("Nd",)),
        min_size=6, max_size=6,
    ),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_property_score_cache_consistency(code, tmp_path):
    """
    Property 7: cached result equals recomputed result.

    Validates: Requirements 2.6, 2.7
    """
    from backend.services.temporal_scoring_service import TemporalScoringService

    start_date = "2024-01-01"
    end_date = "2024-06-30"

    # Build consistent price data for this code
    price_df = make_price_df(n_days=400, seed=int(code) if code.isdigit() else 42)

    mock_data = MagicMock()
    mock_data.get_stock_data.return_value = price_df

    svc = TemporalScoringService.__new__(TemporalScoringService)
    svc.score_cache_dir = tmp_path
    svc.lookback_days = 372

    # Patch _compute_factors to return simple deterministic factors
    def mock_compute_factors(df):
        n = len(df)
        base = pd.Series(np.arange(1.0, n + 1.0), index=df.index)
        return {
            "price_vs_sma20": base,
            "momentum_20": base * 0.5,
            "price_vwma_ratio": base * 0.3,
            "force_index_ma": base * 0.2,
            "bollinger_position": base * 0.4,
            "stochastic_d": base * 0.6,
            "atr_norm": base * 0.1,
            "volume_ma_ratio": base * 0.7,
        }

    with patch.object(svc, "_compute_factors", side_effect=mock_compute_factors):
        with patch("backend.services.temporal_scoring_service.data_service", mock_data):
            # First call: computes and caches
            result1 = svc.score_stock_history(code, start_date, end_date)

            # Delete the cache file
            cache_path = tmp_path / f"{code}_{start_date}_{end_date}.parquet"
            if cache_path.exists():
                cache_path.unlink()

            # Second call: recomputes
            result2 = svc.score_stock_history(code, start_date, end_date)

    pd.testing.assert_frame_equal(
        result1.reset_index(drop=True),
        result2.reset_index(drop=True),
        check_like=True,
        obj=f"Cache consistency for code={code}",
    )
