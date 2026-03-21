import numpy as np
import pandas as pd
import pytest

from backend.services.recent_factor_mining_service import RecentFactorMiningService


def _make_panel(seed: int, n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    daily_ret = 0.001 + rng.normal(0, 0.01, n)
    close = 100 * np.cumprod(1 + daily_ret)
    close_series = pd.Series(close, index=dates)

    open_ = close_series * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close_series) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close_series) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    fwd_ret_1 = close_series.shift(-1) / close_series - 1
    alpha = fwd_ret_1 + rng.normal(0, 0.001, n)
    alpha_copy = alpha * 1.01 + rng.normal(0, 0.0002, n)
    inverse = -fwd_ret_1 + rng.normal(0, 0.001, n)
    noise = rng.normal(0, 1, n)

    return pd.DataFrame(
        {
            "open": open_.values,
            "high": high.values,
            "low": low.values,
            "close": close_series.values,
            "volume": volume,
            "f_alpha": alpha.values,
            "f_alpha_copy": alpha_copy.values,
            "f_inverse": inverse.values,
            "f_noise": noise,
        },
        index=dates,
    )


def test_recent_factor_mining_identifies_alpha_and_prunes_redundancy(monkeypatch):
    service = RecentFactorMiningService()
    panel_a = _make_panel(seed=1)
    panel_b = _make_panel(seed=2)

    monkeypatch.setattr(
        service,
        "_load_active_factor_definitions",
        lambda: {
            "f_alpha": "unused",
            "f_alpha_copy": "unused",
            "f_inverse": "unused",
            "f_noise": "unused",
        },
    )
    monkeypatch.setattr(
        service,
        "_build_factor_panels",
        lambda **_: {
            "000001.SZ": panel_a,
            "600519.SH": panel_b,
        },
    )

    result = service.mine_recent_valuable_factors(
        stock_codes=["000001.SZ", "600519.SH"],
        end_date="2025-03-31",
        eval_days=120,
        lookback_days=240,
        horizons=[1],
        top_k=3,
        min_samples=20,
        correlation_threshold=0.8,
    )

    assert result["metadata"]["stock_count_with_result"] == 2
    assert result["top_factors"][0]["factor_name"] in {"f_alpha", "f_alpha_copy", "f_inverse"}

    pair_items = {
        item["factor_name"]: item
        for item in result["all_factor_scores"]
        if item["factor_name"] in {"f_alpha", "f_alpha_copy"}
    }
    assert len(pair_items) == 2
    assert any(
        (not item["selected"]) and (item["redundant_with"] is not None)
        for item in pair_items.values()
    )

    assert "000001.SZ" in result["per_stock_top_factors"]
    assert result["per_stock_top_factors"]["000001.SZ"]["top_factors"]


def test_recent_factor_mining_raises_when_no_valid_signal(monkeypatch):
    service = RecentFactorMiningService()
    panel = _make_panel(seed=3)
    panel["f_nan"] = np.nan

    monkeypatch.setattr(
        service,
        "_load_active_factor_definitions",
        lambda: {"f_nan": "unused"},
    )
    monkeypatch.setattr(
        service,
        "_build_factor_panels",
        lambda **_: {"000001.SZ": panel},
    )

    with pytest.raises(ValueError, match="未得到有效因子评估结果"):
        service.mine_recent_valuable_factors(
            stock_codes=["000001.SZ"],
            end_date="2025-03-31",
            eval_days=60,
            lookback_days=120,
            horizons=[1],
            top_k=5,
            min_samples=20,
            correlation_threshold=0.8,
        )

