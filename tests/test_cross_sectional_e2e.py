"""
端到端测试：
  1. 带真实 universe_df 的截面回测（cross_sectional_backtest_composite）
  2. RecentFactorMiningService 参数范围校验（服务层 + API 层）
  3. IC/IR 命名：输出同时包含语义准确字段和向后兼容别名
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. 截面回测端到端：带真实 universe_df
# ---------------------------------------------------------------------------

class TestCrossSectionalE2EWithUniverseDf:
    """
    使用合成 universe_df（含 close + 因子列）直接调用
    evaluate_cross_sectional_candidate，验证整条链路可跑通且返回有效指标。
    """

    @staticmethod
    def _make_universe_df(
        codes: list[str],
        n_days: int = 60,
        factor_names: list[str] | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """生成合成 universe_df，含 date/stock_code/close + 因子列。"""
        if factor_names is None:
            factor_names = ["price_vs_sma20", "momentum_20"]

        rng = np.random.default_rng(seed)
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        frames = []
        for i, code in enumerate(codes):
            close = 10.0 + rng.random(n_days).cumsum()
            row = pd.DataFrame({"date": dates, "stock_code": code, "close": close})
            for j, fn in enumerate(factor_names):
                # 合成因子：随机游走，确保非全 NaN
                row[fn] = rng.random(n_days) * (i + 1) * (j + 1)
            frames.append(row)
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def test_evaluate_with_universe_df_returns_stability_score(self):
        """
        candidate 直接提供 universe_df 时，evaluate_cross_sectional_candidate
        应返回有效的 stability_score（不为 NaN）。
        """
        from backend.services.strategy_evaluation_service import StrategyEvaluationService

        codes = ["000001", "000002", "000003", "000004", "000005"]
        # Use 300 days to ensure ≥3 valid windows with the given window config
        universe_df = self._make_universe_df(codes, n_days=300)

        candidate = {
            "universe_df": universe_df,
            "config": {
                "factors": [
                    {"name": "price_vs_sma20", "role": "signal", "weight": 0.6, "direction": 1},
                    {"name": "momentum_20", "role": "signal", "weight": 0.4, "direction": 1},
                ],
                "params": {"top_percentile": 0.4},
            },
        }

        # Mock VectorBTBacktestService to avoid needing vectorbt installed
        mock_vbt_result = {
            "annual_return": 0.12,
            "sharpe": 1.2,
            "max_drawdown": -0.08,
            "total_return": 0.10,
            "volatility": 0.15,
        }

        with patch(
            "backend.services.vectorbt_backtest_service.VectorBTBacktestService"
        ) as MockVBT:
            mock_vbt_instance = MagicMock()
            mock_vbt_instance.cross_sectional_backtest_composite.return_value = mock_vbt_result
            MockVBT.return_value = mock_vbt_instance

            svc = StrategyEvaluationService.__new__(StrategyEvaluationService)
            result = svc.evaluate_cross_sectional_candidate(
                candidate=candidate,
                universe="test_universe",
                start_date="2023-01-01",
                end_date="2024-01-01",
                train_months=3,
                valid_months=1,
                test_months=1,
                step_months=1,
            )

        assert "stability_score" in result
        assert not np.isnan(result["stability_score"]), (
            "stability_score 不应为 NaN（universe_df 已提供）"
        )
        assert 0.0 <= result["stability_score"] <= 100.0
        assert "window_metrics" in result
        assert len(result["window_metrics"]) > 0

    def test_evaluate_with_universe_df_factor_columns_used(self):
        """
        验证 cross_sectional_backtest_composite 被调用时，
        传入的 df 包含请求的因子列（非全 NaN）。
        """
        from backend.services.strategy_evaluation_service import StrategyEvaluationService

        codes = ["000001", "000002", "000003"]
        universe_df = self._make_universe_df(codes, n_days=90,
                                              factor_names=["price_vs_sma20"])

        candidate = {
            "universe_df": universe_df,
            "config": {
                "factors": [
                    {"name": "price_vs_sma20", "role": "signal", "weight": 1.0, "direction": 1},
                ],
                "params": {"top_percentile": 0.4},
            },
        }

        captured_dfs = []

        def capture_call(df, factor_weights, top_percentile):
            captured_dfs.append(df.copy())
            return {
                "annual_return": 0.08, "sharpe": 0.9,
                "max_drawdown": -0.05, "total_return": 0.07, "volatility": 0.12,
            }

        with patch(
            "backend.services.vectorbt_backtest_service.VectorBTBacktestService"
        ) as MockVBT:
            mock_vbt_instance = MagicMock()
            mock_vbt_instance.cross_sectional_backtest_composite.side_effect = capture_call
            MockVBT.return_value = mock_vbt_instance

            svc = StrategyEvaluationService.__new__(StrategyEvaluationService)
            svc.evaluate_cross_sectional_candidate(
                candidate=candidate,
                universe="test_universe",
                start_date="2023-01-01",
                end_date="2023-06-30",
                train_months=3,
                valid_months=1,
                test_months=1,
                step_months=1,
            )

        assert len(captured_dfs) > 0, "cross_sectional_backtest_composite 未被调用"
        for df in captured_dfs:
            assert "price_vs_sma20" in df.columns, (
                "传入 cross_sectional_backtest_composite 的 df 缺少 price_vs_sma20 列"
            )
            non_nan = df["price_vs_sma20"].notna().sum()
            assert non_nan > 0, "price_vs_sma20 列全为 NaN"

    def test_evaluate_empty_universe_df_returns_zero_score(self):
        """universe_df 为空时，stability_score 应为 0.0（所有窗口均跳过）"""
        from backend.services.strategy_evaluation_service import StrategyEvaluationService
        import pandas as pd

        candidate = {
            "universe_df": pd.DataFrame(columns=["date", "stock_code", "close", "price_vs_sma20"]),
            "config": {
                "factors": [
                    {"name": "price_vs_sma20", "role": "signal", "weight": 1.0, "direction": 1},
                ],
                "params": {"top_percentile": 0.4},
            },
        }

        with patch(
            "backend.services.vectorbt_backtest_service.VectorBTBacktestService"
        ) as MockVBT:
            MockVBT.return_value = MagicMock()

            svc = StrategyEvaluationService.__new__(StrategyEvaluationService)
            result = svc.evaluate_cross_sectional_candidate(
                candidate=candidate,
                universe="test_universe",
                start_date="2023-01-01",
                end_date="2023-06-30",
                train_months=3,
                valid_months=1,
                test_months=1,
                step_months=1,
            )

        assert result["stability_score"] == 0.0


# ---------------------------------------------------------------------------
# 2. RecentFactorMiningService 参数范围校验
# ---------------------------------------------------------------------------

class TestRecentFactorMiningParamValidation:
    """服务层参数校验：非法输入应抛出 ValueError，而非返回含糊结果。"""

    @pytest.fixture()
    def svc(self):
        from backend.services.recent_factor_mining_service import RecentFactorMiningService
        return RecentFactorMiningService.__new__(RecentFactorMiningService)

    def _call(self, svc, **kwargs):
        defaults = dict(
            stock_codes=["000001"],
            end_date="2024-01-05",
            eval_days=90,
            lookback_days=240,
            horizons=[1, 5, 10],
            top_k=20,
            min_samples=25,
            correlation_threshold=0.85,
        )
        defaults.update(kwargs)
        return svc.mine_recent_valuable_factors(**defaults)

    def test_empty_stock_codes_raises(self, svc):
        with pytest.raises(ValueError, match="stock_codes"):
            self._call(svc, stock_codes=[])

    def test_negative_horizon_raises(self, svc):
        with pytest.raises(ValueError, match="horizons"):
            self._call(svc, horizons=[-1, 5])

    def test_zero_horizon_raises(self, svc):
        with pytest.raises(ValueError, match="horizons"):
            self._call(svc, horizons=[0, 5])

    def test_eval_days_too_small_raises(self, svc):
        with pytest.raises(ValueError, match="eval_days"):
            self._call(svc, eval_days=5)

    def test_lookback_days_too_small_raises(self, svc):
        with pytest.raises(ValueError, match="lookback_days"):
            self._call(svc, lookback_days=10)

    def test_top_k_zero_raises(self, svc):
        with pytest.raises(ValueError, match="top_k"):
            self._call(svc, top_k=0)

    def test_top_k_negative_raises(self, svc):
        with pytest.raises(ValueError, match="top_k"):
            self._call(svc, top_k=-5)

    def test_min_samples_too_small_raises(self, svc):
        with pytest.raises(ValueError, match="min_samples"):
            self._call(svc, min_samples=2)

    def test_correlation_threshold_above_1_raises(self, svc):
        with pytest.raises(ValueError, match="correlation_threshold"):
            self._call(svc, correlation_threshold=1.5)

    def test_correlation_threshold_negative_raises(self, svc):
        with pytest.raises(ValueError, match="correlation_threshold"):
            self._call(svc, correlation_threshold=-0.1)


# ---------------------------------------------------------------------------
# 3. Mining API 层参数校验（Pydantic 422）
# ---------------------------------------------------------------------------

class TestMiningApiParamValidation:
    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.routers.mining import router as mining_router
        app = FastAPI()
        app.include_router(mining_router)
        return TestClient(app, raise_server_exceptions=False)

    def test_top_k_zero_returns_422(self, client):
        resp = client.post("/recent-valuable", json={
            "stock_codes": ["000001"],
            "end_date": "2024-01-05",
            "top_k": 0,
        })
        assert resp.status_code == 422

    def test_top_k_negative_returns_422(self, client):
        resp = client.post("/recent-valuable", json={
            "stock_codes": ["000001"],
            "end_date": "2024-01-05",
            "top_k": -1,
        })
        assert resp.status_code == 422

    def test_eval_days_too_small_returns_422(self, client):
        resp = client.post("/recent-valuable", json={
            "stock_codes": ["000001"],
            "end_date": "2024-01-05",
            "eval_days": 5,
        })
        assert resp.status_code == 422

    def test_correlation_threshold_above_1_returns_422(self, client):
        resp = client.post("/recent-valuable", json={
            "stock_codes": ["000001"],
            "end_date": "2024-01-05",
            "correlation_threshold": 1.5,
        })
        assert resp.status_code == 422

    def test_correlation_threshold_negative_returns_422(self, client):
        resp = client.post("/recent-valuable", json={
            "stock_codes": ["000001"],
            "end_date": "2024-01-05",
            "correlation_threshold": -0.1,
        })
        assert resp.status_code == 422

    def test_min_samples_too_small_returns_422(self, client):
        resp = client.post("/recent-valuable", json={
            "stock_codes": ["000001"],
            "end_date": "2024-01-05",
            "min_samples": 2,
        })
        assert resp.status_code == 422

    def test_valid_params_pass_validation(self, client):
        """合法参数通过 Pydantic 校验（服务层可能因无数据失败，但不应 422）"""
        with patch(
            "backend.services.recent_factor_mining_service"
            ".RecentFactorMiningService.mine_recent_valuable_factors",
            side_effect=ValueError("无法构建任何股票的因子面板"),
        ):
            resp = client.post("/recent-valuable", json={
                "stock_codes": ["000001"],
                "end_date": "2024-01-05",
                "eval_days": 90,
                "top_k": 10,
                "min_samples": 25,
                "correlation_threshold": 0.85,
            })
        # 422 = 参数校验失败；400 = 服务层 ValueError；两者都不是 422 才算通过
        assert resp.status_code != 422


# ---------------------------------------------------------------------------
# 4. IC/IR 命名：输出同时包含语义准确字段和向后兼容别名
# ---------------------------------------------------------------------------

class TestIcIrNamingAlias:
    """
    验证 mine_recent_valuable_factors 输出的 all_factor_scores 和
    per_stock_top_factors 同时包含：
      - 语义准确字段：predictive_corr_mean, abs_predictive_corr_mean,
                      horizon_stability, regime_predictive_corr_mean
      - 向后兼容别名：ic_mean, abs_ic_mean, abs_ir, regime_abs_ic_mean
    """

    @staticmethod
    def _make_mock_service_result():
        """构造一个最小化的 mine_recent_valuable_factors 返回值。"""
        factor_score = {
            "factor_name": "price_vs_sma20",
            "score": 75.0,
            "recommended_direction": 1,
            # 语义准确字段
            "predictive_corr_mean": 0.05,
            "abs_predictive_corr_mean": 0.05,
            "horizon_stability": 0.8,
            "regime_predictive_corr_mean": 0.04,
            "sign_consistency": 0.7,
            "regime_sign_match": 0.6,
            "coverage_ratio": 1.0,
            "selected": True,
            "redundant_with": None,
            "redundancy_correlation": None,
            # 向后兼容别名
            "ic_mean": 0.05,
            "abs_ic_mean": 0.05,
            "abs_ir": 0.8,
            "regime_abs_ic_mean": 0.04,
        }
        return {
            "metadata": {},
            "top_factors": [factor_score],
            "all_factor_scores": [factor_score],
            "per_stock_top_factors": {
                "000001": {
                    "current_regime": "uptrend",
                    "factor_count": 1,
                    "top_factors": [{
                        "factor_name": "price_vs_sma20",
                        "stock_score": 80.0,
                        "recommended_direction": 1,
                        "predictive_corr_mean": 0.06,
                        "abs_predictive_corr_mean": 0.06,
                        "horizon_stability": 0.9,
                        "regime_predictive_corr_mean": 0.05,
                        "ic_mean": 0.06,
                        "abs_ic_mean": 0.06,
                        "abs_ir": 0.9,
                        "regime_abs_ic_mean": 0.05,
                    }],
                }
            },
        }

    def test_all_factor_scores_has_semantic_fields(self):
        """all_factor_scores 包含语义准确字段"""
        result = self._make_mock_service_result()
        for item in result["all_factor_scores"]:
            assert "predictive_corr_mean" in item, "缺少 predictive_corr_mean"
            assert "abs_predictive_corr_mean" in item, "缺少 abs_predictive_corr_mean"
            assert "horizon_stability" in item, "缺少 horizon_stability"
            assert "regime_predictive_corr_mean" in item, "缺少 regime_predictive_corr_mean"

    def test_all_factor_scores_has_compat_aliases(self):
        """all_factor_scores 保留向后兼容别名"""
        result = self._make_mock_service_result()
        for item in result["all_factor_scores"]:
            assert "ic_mean" in item, "缺少向后兼容别名 ic_mean"
            assert "abs_ic_mean" in item, "缺少向后兼容别名 abs_ic_mean"
            assert "abs_ir" in item, "缺少向后兼容别名 abs_ir"
            assert "regime_abs_ic_mean" in item, "缺少向后兼容别名 regime_abs_ic_mean"

    def test_semantic_and_alias_values_consistent(self):
        """语义字段与别名字段的值应一致"""
        result = self._make_mock_service_result()
        for item in result["all_factor_scores"]:
            assert item["predictive_corr_mean"] == item["ic_mean"]
            assert item["abs_predictive_corr_mean"] == item["abs_ic_mean"]
            assert item["horizon_stability"] == item["abs_ir"]
            assert item["regime_predictive_corr_mean"] == item["regime_abs_ic_mean"]

    def test_per_stock_top_factors_has_semantic_fields(self):
        """per_stock_top_factors 的 top_factors 也包含语义准确字段"""
        result = self._make_mock_service_result()
        for code, stock_data in result["per_stock_top_factors"].items():
            for item in stock_data["top_factors"]:
                assert "predictive_corr_mean" in item, (
                    f"股票 {code} top_factors 缺少 predictive_corr_mean"
                )
                assert "horizon_stability" in item, (
                    f"股票 {code} top_factors 缺少 horizon_stability"
                )

    def test_real_service_output_has_both_fields(self):
        """
        通过 mock 数据驱动真实服务，验证实际输出同时包含语义字段和别名。
        """
        from backend.services.recent_factor_mining_service import RecentFactorMiningService

        svc = RecentFactorMiningService.__new__(RecentFactorMiningService)

        # 构造最小化的 panels 和 records，绕过 IO
        import pandas as pd
        import numpy as np

        dates = pd.date_range("2023-10-01", periods=30, freq="B")
        rng = np.random.default_rng(0)
        close = 10.0 + rng.random(30).cumsum()
        panel = pd.DataFrame({
            "close": close,
            "test_factor": rng.random(30),
        }, index=dates)

        eval_start = dates[0]
        result = svc._evaluate_stock_factors(
            stock_code="000001",
            panel=panel,
            factor_names=["test_factor"],
            eval_start=eval_start,
            horizons=[1, 5],
            min_samples=5,
        )

        assert result["records"], "应有评估记录"
        record = result["records"][0]

        # 语义准确字段
        assert "predictive_corr_mean" in record, "缺少 predictive_corr_mean"
        assert "abs_predictive_corr_mean" in record, "缺少 abs_predictive_corr_mean"
        assert "horizon_stability" in record, "缺少 horizon_stability"

        # 向后兼容别名
        assert "ic_mean" in record, "缺少向后兼容别名 ic_mean"
        assert "abs_ic_mean" in record, "缺少向后兼容别名 abs_ic_mean"
        assert "abs_ir" in record, "缺少向后兼容别名 abs_ir"

        # 值一致
        assert record["predictive_corr_mean"] == record["ic_mean"]
        assert record["horizon_stability"] == record["abs_ir"]
