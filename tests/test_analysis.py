import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backend.api.routers.analysis import CalculateRequest, calculate_factor
from backend.api.routers.factors import PreselectRequest, preselect_factors
from backend.services.analysis_service import AnalysisService
from backend.services.factor_neutralization_service import FactorNeutralizationService
from backend.services.position_analysis_service import PositionAnalysisService
from backend.services.portfolio_analysis_service import PortfolioAnalysisService


def test_analysis_cache_key_changes_with_window_and_signature():
    service = AnalysisService()

    key_a = service._generate_cache_key(
        stock_codes=["000001.SZ", "600000.SH"],
        factor_names=["alpha"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        rolling_window=252,
        factor_signature="alpha:hash-a",
    )
    key_b = service._generate_cache_key(
        stock_codes=["000001.SZ", "600000.SH"],
        factor_names=["alpha"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        rolling_window=126,
        factor_signature="alpha:hash-a",
    )
    key_c = service._generate_cache_key(
        stock_codes=["000001.SZ", "600000.SH"],
        factor_names=["alpha"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        rolling_window=252,
        factor_signature="alpha:hash-b",
    )

    assert key_a != key_b
    assert key_a != key_c


def test_portfolio_exposure_aggregates_weights_and_aligns_factor_values():
    service = PortfolioAnalysisService()
    positions = pd.DataFrame(
        {
            "stock_code": ["A", "B", "C"],
            "industry": ["Tech", "Tech", "Finance"],
            "weight": [0.2, 0.3, 0.5],
        }
    )

    industry = service.calculate_industry_exposure(positions)
    factor = service.calculate_factor_exposure(
        positions,
        {"value": pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})},
    )

    assert industry["industry_exposure"]["Tech"] == 0.5
    assert industry["industry_exposure"]["Finance"] == 0.5
    assert factor["factor_exposures"]["value"] == 2.3


def test_preselect_filters_candidates(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    close = pd.Series(np.linspace(10, 20, 60), index=dates)
    stock_df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1000, 2000, 60),
        },
        index=dates,
    )

    from backend.services import data_service as data_service_module
    from backend.services import factor_service as factor_service_module
    from backend.services import factor_validation_service as factor_validation_module
    from backend.repositories import factor_repository as factor_repository_module
    from backend.core import database as database_module

    monkeypatch.setattr(
        data_service_module.data_service,
        "get_multiple_stocks_data",
        lambda **kwargs: {"000001.SZ": stock_df},
    )

    def fake_calculate(df, factor_code, formula_type=None):
        if factor_code == "good":
            return df["close"].pct_change().shift(-1).fillna(0.0)
        if factor_code == "bad":
            return pd.Series(0.0, index=df.index)
        raise ValueError(f"unexpected factor: {factor_code}")

    monkeypatch.setattr(
        factor_service_module.factor_service.calculator,
        "calculate",
        fake_calculate,
    )

    class FakeValidator:
        def __init__(self, ic_threshold, ir_threshold):
            self.ic_threshold = ic_threshold
            self.ir_threshold = ir_threshold

        def validate_factor(self, factor_values, return_values):
            passed = factor_values.mean() > 0
            return {
                "ic_validation": {"ic": 0.2 if passed else 0.0},
                "ir_validation": {"ir": 0.3 if passed else 0.0},
            }

    monkeypatch.setattr(
        factor_validation_module,
        "FactorValidationService",
        FakeValidator,
    )

    class FakeDb:
        def close(self):
            return None

    class FakeRepo:
        def __init__(self, db):
            self.db = db

        def get_by_name(self, name):
            return None

    monkeypatch.setattr(database_module, "get_db_session", lambda: FakeDb())
    monkeypatch.setattr(factor_repository_module, "FactorRepository", FakeRepo)

    result = asyncio.run(
        preselect_factors(
            PreselectRequest(
                factors=["good", "bad"],
                ic_threshold=0.1,
                ir_threshold=0.1,
                min_valid_ratio=0.5,
            )
        )
    )

    assert result["success"] is True
    assert result["data"]["factors"] == ["good"]
    details = {item["factor"]: item for item in result["data"]["details"]}
    assert details["good"]["passed"] is True
    assert details["bad"]["passed"] is False


def test_factor_neutralization_uses_real_industry_loader(monkeypatch):
    service = FactorNeutralizationService()

    monkeypatch.setattr(
        service,
        "_load_industry_map_from_spot",
        lambda pending_codes: None,
    )
    monkeypatch.setattr(
        service,
        "_load_industry_from_individual_info",
        lambda stock_code: {"000001": "银行", "600519": "白酒"}[stock_code],
    )

    mapping = service.get_industry_classification(["000001.SZ", "600519.SH"])

    assert mapping == {"000001.SZ": "银行", "600519.SH": "白酒"}


def test_position_analysis_uses_real_turnover_formula():
    service = PositionAnalysisService()
    positions = pd.Series([0.0, 1.0, 0.0], index=pd.date_range("2024-01-01", periods=3, freq="D"))

    result = service.analyze_positions(positions)

    assert result["turnover"] == 1.0
    assert result["position_changes"]["max_position_change"] == 0.5


def test_ir_weight_prefers_less_negative_factor():
    service = PortfolioAnalysisService()
    factor_returns = pd.DataFrame(
        {
            "better": [-0.01, -0.02, -0.01, -0.015],
            "worse": [-0.03, -0.04, -0.025, -0.035],
        }
    )

    result = service.optimize_weights(factor_returns, method="ir_weight")

    assert result["weights"]["better"] > result["weights"]["worse"]


def test_analysis_calculate_returns_plot_series(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    stock_df = pd.DataFrame(
        {
            "open": [10.0, 10.5, 11.0, 11.5],
            "high": [10.2, 10.7, 11.2, 11.7],
            "low": [9.8, 10.3, 10.8, 11.3],
            "close": [10.1, 10.6, 11.1, 11.6],
            "volume": [1000, 1100, 1200, 1300],
            "amount": [10100, 11660, 13320, 15080],
        },
        index=dates,
    )

    from backend.services import data_service as data_service_module
    from backend.services import factor_service as factor_service_module
    from backend.repositories import factor_repository as factor_repository_module
    from backend.core import database as database_module

    monkeypatch.setattr(
        data_service_module.data_service,
        "get_stock_data",
        lambda stock_code, start_date, end_date: stock_df.copy(),
    )

    class FakeDb:
        def close(self):
            return None

    class FakeRepo:
        def __init__(self, db):
            self.db = db

        def get_by_name(self, name):
            return SimpleNamespace(
                name=name,
                code="DIFF: CLOSE - OPEN; DEA: MA(CLOSE, 2); XG: CLOSE / OPEN",
                formula_type="mylanguage",
            )

    monkeypatch.setattr(database_module, "get_db_session", lambda: FakeDb())
    monkeypatch.setattr(factor_repository_module, "FactorRepository", FakeRepo)

    response = asyncio.run(
        calculate_factor(
            CalculateRequest(
                factor_name="demo_factor",
                stock_codes=["000001.SZ"],
                start_date="2024-01-01",
                end_date="2024-01-04",
            )
        )
    )

    payload = response["data"]["000001.SZ"]
    assert payload["dates"] == ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    assert np.allclose(
        payload["factor_values"],
        [1.01, 1.0095238095238095, 1.009090909090909, 1.008695652173913],
    )
    assert payload["formula_type"] == "mylanguage"
    assert [item["name"] for item in payload["plot_series"]] == ["DIFF", "DEA", "XG"]
    assert np.allclose(payload["plot_series"][0]["values"], [0.1, 0.1, 0.1, 0.1])
    assert np.allclose(payload["plot_series"][2]["values"], payload["factor_values"])
