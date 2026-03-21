from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api.main import app


client = TestClient(app)


def test_recent_valuable_mining_endpoint_returns_service_result():
    fake_result = {
        "metadata": {"stock_count_with_result": 1},
        "top_factors": [
            {
                "factor_name": "price_vs_sma20",
                "score": 95.0,
                "recommended_direction": 1,
                "selected": True,
            }
        ],
        "all_factor_scores": [],
        "per_stock_top_factors": {
            "000001.SZ": {
                "current_regime": "uptrend",
                "factor_count": 1,
                "top_factors": [],
            }
        },
    }

    with patch(
        "backend.services.recent_factor_mining_service.recent_factor_mining_service.mine_recent_valuable_factors",
        return_value=fake_result,
    ):
        response = client.post(
            "/api/mining/recent-valuable",
            json={
                "stock_codes": ["000001.SZ"],
                "end_date": "2025-03-31",
                "horizons": [1, 5],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["metadata"]["stock_count_with_result"] == 1
    assert body["data"]["top_factors"][0]["factor_name"] == "price_vs_sma20"

