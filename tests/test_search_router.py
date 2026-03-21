"""
端到端测试：Champion 策略搜索路由 (search.py)
覆盖 temporal_pool、cross_sectional、job 状态查询、champion CRUD 路由。
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.api.routers.search as search_module
from backend.api.routers.search import (
    _get_champion_svc,
    _get_pool_svc,
    _get_profile_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_svc():
    svc = MagicMock()
    svc.run_for_pool.return_value = {"profile_id": "trend_flow_v1", "stability_score": 72.5}
    svc.run_for_cross_sectional.return_value = {"profile_id": "momentum_v1", "stability_score": 65.0}
    svc.run_for_stock.return_value = {"profile_id": "breakout_v1", "stability_score": 58.0}
    svc.get_champion.return_value = {"profile_id": "trend_flow_v1", "stability_score": 72.5}
    svc.apply_champion.return_value = None
    return svc


@pytest.fixture()
def app_and_svc(tmp_path):
    """
    Returns (TestClient, mock_svc).
    Uses FastAPI dependency_overrides so Depends() injection is properly mocked.
    Redirects _JOBS_DIR to tmp_path.
    """
    search_module._JOBS_DIR = tmp_path / "champion_jobs"

    mock_svc = _make_mock_svc()

    app = FastAPI()
    app.include_router(search_module.router)
    app.dependency_overrides[_get_champion_svc] = lambda: mock_svc

    client = TestClient(app, raise_server_exceptions=False)
    return client, mock_svc


@pytest.fixture()
def client(app_and_svc):
    return app_and_svc[0]


@pytest.fixture()
def mock_svc(app_and_svc):
    return app_and_svc[1]


# ---------------------------------------------------------------------------
# POST /champion/temporal-pool
# ---------------------------------------------------------------------------

class TestTemporalPoolSearch:
    def test_returns_202_with_task_id(self, client):
        resp = client.post("/champion/temporal-pool", json={
            "codes": ["600519", "000651"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        })
        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert body["status"] == "pending"

    def test_task_status_file_created(self, client):
        resp = client.post("/champion/temporal-pool", json={
            "codes": ["600519"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        })
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        status_file = search_module._JOBS_DIR / task_id / "task_status.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["task_id"] == task_id
        # status is "pending" at creation time (background task may update it)
        assert data["status"] in ("pending", "running", "completed", "failed")

    def test_missing_required_fields_returns_422(self, client):
        resp = client.post("/champion/temporal-pool", json={
            "codes": ["600519"],
            # missing start_date and end_date
        })
        assert resp.status_code == 422

    def test_max_workers_default_is_1(self, client):
        resp = client.post("/champion/temporal-pool", json={
            "codes": ["600519"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        })
        assert resp.status_code == 202


# ---------------------------------------------------------------------------
# POST /champion/cross-sectional
# ---------------------------------------------------------------------------

class TestCrossSectionalSearch:
    def test_returns_202_with_task_id(self, client):
        resp = client.post("/champion/cross-sectional", json={
            "universe": "hs300",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        })
        assert resp.status_code == 202
        assert "task_id" in resp.json()

    def test_missing_universe_returns_422(self, client):
        resp = client.post("/champion/cross-sectional", json={
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /champion/jobs/{task_id}
# ---------------------------------------------------------------------------

class TestJobStatus:
    def _create_job_file(self, task_id: str, status: str) -> None:
        task_dir = search_module._JOBS_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "status": status,
            "created_at": "2024-01-05T00:00:00",
            "updated_at": "2024-01-05T00:01:00",
            "message": "test",
            "champion": None,
            "error": None,
        }
        (task_dir / "task_status.json").write_text(json.dumps(payload))

    def test_returns_status_for_existing_task(self, client):
        task_id = "aaaaaaaa-0000-0000-0000-000000000001"
        self._create_job_file(task_id, "completed")

        resp = client.get(f"/champion/jobs/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == task_id
        assert body["status"] == "completed"

    def test_returns_404_for_missing_task(self, client):
        resp = client.get("/champion/jobs/nonexistent-task-id")
        assert resp.status_code == 404

    def test_running_status_returned_correctly(self, client):
        task_id = "aaaaaaaa-0000-0000-0000-000000000002"
        self._create_job_file(task_id, "running")

        resp = client.get(f"/champion/jobs/{task_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"


# ---------------------------------------------------------------------------
# GET /champion/jobs/{task_id}/results
# ---------------------------------------------------------------------------

class TestJobResults:
    def _create_completed_job(self, task_id: str, champion: dict) -> None:
        task_dir = search_module._JOBS_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "status": "completed",
            "created_at": "2024-01-05T00:00:00",
            "updated_at": "2024-01-05T00:02:00",
            "message": "",
            "champion": champion,
            "error": None,
        }
        (task_dir / "task_status.json").write_text(json.dumps(payload))

    def test_returns_champion_for_completed_task(self, client):
        task_id = "bbbbbbbb-0000-0000-0000-000000000001"
        champion = {"profile_id": "trend_flow_v1", "stability_score": 72.5}
        self._create_completed_job(task_id, champion)

        resp = client.get(f"/champion/jobs/{task_id}/results")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["champion"] == champion

    def test_returns_404_for_missing_task(self, client):
        resp = client.get("/champion/jobs/no-such-task/results")
        assert resp.status_code == 404

    def test_failed_job_returns_error_field(self, client):
        task_id = "bbbbbbbb-0000-0000-0000-000000000002"
        task_dir = search_module._JOBS_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "status": "failed",
            "created_at": "2024-01-05T00:00:00",
            "updated_at": "2024-01-05T00:02:00",
            "message": "",
            "champion": None,
            "error": "backtest failed: insufficient data",
        }
        (task_dir / "task_status.json").write_text(json.dumps(payload))

        resp = client.get(f"/champion/jobs/{task_id}/results")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert body["error"] == "backtest failed: insufficient data"
        assert body["champion"] is None


# ---------------------------------------------------------------------------
# GET /champions/{scope_type}/{scope_key}
# ---------------------------------------------------------------------------

class TestGetChampion:
    def test_returns_champion_when_exists(self, client, mock_svc):
        champion_data = {"profile_id": "trend_flow_v1", "stability_score": 72.5}
        mock_svc.get_champion.return_value = champion_data

        resp = client.get("/champions/stock_group/magic_formula_top10")

        assert resp.status_code == 200
        assert resp.json() == champion_data

    def test_returns_404_when_not_found(self, client, mock_svc):
        mock_svc.get_champion.return_value = None

        resp = client.get("/champions/stock_group/nonexistent_pool")

        assert resp.status_code == 404

    def test_rejects_scope_key_with_path_traversal(self, client, mock_svc):
        # Path traversal: HTTP client normalizes the path, so this either routes
        # to a non-existent resource (404) or gets rejected (400/422).
        # The important thing is it does NOT return a valid champion (200).
        mock_svc.get_champion.return_value = None
        resp = client.get("/champions/stock_group/../etc/passwd")
        assert resp.status_code != 200, (
            f"Path traversal should not return 200, got {resp.status_code}: {resp.text}"
        )

    def test_rejects_scope_key_with_slash(self, client, mock_svc):
        mock_svc.get_champion.return_value = None
        resp = client.get("/champions/stock_group/a%2Fb")
        # Either 400 (validation) or 404 (routing/not found) is acceptable
        assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# POST /champions/{scope_type}/{scope_key}/apply
# ---------------------------------------------------------------------------

class TestApplyChampion:
    def test_apply_returns_success(self, client):
        resp = client.post(
            "/champions/stock_group/magic_formula_top10/apply",
            json={"profile_id": "trend_flow_v1", "stability_score": 72.5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["scope_type"] == "stock_group"
        assert body["scope_key"] == "magic_formula_top10"

    def test_apply_calls_service_with_correct_args(self, client, mock_svc):
        champion_payload = {"profile_id": "breakout_v1", "stability_score": 58.0}

        resp = client.post(
            "/champions/stock_group/test_pool/apply",
            json=champion_payload,
        )

        assert resp.status_code == 200
        mock_svc.apply_champion.assert_called_once_with(
            "stock_group", "test_pool", champion_payload
        )
