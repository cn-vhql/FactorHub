"""
接口测试：大批量异步评分 jobs + profile_id 路径
覆盖：
  - POST /temporal-score 超阈值自动升级为异步（202）
  - POST /temporal-score/jobs 显式异步创建
  - GET  /temporal-score/jobs/{task_id} 状态查询
  - GET  /temporal-score/jobs/{task_id}/results 结果查询
  - profile_id 在 status.json 中正确持久化
  - per_stock_profiles 路由到各自 profile
"""
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.api.routers.scoring as scoring_module
from backend.api.routers.scoring import router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score_result(code: str, date: str = "2024-01-05", score: float = 72.0) -> dict:
    return {"date": date, "code": code, "score": score, "factors": {}, "pcts": {}}


def _make_app(tmp_path: Path, mock_task_manager, mock_scoring_svc):
    """Build a TestClient with mocked TaskManager and TemporalScoringService."""
    app = FastAPI()
    app.include_router(router)

    # Patch at module level so the router picks them up
    scoring_module.task_manager = mock_task_manager
    scoring_module.temporal_scoring_service = mock_scoring_svc

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_scoring_module_globals():
    """Restore scoring module globals after each test to avoid state pollution."""
    original_task_manager = scoring_module.task_manager
    original_scoring_svc = scoring_module.temporal_scoring_service
    yield
    scoring_module.task_manager = original_task_manager
    scoring_module.temporal_scoring_service = original_scoring_svc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_jobs(tmp_path):
    return tmp_path / "scoring_jobs"


@pytest.fixture()
def mock_scoring_svc():
    svc = MagicMock()
    svc.score_one_stock.side_effect = lambda code, date: _make_score_result(code, date)
    svc.score_one_stock_with_profile.side_effect = (
        lambda code, date, profile_id: _make_score_result(code, date, score=88.0)
    )
    return svc


@pytest.fixture()
def real_task_manager(tmp_jobs):
    """Use a real TaskManager backed by tmp_path so we can inspect files."""
    from backend.services.temporal_scoring_service import TaskManager
    mgr = TaskManager.__new__(TaskManager)
    mgr.results_dir = tmp_jobs
    mgr.max_workers = 2
    mgr._file_locks = {}
    # Wire up a mock scoring service
    mock_svc = MagicMock()
    mock_svc.score_one_stock.side_effect = lambda code, date: _make_score_result(code, date)
    mock_svc.score_one_stock_with_profile.side_effect = (
        lambda code, date, profile_id: _make_score_result(code, date, score=88.0)
    )
    mgr.scoring_service = mock_svc
    return mgr


@pytest.fixture()
def client_real(tmp_jobs, real_task_manager, mock_scoring_svc):
    app = FastAPI()
    app.include_router(router)
    scoring_module.task_manager = real_task_manager
    scoring_module.temporal_scoring_service = mock_scoring_svc
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. 同步端点超阈值自动升级为异步（202）
# ---------------------------------------------------------------------------

class TestSyncAutoUpgrade:
    def test_small_batch_returns_200(self, client_real):
        """codes <= threshold → 同步返回 200"""
        with patch.object(scoring_module, "_sync_threshold", return_value=5):
            resp = client_real.post("/temporal-score", json={
                "codes": ["000001", "000002"],
                "trade_date": "2024-01-05",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["data"]) == 2

    def test_large_batch_auto_upgrades_to_202(self, client_real):
        """codes > threshold → 自动升级为异步，返回 202 + task_id"""
        with patch.object(scoring_module, "_sync_threshold", return_value=2):
            resp = client_real.post("/temporal-score", json={
                "codes": ["000001", "000002", "000003"],
                "trade_date": "2024-01-05",
                "profile_id": "trend_flow_v1",
            })
        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert body["status"] == "pending"

    def test_large_batch_profile_id_persisted(self, client_real, real_task_manager):
        """自动升级时 profile_id 写入 status.json"""
        with patch.object(scoring_module, "_sync_threshold", return_value=1):
            resp = client_real.post("/temporal-score", json={
                "codes": ["000001", "000002"],
                "trade_date": "2024-01-05",
                "profile_id": "breakout_v1",
            })
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]

        status = real_task_manager.get_status(task_id)
        assert status["profile_id"] == "breakout_v1"


# ---------------------------------------------------------------------------
# 2. 显式异步 POST /temporal-score/jobs
# ---------------------------------------------------------------------------

class TestAsyncJobCreation:
    def test_creates_job_returns_202(self, client_real):
        resp = client_real.post("/temporal-score/jobs", json={
            "codes": ["600519", "000651", "000001"],
            "trade_date": "2024-01-05",
        })
        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert body["status"] == "pending"

    def test_profile_id_persisted_in_status_json(self, client_real, real_task_manager):
        """profile_id 必须写入 status.json"""
        resp = client_real.post("/temporal-score/jobs", json={
            "codes": ["600519", "000651"],
            "trade_date": "2024-01-05",
            "profile_id": "trend_flow_v1",
        })
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]

        status = real_task_manager.get_status(task_id)
        assert status["profile_id"] == "trend_flow_v1", (
            f"profile_id 未写入 status.json，实际: {status.get('profile_id')}"
        )

    def test_null_profile_id_stored_as_none(self, client_real, real_task_manager):
        """未传 profile_id 时 status.json 中应为 null"""
        resp = client_real.post("/temporal-score/jobs", json={
            "codes": ["600519"],
            "trade_date": "2024-01-05",
        })
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        status = real_task_manager.get_status(task_id)
        assert status["profile_id"] is None

    def test_large_batch_100_codes(self, client_real, real_task_manager):
        """100 只股票异步任务：status.json total=100"""
        codes = [f"{i:06d}" for i in range(1, 101)]
        resp = client_real.post("/temporal-score/jobs", json={
            "codes": codes,
            "trade_date": "2024-01-05",
            "profile_id": "momentum_v1",
        })
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        status = real_task_manager.get_status(task_id)
        assert status["total"] == 100
        assert status["profile_id"] == "momentum_v1"


# ---------------------------------------------------------------------------
# 3. GET /temporal-score/jobs/{task_id} 状态查询
# ---------------------------------------------------------------------------

class TestJobStatusQuery:
    def test_pending_status_returned(self, client_real):
        resp = client_real.post("/temporal-score/jobs", json={
            "codes": ["000001"],
            "trade_date": "2024-01-05",
        })
        task_id = resp.json()["task_id"]

        status_resp = client_real.get(f"/temporal-score/jobs/{task_id}")
        assert status_resp.status_code == 200
        body = status_resp.json()
        assert body["task_id"] == task_id
        assert body["status"] in ("pending", "running", "completed")
        assert "total" in body
        assert "progress" in body

    def test_nonexistent_task_returns_404(self, client_real):
        resp = client_real.get("/temporal-score/jobs/nonexistent-task-id-xyz")
        assert resp.status_code == 404

    def test_status_fields_complete(self, client_real):
        """status 响应包含所有必要字段"""
        resp = client_real.post("/temporal-score/jobs", json={
            "codes": ["000001", "000002"],
            "trade_date": "2024-01-05",
            "profile_id": "trend_flow_v1",
        })
        task_id = resp.json()["task_id"]
        status_resp = client_real.get(f"/temporal-score/jobs/{task_id}")
        body = status_resp.json()
        for field in ("task_id", "status", "total", "processed", "success_count",
                      "failed_count", "progress", "trade_date"):
            assert field in body, f"status 响应缺少字段: {field}"


# ---------------------------------------------------------------------------
# 4. 完整异步任务执行 + 结果查询
# ---------------------------------------------------------------------------

class TestAsyncJobExecution:
    def test_run_task_completes_and_results_readable(self, real_task_manager):
        """直接调用 run_task，验证完成后 get_results 可读"""
        codes = ["000001", "000002", "000003"]
        task_id = real_task_manager.create_task(codes, "2024-01-05", "trend_flow_v1")

        # 同步执行（测试环境不用 BackgroundTasks）
        real_task_manager.run_task(task_id, codes, "2024-01-05", "trend_flow_v1")

        status = real_task_manager.get_status(task_id)
        assert status["status"] == "completed"
        assert status["success_count"] == 3
        assert status["failed_count"] == 0
        assert status["profile_id"] == "trend_flow_v1"

        results = real_task_manager.get_results(task_id)
        assert len(results) == 3
        codes_in_results = {r["code"] for r in results}
        assert codes_in_results == set(codes)

    def test_run_task_profile_id_from_status_json(self, real_task_manager):
        """run_task 不传 profile_id 时，从 status.json 读取"""
        codes = ["600519"]
        task_id = real_task_manager.create_task(codes, "2024-01-05", "breakout_v1")

        # 不传 profile_id 参数
        real_task_manager.run_task(task_id, codes, "2024-01-05")

        # 验证 scoring_service 被以 profile 方式调用
        real_task_manager.scoring_service.score_one_stock_with_profile.assert_called_once_with(
            "600519", "2024-01-05", "breakout_v1"
        )

    def test_run_task_100_codes_all_succeed(self, real_task_manager):
        """100 只股票全部成功"""
        codes = [f"{i:06d}" for i in range(1, 101)]
        task_id = real_task_manager.create_task(codes, "2024-01-05", "momentum_v1")
        real_task_manager.run_task(task_id, codes, "2024-01-05", "momentum_v1")

        status = real_task_manager.get_status(task_id)
        assert status["status"] == "completed"
        assert status["success_count"] == 100
        assert status["failed_count"] == 0
        assert status["progress"] == 100.0

    def test_run_task_partial_failure_recorded(self, real_task_manager):
        """部分股票失败时，failed_count 正确，任务仍完成"""
        def flaky_score(code, date, profile_id):
            if code == "000002":
                raise ValueError("模拟数据不足")
            return _make_score_result(code, date)

        real_task_manager.scoring_service.score_one_stock_with_profile.side_effect = flaky_score

        codes = ["000001", "000002", "000003"]
        task_id = real_task_manager.create_task(codes, "2024-01-05", "trend_flow_v1")
        real_task_manager.run_task(task_id, codes, "2024-01-05", "trend_flow_v1")

        status = real_task_manager.get_status(task_id)
        assert status["status"] == "completed"
        assert status["success_count"] == 2
        assert status["failed_count"] == 1

    def test_results_not_available_before_completion(self, real_task_manager):
        """任务未完成时 get_results 抛出 ValueError"""
        codes = ["000001"]
        task_id = real_task_manager.create_task(codes, "2024-01-05")
        # 不执行 run_task，任务保持 pending
        with pytest.raises(ValueError, match="尚未完成"):
            real_task_manager.get_results(task_id)


# ---------------------------------------------------------------------------
# 5. per_stock_profiles 路由正确性
# ---------------------------------------------------------------------------

class TestPerStockProfiles:
    def test_per_stock_profiles_routes_each_code(self, client_real, mock_scoring_svc):
        """每只股票使用各自的 profile_id"""
        with patch.object(scoring_module, "_sync_threshold", return_value=100):
            resp = client_real.post("/temporal-score", json={
                "codes": ["000001", "000002"],
                "trade_date": "2024-01-05",
                "per_stock_profiles": {
                    "000001": "trend_flow_v1",
                    "000002": "breakout_v1",
                },
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["data"]) == 2

        calls = mock_scoring_svc.score_one_stock_with_profile.call_args_list
        call_map = {c.args[0]: c.args[2] for c in calls}
        assert call_map["000001"] == "trend_flow_v1"
        assert call_map["000002"] == "breakout_v1"

    def test_per_stock_fallback_to_profile_id(self, client_real, mock_scoring_svc):
        """per_stock_profiles 未覆盖的股票回退到 profile_id"""
        with patch.object(scoring_module, "_sync_threshold", return_value=100):
            resp = client_real.post("/temporal-score", json={
                "codes": ["000001", "000002"],
                "trade_date": "2024-01-05",
                "profile_id": "momentum_v1",
                "per_stock_profiles": {"000001": "trend_flow_v1"},
            })
        assert resp.status_code == 200
        calls = mock_scoring_svc.score_one_stock_with_profile.call_args_list
        call_map = {c.args[0]: c.args[2] for c in calls}
        assert call_map["000001"] == "trend_flow_v1"
        assert call_map["000002"] == "momentum_v1"

    def test_no_profile_uses_default_scoring(self, client_real, mock_scoring_svc):
        """无 profile_id 且无 per_stock_profiles 时使用默认评分"""
        with patch.object(scoring_module, "_sync_threshold", return_value=100):
            resp = client_real.post("/temporal-score", json={
                "codes": ["000001"],
                "trade_date": "2024-01-05",
            })
        assert resp.status_code == 200
        mock_scoring_svc.score_one_stock.assert_called_once_with("000001", "2024-01-05")
        mock_scoring_svc.score_one_stock_with_profile.assert_not_called()
