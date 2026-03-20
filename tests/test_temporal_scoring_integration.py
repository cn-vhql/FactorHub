"""
集成验收测试：日频时序打分模块（temporal-scoring）

使用 FastAPI TestClient + unittest.mock.patch mock 掉真实网络请求，
返回合成 OHLCV 数据，验证 API 端点的完整行为。
"""
import uuid
import time
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.core.settings import settings

client = TestClient(app)


# ---------------------------------------------------------------------------
# 合成数据生成辅助函数
# ---------------------------------------------------------------------------

def make_ohlcv(n=400, seed=42) -> pd.DataFrame:
    """生成 n 行合成 OHLCV 数据，足够计算所有因子"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df


# ---------------------------------------------------------------------------
# 任务 13.1：10 只股票同步评分集成测试
# 需求 1.1、1.3
# ---------------------------------------------------------------------------

def test_sync_score_10_stocks():
    """
    需求 1.1、1.3
    调用 POST /api/scoring/temporal-score，codes 长度为 10
    验证响应包含 success=true，data 数组长度 + errors 数组长度 = 10
    验证每条记录的 date 字段与请求的 trade_date 一致
    """
    codes = [f"60000{i}" for i in range(1, 11)]  # 600001 ~ 600010
    trade_date = "2024-12-31"
    mock_df = make_ohlcv()

    with patch(
        "backend.services.data_service.data_service.get_stock_data",
        return_value=mock_df,
    ):
        response = client.post(
            "/api/scoring/temporal-score",
            json={"codes": codes, "trade_date": trade_date},
        )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    body = response.json()
    assert body["success"] is True, f"Expected success=True, got {body['success']}"

    data = body["data"]
    errors = body["errors"]
    assert len(data) + len(errors) == 10, (
        f"Expected data+errors=10, got data={len(data)}, errors={len(errors)}"
    )

    for item in data:
        assert item["date"] == trade_date, (
            f"Expected date={trade_date!r}, got {item['date']!r} for code={item['code']}"
        )


# ---------------------------------------------------------------------------
# 任务 13.2：50 只股票异步评分集成测试
# 需求 2.1、3.3、8.1
# ---------------------------------------------------------------------------

def test_async_score_50_stocks():
    """
    需求 2.1、3.3、8.1
    调用 POST /api/scoring/temporal-score/jobs，codes 长度为 50
    验证响应状态码为 202，响应体包含有效 UUID 格式的 task_id
    轮询 GET /api/scoring/temporal-score/jobs/{task_id} 直至 status=completed
    验证 daily_scores.csv 行数等于 success_count
    """
    codes = [f"{600000 + i:06d}" for i in range(1, 51)]  # 600001 ~ 600050
    trade_date = "2024-12-31"
    mock_df = make_ohlcv()

    with patch(
        "backend.services.data_service.data_service.get_stock_data",
        return_value=mock_df,
    ):
        response = client.post(
            "/api/scoring/temporal-score/jobs",
            json={"codes": codes, "trade_date": trade_date},
        )

    assert response.status_code == 202, (
        f"Expected 202, got {response.status_code}: {response.text}"
    )

    body = response.json()
    task_id = body["task_id"]

    # 验证 task_id 是有效 UUID
    try:
        uuid.UUID(task_id)
    except ValueError:
        pytest.fail(f"task_id {task_id!r} is not a valid UUID")

    # TestClient 中 BackgroundTasks 是同步执行的，任务在响应返回前已完成
    # 轮询状态（最多等待 60 秒，实际上应立即完成）
    deadline = time.time() + 60
    final_status = None
    while time.time() < deadline:
        status_resp = client.get(f"/api/scoring/temporal-score/jobs/{task_id}")
        assert status_resp.status_code == 200
        final_status = status_resp.json()
        if final_status["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert final_status is not None
    assert final_status["status"] == "completed", (
        f"Expected status=completed, got {final_status['status']}"
    )

    success_count = final_status["success_count"]

    # 读取 daily_scores.csv，验证数据行数（不含表头）== success_count
    task_dir = settings.SCORING_RESULTS_DIR / task_id
    scores_csv = task_dir / "daily_scores.csv"
    assert scores_csv.exists(), f"daily_scores.csv not found at {scores_csv}"

    lines = scores_csv.read_text(encoding="utf-8").splitlines()
    # 第一行是表头
    data_lines = [l for l in lines[1:] if l.strip()]
    assert len(data_lines) == success_count, (
        f"Expected {success_count} data rows in daily_scores.csv, got {len(data_lines)}"
    )


# ---------------------------------------------------------------------------
# 任务 13.3：200 只股票异步评分增量落盘集成测试
# 需求 8.1、8.4、3.2
# ---------------------------------------------------------------------------

def test_async_score_200_stocks_incremental():
    """
    需求 8.1、8.4、3.2
    调用 POST /api/scoring/temporal-score/jobs，codes 长度为 200
    等待任务完成后验证 status.json 中 processed == total
    """
    codes = [f"{600000 + i:06d}" for i in range(1, 201)]  # 600001 ~ 600200
    trade_date = "2024-12-31"
    mock_df = make_ohlcv()

    with patch(
        "backend.services.data_service.data_service.get_stock_data",
        return_value=mock_df,
    ):
        response = client.post(
            "/api/scoring/temporal-score/jobs",
            json={"codes": codes, "trade_date": trade_date},
        )

    assert response.status_code == 202, (
        f"Expected 202, got {response.status_code}: {response.text}"
    )

    task_id = response.json()["task_id"]

    # TestClient 中 BackgroundTasks 同步执行，任务已完成
    # 轮询直到 completed 或 failed
    deadline = time.time() + 120
    final_status = None
    while time.time() < deadline:
        status_resp = client.get(f"/api/scoring/temporal-score/jobs/{task_id}")
        assert status_resp.status_code == 200
        final_status = status_resp.json()
        if final_status["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert final_status is not None
    assert final_status["status"] == "completed", (
        f"Expected status=completed, got {final_status['status']}"
    )

    # 验证 processed == total
    assert final_status["processed"] == final_status["total"], (
        f"Expected processed={final_status['total']}, got {final_status['processed']}"
    )

    # 验证 daily_scores.csv 存在且有数据
    task_dir = settings.SCORING_RESULTS_DIR / task_id
    scores_csv = task_dir / "daily_scores.csv"
    assert scores_csv.exists(), f"daily_scores.csv not found at {scores_csv}"

    lines = scores_csv.read_text(encoding="utf-8").splitlines()
    data_lines = [l for l in lines[1:] if l.strip()]
    assert len(data_lines) > 0, "daily_scores.csv should have at least one data row"
