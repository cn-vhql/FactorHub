"""
时序打分 API 路由
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from backend.core.settings import settings
from backend.services.temporal_scoring_service import task_manager, temporal_scoring_service

router = APIRouter()


# ── 辅助函数 ───────────────────────────────────────────────────────────────────

def _resolve_trade_date(trade_date: Optional[str]) -> str:
    """trade_date 缺省时使用今日日期；服务层会自动回退到最近交易日"""
    if trade_date:
        return trade_date
    return datetime.now().strftime("%Y-%m-%d")


def _sync_threshold() -> int:
    return getattr(settings, "SCORING_SYNC_THRESHOLD", 20)


# ── Pydantic 模型 ──────────────────────────────────────────────────────────────

class ScoringRequest(BaseModel):
    codes: list[str]
    trade_date: Optional[str] = None
    profile_id: Optional[str] = None
    # 每股独立 profile：{"600519": "trend_flow_v1", "000651": "breakout_v1"}
    # 优先级高于 profile_id；未在此字典中的股票回退到 profile_id 或默认评分
    per_stock_profiles: Optional[dict[str, str]] = None


class ScoreItem(BaseModel):
    date: str
    code: str
    score: float


class SyncScoringResponse(BaseModel):
    success: bool
    data: list[ScoreItem]
    errors: list[dict]


class AsyncJobResponse(BaseModel):
    task_id: str
    status: str


class JobStatus(BaseModel):
    task_id: str
    status: str
    trade_date: str
    total: int
    processed: int
    success_count: int
    failed_count: int
    progress: float
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None


# ── 端点 ───────────────────────────────────────────────────────────────────────

@router.post("/temporal-score", response_model=SyncScoringResponse)
async def sync_score(request: ScoringRequest, background_tasks: BackgroundTasks):
    """
    同步评分（需求 1.1-1.4）。
    codes 超过 SCORING_SYNC_THRESHOLD 时自动升级为异步任务，返回 202。
    """
    trade_date = _resolve_trade_date(request.trade_date)
    if len(request.codes) > _sync_threshold():
        # 超限自动转异步，避免阻塞 worker
        try:
            task_id = task_manager.create_task(
                request.codes, trade_date, request.profile_id,
                per_stock_profiles=request.per_stock_profiles,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        background_tasks.add_task(
            task_manager.run_task, task_id, request.codes, trade_date, request.profile_id,
            per_stock_profiles=request.per_stock_profiles,
        )
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=202,
            content={"task_id": task_id, "status": "pending",
                     "detail": f"codes 数量 {len(request.codes)} 超过同步阈值 {_sync_threshold()}，已自动转为异步任务"},
        )
    results: list[dict] = []
    errors: list[dict] = []
    per_stock = request.per_stock_profiles or {}
    for code in request.codes:
        # 优先级：per_stock_profiles[code] > profile_id > 默认评分
        effective_profile = per_stock.get(code) or request.profile_id
        try:
            if effective_profile is not None:
                result = temporal_scoring_service.score_one_stock_with_profile(
                    code, trade_date, effective_profile
                )
            else:
                result = temporal_scoring_service.score_one_stock(code, trade_date)
            results.append(result)
        except Exception as exc:
            errors.append({"code": code, "error": str(exc)})
    data = [ScoreItem(date=r["date"], code=r["code"], score=r["score"]) for r in results]
    return SyncScoringResponse(success=True, data=data, errors=errors)

@router.post("/temporal-score/jobs", status_code=202, response_model=AsyncJobResponse)
async def async_score(request: ScoringRequest, background_tasks: BackgroundTasks):
    """异步任务创建（需求 2.1-2.5）"""
    trade_date = _resolve_trade_date(request.trade_date)
    try:
        task_id = task_manager.create_task(
            request.codes, trade_date, request.profile_id,
            per_stock_profiles=request.per_stock_profiles,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    background_tasks.add_task(
        task_manager.run_task, task_id, request.codes, trade_date, request.profile_id,
        per_stock_profiles=request.per_stock_profiles,
    )
    return AsyncJobResponse(task_id=task_id, status="pending")


@router.get("/temporal-score/jobs/{task_id}", response_model=JobStatus)
async def get_job_status(task_id: str):
    """任务状态查询（需求 3.1-3.5）"""
    try:
        status = task_manager.get_status(task_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    return status


@router.get("/temporal-score/jobs/{task_id}/results")
async def get_job_results(task_id: str, detail: bool = False):
    """任务结果查询（需求 4.1-4.4）"""
    try:
        results = task_manager.get_results(task_id, detail=detail)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return results
