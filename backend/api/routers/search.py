"""
Champion 策略搜索 API 路由
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.champion_strategy_service import ChampionStrategyService
from backend.services.factor_profile_registry_service import FactorProfileRegistryService
from backend.services.temporal_pool_service import TemporalPoolService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["champion-search"])

_JOBS_DIR = Path("data/champion_jobs")

# ---------------------------------------------------------------------------
# 依赖注入
# ---------------------------------------------------------------------------

def _get_champion_svc() -> ChampionStrategyService:
    return ChampionStrategyService()


def _get_profile_registry() -> FactorProfileRegistryService:
    return FactorProfileRegistryService()


def _get_pool_svc() -> TemporalPoolService:
    return TemporalPoolService()


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------

class TemporalPoolSearchRequest(BaseModel):
    codes: list[str] = Field(..., description="股票代码列表")
    search_space_id: str = Field(default="temporal_default", description="搜索空间 ID")
    start_date: str = Field(..., description="回测开始日期 YYYY-MM-DD")
    end_date: str = Field(..., description="回测结束日期 YYYY-MM-DD")
    max_workers: int = Field(default=1, ge=1, le=16)


class CrossSectionalSearchRequest(BaseModel):
    universe: str = Field(..., description="截面 universe 名称")
    search_space_id: str = Field(default="cross_sectional_default", description="搜索空间 ID")
    start_date: str = Field(..., description="回测开始日期 YYYY-MM-DD")
    end_date: str = Field(..., description="回测结束日期 YYYY-MM-DD")


class StockSearchRequest(BaseModel):
    code: str = Field(..., description="股票代码")
    search_space_id: str = Field(default="temporal_default", description="搜索空间 ID")
    start_date: str = Field(..., description="回测开始日期 YYYY-MM-DD")
    end_date: str = Field(..., description="回测结束日期 YYYY-MM-DD")


class JobStatusResponse(BaseModel):
    task_id: str
    status: str  # pending | running | completed | failed
    created_at: str
    updated_at: str
    message: str = ""


class JobResultResponse(BaseModel):
    task_id: str
    status: str
    champion: dict | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# 路径安全校验
# ---------------------------------------------------------------------------

def _validate_scope_key(scope_key: str) -> None:
    if "/" in scope_key or ".." in scope_key:
        raise HTTPException(status_code=400, detail=f"scope_key 包含非法字符: {scope_key!r}")


# ---------------------------------------------------------------------------
# 后台任务执行器
# ---------------------------------------------------------------------------

def _run_champion_job(task_id: str, job_type: str, request: Any, svc: ChampionStrategyService) -> None:
    """在后台线程中执行 champion 搜索，并将结果写入任务目录。"""
    task_dir = _JOBS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    status_file = task_dir / "task_status.json"

    def _write_status(status: str, message: str = "", champion: dict | None = None, error: str | None = None):
        # 读取已有 created_at，避免覆盖
        existing_created_at = None
        if status_file.exists():
            try:
                existing_created_at = json.loads(status_file.read_text()).get("created_at")
            except Exception:
                pass
        payload = {
            "task_id": task_id,
            "status": status,
            "created_at": existing_created_at or datetime.now(timezone.utc).isoformat(),
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "champion": champion,
            "error": error,
        }
        status_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    _write_status("running")
    try:
        if job_type == "temporal_pool":
            result = svc.run_for_pool(
                codes=request.codes,
                search_space_id=request.search_space_id,
                start_date=request.start_date,
                end_date=request.end_date,
                max_workers=request.max_workers,
            )
        elif job_type == "cross_sectional":
            result = svc.run_for_cross_sectional(
                universe=request.universe,
                search_space_id=request.search_space_id,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        elif job_type == "single_stock":
            result = svc.run_for_stock(
                code=request.code,
                search_space_id=request.search_space_id,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        else:
            raise ValueError(f"Unknown job_type: {job_type}")

        _write_status("completed", champion=result)
    except Exception as exc:
        logger.exception("Champion job %s failed", task_id)
        _write_status("failed", error=str(exc))


def _create_job(task_id: str) -> dict:
    """创建初始任务状态文件，返回状态字典。"""
    task_dir = _JOBS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "task_id": task_id,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "message": "任务已创建，等待执行",
        "champion": None,
        "error": None,
    }
    (task_dir / "task_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@router.post("/champion/temporal-pool", response_model=JobStatusResponse, status_code=202)
async def start_temporal_pool_search(
    request: TemporalPoolSearchRequest,
    background_tasks: BackgroundTasks,
    svc: ChampionStrategyService = Depends(_get_champion_svc),
):
    """启动时序股票池 champion 搜索任务。"""
    task_id = str(uuid.uuid4())
    payload = _create_job(task_id)
    background_tasks.add_task(_run_champion_job, task_id, "temporal_pool", request, svc)
    return JobStatusResponse(
        task_id=task_id,
        status=payload["status"],
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
        message=payload["message"],
    )


@router.post("/champion/cross-sectional", response_model=JobStatusResponse, status_code=202)
async def start_cross_sectional_search(
    request: CrossSectionalSearchRequest,
    background_tasks: BackgroundTasks,
    svc: ChampionStrategyService = Depends(_get_champion_svc),
):
    """启动截面 champion 搜索任务。"""
    task_id = str(uuid.uuid4())
    payload = _create_job(task_id)
    background_tasks.add_task(_run_champion_job, task_id, "cross_sectional", request, svc)
    return JobStatusResponse(
        task_id=task_id,
        status=payload["status"],
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
        message=payload["message"],
    )


@router.post("/champion/single-stock", response_model=JobStatusResponse, status_code=202)
async def start_single_stock_search(
    request: StockSearchRequest,
    background_tasks: BackgroundTasks,
    svc: ChampionStrategyService = Depends(_get_champion_svc),
):
    """启动单股 champion 搜索任务。"""
    task_id = str(uuid.uuid4())
    payload = _create_job(task_id)
    background_tasks.add_task(_run_champion_job, task_id, "single_stock", request, svc)
    return JobStatusResponse(
        task_id=task_id,
        status=payload["status"],
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
        message=payload["message"],
    )


@router.get("/champion/jobs/{task_id}", response_model=JobStatusResponse)
async def get_job_status(task_id: str):
    """查询 champion 搜索任务状态。"""
    status_file = _JOBS_DIR / task_id / "task_status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    data = json.loads(status_file.read_text())
    return JobStatusResponse(
        task_id=data["task_id"],
        status=data["status"],
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        message=data.get("message", ""),
    )


@router.get("/champion/jobs/{task_id}/results", response_model=JobResultResponse)
async def get_job_results(task_id: str):
    """获取 champion 搜索任务结果。"""
    status_file = _JOBS_DIR / task_id / "task_status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    data = json.loads(status_file.read_text())
    return JobResultResponse(
        task_id=data["task_id"],
        status=data["status"],
        champion=data.get("champion"),
        error=data.get("error"),
    )


@router.get("/champions/{scope_type}/{scope_key}")
async def get_champion(
    scope_type: str,
    scope_key: str,
    svc: ChampionStrategyService = Depends(_get_champion_svc),
):
    """获取指定作用域的 champion 配置。"""
    _validate_scope_key(scope_key)
    champion = svc.get_champion(scope_type, scope_key)
    if champion is None:
        raise HTTPException(status_code=404, detail=f"Champion 不存在: {scope_type}/{scope_key}")
    return champion


@router.post("/champions/{scope_type}/{scope_key}/apply")
async def apply_champion(
    scope_type: str,
    scope_key: str,
    champion: dict,
    svc: ChampionStrategyService = Depends(_get_champion_svc),
):
    """将指定 champion 应用到对应作用域。"""
    _validate_scope_key(scope_key)
    svc.apply_champion(scope_type, scope_key, champion)
    return {"success": True, "scope_type": scope_type, "scope_key": scope_key}


@router.get("/profiles")
async def list_profiles(
    mode: Optional[str] = Query(default=None, description="按 mode 过滤（temporal_pool / cross_sectional_filter）"),
    registry: FactorProfileRegistryService = Depends(_get_profile_registry),
):
    """列出所有可用的因子 profile，支持按 mode 过滤。"""
    if mode is not None:
        profiles = registry.load_profiles(mode=mode)
    else:
        # 返回两种 mode 的所有 profile
        profiles = (
            registry.load_profiles("temporal_pool")
            + registry.load_profiles("cross_sectional_filter")
        )
    return {"profiles": profiles}


@router.get("/search-spaces")
async def list_search_spaces(
    registry: FactorProfileRegistryService = Depends(_get_profile_registry),
):
    """列出所有可用的 search space 配置。"""
    # 优先从 settings 读取，回退到配置文件中已知的默认 ID
    try:
        from backend.core.settings import settings
        space_ids: list[str] = getattr(settings, "SEARCH_SPACE_IDS", [])
    except Exception:
        space_ids = []

    if not space_ids:
        space_ids = ["temporal_default", "cross_sectional_default"]

    spaces = []
    for sid in space_ids:
        try:
            spaces.append({"id": sid, **registry.load_search_space(sid)})
        except KeyError:
            pass
    return {"search_spaces": spaces}


@router.get("/pools")
async def list_pools(
    pool_svc: TemporalPoolService = Depends(_get_pool_svc),
):
    """列出所有可用的时序股票池。"""
    pools = pool_svc.list_pools() if hasattr(pool_svc, "list_pools") else []
    return {"pools": pools}
