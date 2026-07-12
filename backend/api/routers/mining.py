"""
因子挖掘API路由
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

router = APIRouter()


# ========== 数据模型 ==========

class GeneticMiningRequest(BaseModel):
    """遗传算法挖掘请求"""
    stock_code: str
    base_factors: List[str] = []  # 基础因子列表
    start_date: str
    end_date: str
    population_size: int = 50
    n_generations: int = 10
    cx_prob: float = 0.7
    mut_prob: float = 0.3
    elite_size: int = 5
    fitness_objective: str = "ic_mean"
    ic_threshold: float = 0.03


# ========== 任务存储（内存） ==========
mining_tasks = {}


# ========== API端点 ==========

@router.post("/genetic")
async def start_genetic_mining(request: GeneticMiningRequest, background_tasks: BackgroundTasks):
    """启动遗传算法挖掘"""
    try:
        from backend.services.genetic_factor_mining_service import DEAP_AVAILABLE

        if not DEAP_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="遗传算法依赖 DEAP 未安装，无法启动真实挖掘任务"
            )

        # 生成任务ID
        import uuid
        task_id = str(uuid.uuid4())

        # 初始化任务状态
        mining_tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "result": None,
            "error": None
        }

        # 在后台执行挖掘
        background_tasks.add_task(
            _run_genetic_mining,
            task_id,
            request
        )

        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "status": "pending"
            },
            "message": "因子挖掘任务已启动"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _run_genetic_mining(task_id: str, request: GeneticMiningRequest):
    """后台执行遗传算法挖掘"""
    try:
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Starting mining task {task_id}")
        logger.info(f"Stock: {request.stock_code}, Base factors: {request.base_factors}")
        logger.info(f"Parameters: population={request.population_size}, generations={request.n_generations}")

        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session
        from backend.services.data_service import data_service

        # 更新状态
        mining_tasks[task_id]["status"] = "running"
        logger.info(f"Task {task_id} status set to running")

        # 获取数据
        data = data_service.get_stock_data(
            request.stock_code,
            request.start_date,
            request.end_date
        )

        if data is None or len(data) == 0:
            raise Exception("未获取到有效数据")

        logger.info(f"Retrieved {len(data)} rows of data")

        # 计算收益率
        if "close" in data.columns:
            data["return"] = data["close"].pct_change()

        # 获取基础因子列表
        # 前端传递的是因子名称，需要从数据库获取因子代码
        base_factor_codes = []
        if request.base_factors and len(request.base_factors) > 0:
            # 从数据库获取因子定义
            try:
                from backend.repositories.factor_repository import FactorRepository
                db = get_db_session()
                repo = FactorRepository(db)

                for factor_name in request.base_factors:
                    factor = repo.get_by_name(factor_name)
                    if factor:
                        base_factor_codes.append(factor.code)
                        logger.info(f"Found factor: {factor_name} -> {factor.code}")
                    else:
                        logger.warning(f"Factor not found in database: {factor_name}")

                db.close()
            except Exception as e:
                logger.error(f"Error loading factors from database: {e}")

        if not request.base_factors:
            raise Exception("未提供基础因子，已中止因子挖掘任务")

        if not base_factor_codes:
            raise Exception("所选基础因子均不可用，已中止因子挖掘任务")

        logger.info(f"Using {len(base_factor_codes)} base factor codes")

        from backend.services.genetic_factor_mining_service import create_genetic_mining_service

        logger.info("Using real genetic algorithm mining")

        mining_service = create_genetic_mining_service(
            base_factors=base_factor_codes,
            data=data,
            return_column="return",
            population_size=request.population_size,
            n_generations=request.n_generations,
            cx_prob=request.cx_prob,
            mut_prob=request.mut_prob,
            factor_calculator=factor_service.calculator,
        )

        def progress_callback(gen, total_gen, best_fitness, avg_fitness):
            progress = int(gen / total_gen * 100)
            mining_tasks[task_id]["progress"] = progress
            mining_tasks[task_id]["current_generation"] = gen
            mining_tasks[task_id]["total_generations"] = total_gen
            mining_tasks[task_id]["best_fitness"] = float(best_fitness)
            mining_tasks[task_id]["avg_fitness"] = float(avg_fitness)

            if "fitness_history" not in mining_tasks[task_id]:
                mining_tasks[task_id]["fitness_history"] = {"best": [], "average": []}
            mining_tasks[task_id]["fitness_history"]["best"].append(float(best_fitness))
            mining_tasks[task_id]["fitness_history"]["average"].append(float(avg_fitness))

            logger.info(f"Progress: {progress}%, Gen {gen}/{total_gen}, Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f}")

        mining_service.set_progress_callback(progress_callback)

        result = mining_service.mine_factors()

        if not result.get("success"):
            raise Exception(result.get("message", "挖掘失败"))

        best_factors = result.get("best_factors", [])

        discovered_factors = []
        for i, factor_info in enumerate(best_factors):
            validation = factor_info.get("validation", {})
            ic = validation.get("ic_validation", {}).get("ic", 0.0)
            ir = validation.get("ir_validation", {}).get("ir", 0.0)
            fitness = factor_info.get("fitness", 0.0)

            discovered_factors.append({
                "name": f"Mined_Factor_{i+1}",
                "expression": factor_info["expression"],
                "ic": float(ic) if ic else 0.0,
                "ir": float(ir) if ir else 0.0,
                "fitness": float(fitness),
            })

        logbook = result.get("logbook")
        if logbook is not None:
            fitness_history = {
                "best": [float(gen["max"]) for gen in logbook],
                "average": [float(gen["avg"]) for gen in logbook]
            }
        else:
            fitness_history = {"best": [], "average": []}

        result_data = {
            "factors": discovered_factors,
            "best_fitness": float(discovered_factors[0]["fitness"]) if discovered_factors else 0.0,
            "avg_fitness": sum(f["fitness"] for f in discovered_factors) / len(discovered_factors) if discovered_factors else 0.0,
            "generations": request.n_generations,
            "fitness_history": fitness_history
        }

        mining_tasks[task_id]["status"] = "completed"
        mining_tasks[task_id]["progress"] = 100
        mining_tasks[task_id]["result"] = result_data
        mining_tasks[task_id]["fitness_history"] = fitness_history

        logger.info(f"Task {task_id} completed successfully")
        logger.info(f"Discovered {len(discovered_factors)} factors")
        logger.info(f"Final status: {mining_tasks[task_id]['status']}")

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        mining_tasks[task_id]["status"] = "failed"
        mining_tasks[task_id]["error"] = str(e)

@router.get("/status/{task_id}")
async def get_mining_status(task_id: str):
    """获取挖掘状态"""
    import logging
    logger = logging.getLogger(__name__)

    if task_id not in mining_tasks:
        logger.warning(f"Status requested for non-existent task {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    task = mining_tasks[task_id]
    logger.info(f"Status check for task {task_id}: {task['status']}")

    # 构造返回数据，包含前端期望的所有字段
    response_data = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
        "error": task.get("error")
    }

    # 如果任务完成，添加结果信息
    if task["status"] == "completed" and "result" in task:
        result = task["result"]
        response_data["current_generation"] = result.get("generations", 0)
        response_data["total_generations"] = result.get("generations", 0)
        response_data["best_fitness"] = result.get("best_fitness", 0)
        response_data["avg_fitness"] = result.get("avg_fitness", 0)
        response_data["fitness_history"] = result.get("fitness_history", {"best": [], "average": []})
        logger.info(f"Returning completed status with fitness_history length: {len(response_data['fitness_history']['best'])}")
    else:
        # 进行中的任务 - 从任务状态获取实时数据
        response_data["current_generation"] = task.get("current_generation", 0)
        response_data["total_generations"] = task.get("total_generations", 10)
        response_data["best_fitness"] = task.get("best_fitness", 0.03)
        response_data["avg_fitness"] = task.get("avg_fitness", 0.03)
        response_data["fitness_history"] = task.get("fitness_history", {"best": [], "average": []})
        logger.info(f"Returning running status: gen {response_data['current_generation']}/{response_data['total_generations']}")

    return {
        "success": True,
        "data": response_data
    }


@router.get("/results/{task_id}")
async def get_mining_results(task_id: str):
    """获取挖掘结果"""
    if task_id not in mining_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = mining_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态: {task['status']}")

    return {
        "success": True,
        "data": task["result"]
    }
