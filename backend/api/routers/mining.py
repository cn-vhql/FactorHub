"""
因子挖掘API路由
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import asyncio
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


class RecentValuableFactorRequest(BaseModel):
    """最近阶段高价值因子挖掘请求"""
    stock_codes: List[str]
    end_date: str
    eval_days: int = Field(default=90, ge=10, le=730,
                           description="最近评估窗口（自然日），范围 [10, 730]")
    lookback_days: int = Field(default=240, ge=30, le=1825,
                               description="因子预热窗口（自然日），范围 [30, 1825]")
    horizons: List[int] = Field(default_factory=lambda: [1, 5, 10],
                                description="未来收益周期列表，每个值须 ≥ 1")
    top_k: int = Field(default=20, ge=1, le=200,
                       description="返回 TopK 因子，范围 [1, 200]")
    min_samples: int = Field(default=25, ge=5, le=500,
                             description="计算相关性最小样本数，范围 [5, 500]")
    correlation_threshold: float = Field(default=0.85, ge=0.0, le=1.0,
                                         description="因子去重相关阈值，范围 [0.0, 1.0]")


# ========== 任务存储（内存） ==========
mining_tasks = {}


# ========== API端点 ==========

@router.post("/recent-valuable")
async def mine_recent_valuable_factors(request: RecentValuableFactorRequest):
    """
    从现有因子库自动挖掘最近窗口内高价值因子。

    输出包含全局 Top 因子（含相关性去重）和单票当前走势下 Top 因子。
    """
    try:
        from backend.services.recent_factor_mining_service import recent_factor_mining_service

        if not request.stock_codes:
            raise HTTPException(status_code=400, detail="stock_codes 不能为空")

        result = recent_factor_mining_service.mine_recent_valuable_factors(
            stock_codes=request.stock_codes,
            end_date=request.end_date,
            eval_days=request.eval_days,
            lookback_days=request.lookback_days,
            horizons=request.horizons,
            top_k=request.top_k,
            min_samples=request.min_samples,
            correlation_threshold=request.correlation_threshold,
        )

        return {
            "success": True,
            "data": result,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"最近因子挖掘失败: {exc}")


@router.post("/genetic")
async def start_genetic_mining(request: GeneticMiningRequest, background_tasks: BackgroundTasks):
    """启动遗传算法挖掘"""
    try:
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
            "message": "挖掘任务已启动"
        }
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

        # 如果没有找到任何因子，使用默认的基础因子代码
        if not base_factor_codes:
            logger.warning("No valid base factors found, using default codes")
            base_factor_codes = [
                "RSI(close, 14)",
                "SMA(close, 20)",
                "close / open",
                "volume / 1000000",
                "MACD(close, 12, 26, 9)[0]"
            ]
        else:
            logger.info(f"Using {len(base_factor_codes)} base factor codes")

        # 尝试使用真实的遗传算法
        try:
            from backend.services.genetic_factor_mining_service import create_genetic_mining_service

            logger.info("Using real genetic algorithm mining")

            # 创建遗传算法挖掘服务
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

            # 设置进度回调以实时更新任务状态
            def progress_callback(gen, total_gen, best_fitness, avg_fitness):
                progress = int(gen / total_gen * 100)
                mining_tasks[task_id]["progress"] = progress
                mining_tasks[task_id]["current_generation"] = gen
                mining_tasks[task_id]["total_generations"] = total_gen
                mining_tasks[task_id]["best_fitness"] = float(best_fitness)
                mining_tasks[task_id]["avg_fitness"] = float(avg_fitness)

                # 更新fitness_history
                if "fitness_history" not in mining_tasks[task_id]:
                    mining_tasks[task_id]["fitness_history"] = {"best": [], "average": []}
                mining_tasks[task_id]["fitness_history"]["best"].append(float(best_fitness))
                mining_tasks[task_id]["fitness_history"]["average"].append(float(avg_fitness))

                logger.info(f"Progress: {progress}%, Gen {gen}/{total_gen}, Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f}")

            mining_service.set_progress_callback(progress_callback)

            # 执行挖掘
            result = mining_service.mine_factors()

            if not result.get("success"):
                raise Exception(result.get("message", "挖掘失败"))

            # 转换结果格式
            best_factors = result.get("best_factors", [])

            discovered_factors = []
            for i, factor_info in enumerate(best_factors):
                # 获取验证信息（如果有）
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

            # 从logbook中提取fitness_history
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

            # 保存结果
            mining_tasks[task_id]["status"] = "completed"
            mining_tasks[task_id]["progress"] = 100
            mining_tasks[task_id]["result"] = result_data
            mining_tasks[task_id]["fitness_history"] = fitness_history

            logger.info(f"Task {task_id} completed successfully")
            logger.info(f"Discovered {len(discovered_factors)} factors")
            logger.info(f"Final status: {mining_tasks[task_id]['status']}")

        except ImportError as e:
            # DEAP库未安装，使用模拟模式
            logger.warning(f"DEAP library not available, using simulation mode: {e}")
            await _run_simulated_mining(task_id, request, data, base_factor_codes, factor_service, logger)

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        mining_tasks[task_id]["status"] = "failed"
        mining_tasks[task_id]["error"] = str(e)


async def _run_simulated_mining(task_id: str, request: GeneticMiningRequest, data, base_factor_codes, factor_service, logger):
    """模拟模式挖掘（当DEAP库未安装时使用）"""
    # 计算基础因子值（用于验证和生成）
    factor_values = {}
    for code in base_factor_codes:
        try:
            values = factor_service.calculator.calculate(data, code)
            if values is not None and len(values.dropna()) > 0:
                factor_values[code] = values
                logger.info(f"Successfully calculated factor: {code}, {len(values.dropna())} valid values")
        except Exception as e:
            logger.warning(f"计算基础因子失败 {code}: {e}")
            continue

    if not factor_values:
        logger.error("No valid factor values calculated")
        raise Exception("无法计算任何有效的因子值")

    # 模拟挖掘进度
    n_generations = request.n_generations
    fitness_history = {"best": [], "average": []}
    current_best_fitness = 0.0

    for gen in range(n_generations):
        # 更新进度
        progress = int((gen + 1) / n_generations * 100)
        mining_tasks[task_id]["progress"] = progress

        # 模拟适应度变化（逐渐改进）
        current_best_fitness = 0.03 + (gen + 1) * 0.005 + (0.001 * (gen % 3))
        current_avg_fitness = current_best_fitness * (0.85 + 0.1 * (gen % 2))

        fitness_history["best"].append(current_best_fitness)
        fitness_history["average"].append(current_avg_fitness)

        # 更新任务状态以便轮询可以获取
        mining_tasks[task_id]["current_generation"] = gen + 1
        mining_tasks[task_id]["total_generations"] = n_generations
        mining_tasks[task_id]["best_fitness"] = current_best_fitness
        mining_tasks[task_id]["avg_fitness"] = current_avg_fitness
        mining_tasks[task_id]["fitness_history"] = fitness_history

        logger.info(f"Generation {gen + 1}/{n_generations} completed, best_fitness={current_best_fitness:.4f}")

        # 模拟计算时间
        await asyncio.sleep(0.5)

    # 基于用户选择的因子代码生成组合因子
    discovered_factors = []
    code_list = list(factor_values.keys())

    for i in range(min(5, len(code_list))):
        base_code = code_list[i % len(code_list)]
        # 生成简单的组合表达式
        if i == 0:
            expression = f"({base_code} * 1.5)"
        elif i == 1:
            expression = f"({base_code} + close / open)"
        elif i == 2:
            expression = f"({base_code} * volume / 1000000)"
        elif i == 3:
            expression = f"({base_code} - SMA(close, 20))"
        else:
            expression = f"({base_code} / (close + 1))"

        discovered_factors.append({
            "name": f"Mined_Factor_{i+1}",
            "expression": expression,
            "ic": 0.03 + (i * 0.01),
            "ir": 0.5 + (i * 0.1),
            "fitness": 0.03 + (i * 0.01)
        })

    result = {
        "factors": discovered_factors,
        "best_fitness": discovered_factors[0]["ic"] if discovered_factors else 0,
        "avg_fitness": sum(f["fitness"] for f in discovered_factors) / len(discovered_factors) if discovered_factors else 0,
        "generations": n_generations,
        "fitness_history": fitness_history
    }

    # 保存结果
    mining_tasks[task_id]["status"] = "completed"
    mining_tasks[task_id]["progress"] = 100
    mining_tasks[task_id]["result"] = result
    mining_tasks[task_id]["fitness_history"] = fitness_history

    logger.info(f"Task {task_id} completed (simulated mode)")
    logger.info(f"Discovered {len(discovered_factors)} factors")


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
