"""
组合分析API路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.portfolio_analysis_service import portfolio_analysis_service

router = APIRouter()


def convert_numpy_types(obj):
    """
    递归转换 numpy 类型为 Python 原生类型
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # 处理特殊值
        if np.isnan(obj):
            return 0.0
        elif np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(x) for x in obj]
    else:
        return obj


def _load_factor_dataset(stock_code: str, factors: List[str], start_date: str, end_date: str):
    """加载单股票时序因子分析所需数据。"""
    from backend.services.data_service import data_service
    from backend.services.factor_service import factor_service
    from backend.repositories.factor_repository import FactorRepository
    from backend.core.database import get_db_session

    stock_data = data_service.get_stock_data(stock_code, start_date, end_date)
    if stock_data is None or len(stock_data) == 0:
        raise HTTPException(status_code=404, detail="未获取到数据")

    db = get_db_session()
    repo = FactorRepository(db)
    try:
        factor_defs = {}
        for factor_name in factors:
            factor = repo.get_by_name(factor_name)
            if factor:
                factor_defs[factor_name] = factor
    finally:
        db.close()

    if not factor_defs:
        raise HTTPException(status_code=400, detail="未找到任何有效的因子定义")

    factor_values = {}
    for factor_name, factor_def in factor_defs.items():
        values = factor_service.calculator.calculate(
            stock_data.copy(),
            factor_def.code,
            getattr(factor_def, "formula_type", None),
        )
        if values is not None and len(values.dropna()) > 0:
            factor_values[factor_name] = values.replace([np.inf, -np.inf], np.nan)

    if not factor_values:
        raise HTTPException(status_code=400, detail="没有有效的因子数据")

    future_returns = stock_data["close"].pct_change().shift(-1)
    return stock_data, factor_values, future_returns


def _build_factor_return_frame(factor_values: Dict[str, any], future_returns):
    """构建用于权重优化的因子收益率序列。"""
    factor_returns = {}
    for factor_name, values in factor_values.items():
        aligned = pd.DataFrame({
            "factor": values,
            "returns": future_returns,
        }).replace([np.inf, -np.inf], np.nan).dropna()

        if len(aligned) < 20:
            continue

        factor_std = aligned["factor"].std()
        if factor_std is None or np.isnan(factor_std) or factor_std == 0:
            continue

        normalized_factor = (aligned["factor"] - aligned["factor"].mean()) / factor_std
        factor_return_series = normalized_factor * aligned["returns"]
        factor_returns[factor_name] = factor_return_series

    factor_return_df = pd.DataFrame(factor_returns).dropna(how="all")
    if factor_return_df.empty:
        raise HTTPException(status_code=400, detail="因子收益率序列为空，无法进行权重优化")
    return factor_return_df


# ========== 数据模型 ==========

class OptimizeWeightsRequest(BaseModel):
    """权重优化请求"""
    stock_code: str
    factors: List[str]
    start_date: str
    end_date: str
    method: str = "equal_weight"
    rebalance_freq: str = "monthly"


class CompositeScoreRequest(BaseModel):
    """计算综合得分请求"""
    stock_code: str
    factors: List[str]
    start_date: str
    end_date: str
    method: str = "equal_weight"
    weights: Optional[Dict[str, float]] = None


class CompareMethodsRequest(BaseModel):
    """对比权重方法请求"""
    stock_code: str
    factors: List[str]
    start_date: str
    end_date: str
    methods: List[str] = ["equal_weight", "ic_weight"]


# ========== API端点 ==========

@router.post("/optimize-weights")
async def optimize_weights(request: OptimizeWeightsRequest):
    """优化权重"""
    try:
        stock_data, factor_values, returns = _load_factor_dataset(
            stock_code=request.stock_code,
            factors=request.factors,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        factor_return_df = _build_factor_return_frame(factor_values, returns)

        optimization_result = portfolio_analysis_service.optimize_weights(
            factor_returns=factor_return_df,
            method=request.method,
        )
        if "error" in optimization_result:
            raise HTTPException(status_code=400, detail=optimization_result["error"])

        weights = optimization_result["weights"]

        # 计算组合因子值和性能指标
        # 构建DataFrame用于计算，使用 stock_data 的索引
        factor_df = pd.DataFrame(index=stock_data.index)

        for factor_name, values in factor_values.items():
            factor_df[factor_name] = values

        # 计算加权组合因子
        weighted_factor = pd.Series(index=factor_df.index, dtype=float).fillna(0)
        for factor_name, weight in weights.items():
            if factor_name in factor_df.columns:
                weighted_factor += factor_df[factor_name].fillna(0) * weight

        weighted_factor = weighted_factor.dropna()

        # 计算未来收益率（用于IC计算）
        returns = stock_data['close'].pct_change().shift(-1)

        # 对齐数据 - 使用共同的索引
        common_index = weighted_factor.index.intersection(returns.index)

        if len(common_index) < 3:
            raise HTTPException(status_code=400, detail=f"有效数据点太少（{len(common_index)}个），无法计算组合指标")

        aligned_factor = weighted_factor.loc[common_index]
        aligned_returns = returns.loc[common_index]

        # 移除 NaN 值
        valid_mask = ~(aligned_factor.isna() | aligned_returns.isna())
        aligned_factor = aligned_factor[valid_mask]
        aligned_returns = aligned_returns[valid_mask]

        if len(aligned_factor) > 3:
            # 计算组合IC
            portfolio_ic = aligned_factor.corr(aligned_returns)

            # 计算组合收益率（因子的平均收益）
            portfolio_return = aligned_returns.mean()

            # 计算组合IR (IC均值 / IC标准差)
            ic_series = aligned_factor.rolling(window=20, min_periods=10).corr(aligned_returns)
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            portfolio_ir = ic_mean / ic_std if ic_std > 0 else 0
        else:
            portfolio_ic = 0
            portfolio_return = 0
            portfolio_ir = 0

        result = {
            "weights": weights,
            "method": request.method,
            "factors": request.factors,
            "stock_code": request.stock_code,
            "optimization": optimization_result,
            "metrics": {
                "return": float(portfolio_return),
                "ic": float(portfolio_ic),
                "ir": float(portfolio_ir)
            }
        }

        # 计算综合得分（使用优化后的权重）
        try:
            composite_score_result = portfolio_analysis_service.calculate_combined_factor_score(
                factor_data=factor_values,
                weights=weights,
                normalize=True
            )

            # 转换为列表格式
            if hasattr(composite_score_result, 'index'):
                composite_score = {
                    "dates": composite_score_result.index.astype(str).tolist(),
                    "values": composite_score_result.values.tolist()
                }
            else:
                composite_score = {"values": list(composite_score_result)}

            # 计算统计指标
            values = composite_score.get("values", [])
            if len(values) > 0:
                import numpy as np
                composite_stats = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
            else:
                composite_stats = {}

            # 转换 numpy 类型
            composite_score = convert_numpy_types(composite_score)
            composite_stats = convert_numpy_types(composite_stats)

        except Exception as e:
            print(f"[WARNING] 计算综合得分失败: {e}")
            composite_score = None
            composite_stats = {}

        # 添加综合得分到结果中
        result["composite_score"] = composite_score
        result["composite_stats"] = composite_stats

        # 转换 numpy 类型为 Python 原生类型，以避免 JSON 序列化错误
        result = convert_numpy_types(result)

        return {
            "success": True,
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/composite-score")
async def calculate_composite_score(request: CompositeScoreRequest):
    """计算综合得分"""
    try:
        stock_data, factor_data, future_returns = _load_factor_dataset(
            stock_code=request.stock_code,
            factors=request.factors,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        if request.weights:
            weights = {
                factor_name: float(weight)
                for factor_name, weight in request.weights.items()
                if factor_name in factor_data
            }
            if not weights:
                raise HTTPException(status_code=400, detail="提供的权重与有效因子不匹配")
        else:
            factor_return_df = _build_factor_return_frame(factor_data, future_returns)
            optimization_result = portfolio_analysis_service.optimize_weights(
                factor_returns=factor_return_df,
                method=request.method,
            )
            if "error" in optimization_result:
                raise HTTPException(status_code=400, detail=optimization_result["error"])
            weights = optimization_result["weights"]

        # 调用综合得分计算
        result = portfolio_analysis_service.calculate_combined_factor_score(
            factor_data=factor_data,
            weights=weights,
            normalize=True
        )

        # 转换为列表
        if hasattr(result, 'index'):
            score_list = {
                "dates": result.index.astype(str).tolist(),
                "values": result.values.tolist()
            }
        else:
            score_list = {"values": list(result)}

        # 转换 numpy 类型为 Python 原生类型，以避免 JSON 序列化错误
        score_list = convert_numpy_types(score_list)

        return {
            "success": True,
            "data": {
                **score_list,
                "weights": convert_numpy_types(weights),
                "method": request.method if not request.weights else "custom",
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-methods")
async def compare_weight_methods(request: CompareMethodsRequest):
    """对比权重方法 - 基于IC/IR指标评估不同权重优化方法的效果"""
    try:
        _, factor_data, returns = _load_factor_dataset(
            stock_code=request.stock_code,
            factors=request.factors,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        factor_return_df = _build_factor_return_frame(factor_data, returns)
        results = portfolio_analysis_service.compare_weight_methods(
            factor_returns=factor_return_df,
            methods=request.methods,
        )

        return {
            "success": True,
            "data": {
                "results": results
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
