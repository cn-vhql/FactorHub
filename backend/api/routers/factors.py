"""
因子管理API路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.factor_service import factor_service
from backend.services.factor_generator_service import factor_generator_service
from backend.formula_engine.runtime import normalize_formula_type

router = APIRouter()


# ========== 数据模型 ==========

class FactorCreate(BaseModel):
    """创建因子请求"""
    name: str
    code: str
    category: str
    description: str = ""
    formula_type: str = "auto"  # auto / mylanguage / python


class FactorUpdate(BaseModel):
    """更新因子请求"""
    name: Optional[str] = None
    code: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    formula_type: Optional[str] = None


class BatchGenerateRequest(BaseModel):
    """批量生成因子请求"""
    base_factors: List[str]
    generate_methods: List[str]  # ["arithmetic", "statistics", "technical"]
    ic_threshold: float = 0.03
    ir_threshold: float = 0.5
    min_valid_ratio: float = 0.7


class PreselectRequest(BaseModel):
    """预筛选因子请求"""
    factors: List[str]
    ic_threshold: float = 0.03
    ir_threshold: float = 0.5
    min_valid_ratio: float = 0.7


# ========== API端点 ==========

@router.get("/")
async def get_factors(
    category: Optional[str] = None,
    source: Optional[str] = None
):
    """
    获取因子列表

    参数:
    - category: 分类筛选（可选）
    - source: 来源筛选 preset/user（可选）
    """
    try:
        factors = factor_service.get_all_factors()

        # 筛选
        if category:
            factors = [f for f in factors if f.get("category") == category]
        if source:
            factors = [f for f in factors if f.get("source") == source]

        return {
            "success": True,
            "data": factors,
            "total": len(factors)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_factor_stats():
    """获取因子统计信息"""
    try:
        stats = factor_service.get_factor_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{factor_id}")
async def get_factor(factor_id: int):
    """获取因子详情"""
    try:
        # 这里需要实现获取单个因子的逻辑
        factors = factor_service.get_all_factors()
        factor = next((f for f in factors if f.get("id") == factor_id), None)

        if not factor:
            raise HTTPException(status_code=404, detail="因子不存在")

        return {
            "success": True,
            "data": factor
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def create_factor(request: FactorCreate):
    """创建新因子"""
    try:
        # 创建因子
        factor = factor_service.create_factor(
            name=request.name,
            code=request.code,
            category=request.category,
            description=request.description,
            formula_type=request.formula_type
        )

        return {
            "success": True,
            "data": factor,
            "message": "因子创建成功"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{factor_id}")
async def update_factor(factor_id: int, request: FactorUpdate):
    """更新因子"""
    try:
        # 更新因子
        factor_service.update_factor(
            factor_id=factor_id,
            name=request.name,
            code=request.code,
            category=request.category,
            description=request.description,
            formula_type=request.formula_type,
        )

        return {
            "success": True,
            "message": "因子更新成功"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{factor_id}")
async def delete_factor(factor_id: int):
    """删除因子"""
    try:
        success = factor_service.delete_factor(factor_id)

        if not success:
            raise HTTPException(status_code=404, detail="因子不存在或删除失败")

        return {
            "success": True,
            "message": "因子删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-generate")
async def batch_generate_factors(request: BatchGenerateRequest):
    """批量生成因子"""
    try:
        all_generated_factors = []

        # 根据选择的生成方法调用相应的函数
        for method in request.generate_methods:
            if method == "arithmetic":
                # 算术运算组合
                factors = factor_generator_service.generate_binary_combinations(
                    base_factors=request.base_factors,
                    max_depth=2,
                    max_combinations=50
                )
                all_generated_factors.extend(factors)

            elif method == "statistics":
                # 统计变换
                factors = factor_generator_service.generate_statistical_combinations(
                    base_factors=request.base_factors,
                    max_combinations=50
                )
                all_generated_factors.extend(factors)

            elif method == "technical":
                # 技术指标组合
                factors = factor_generator_service.generate_indicator_combinations(
                    base_factors=request.base_factors,
                    max_combinations=30
                )
                all_generated_factors.extend(factors)

        # 混合因子生成
        if len(request.generate_methods) > 1:
            hybrid_factors = factor_generator_service.generate_hybrid_factors(
                base_factors=request.base_factors,
                n_factors=20
            )
            all_generated_factors.extend(hybrid_factors)

        # 去重（处理混合了字符串和字典的情况）
        seen = set()
        unique_factors = []
        for factor in all_generated_factors:
            # 如果是字典，使用其expression字段作为唯一标识
            key = factor["expression"] if isinstance(factor, dict) else factor
            if key not in seen:
                seen.add(key)
                unique_factors.append(factor)

        all_generated_factors = unique_factors

        result = {
            "generated_count": len(all_generated_factors),
            "factors": all_generated_factors[:20],  # 只返回前20个示例
            "total_possible": len(all_generated_factors)
        }

        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preselect")
async def preselect_factors(request: PreselectRequest):
    """预筛选因子"""
    try:
        if not request.factors:
            return {
                "success": True,
                "data": {
                    "total": 0,
                    "selected": 0,
                    "factors": [],
                    "details": []
                }
            }

        from backend.core.settings import settings
        from backend.core.database import get_db_session
        from backend.repositories.factor_repository import FactorRepository
        from backend.services.data_service import data_service
        from backend.services.factor_validation_service import FactorValidationService
        import pandas as pd
        import numpy as np

        sample_stock_codes = [
            "000001.SZ",
            "000002.SZ",
            "600000.SH",
            "600036.SH",
            "600519.SH",
        ]
        stock_data_map = data_service.get_multiple_stocks_data(
            stock_codes=sample_stock_codes,
            start_date=settings.DEFAULT_START_DATE,
            end_date=settings.DEFAULT_END_DATE,
            use_cache=True,
        )

        if not stock_data_map:
            raise HTTPException(status_code=503, detail="预筛选失败：无法获取样本股票数据")

        validator = FactorValidationService(
            ic_threshold=request.ic_threshold,
            ir_threshold=request.ir_threshold,
        )

        db = get_db_session()
        repo = FactorRepository(db)
        try:
            selected_factors = []
            details = []

            for factor_identifier in request.factors:
                try:
                    factor_record = repo.get_by_name(factor_identifier)
                    factor_code = factor_record.code if factor_record else factor_identifier
                    factor_formula_type = getattr(factor_record, "formula_type", None)

                    ic_values = []
                    ir_values = []
                    valid_ratios = []

                    for stock_code, stock_df in stock_data_map.items():
                        if stock_df is None or len(stock_df) == 0:
                            continue

                        factor_values = factor_service.calculator.calculate(
                            stock_df.copy(),
                            factor_code,
                            factor_formula_type,
                        )
                        future_returns = stock_df["close"].pct_change().shift(-1)

                        aligned = pd.DataFrame({
                            "factor": factor_values,
                            "return": future_returns,
                        }).replace([np.inf, -np.inf], np.nan).dropna()

                        valid_ratio = len(aligned) / len(stock_df) if len(stock_df) > 0 else 0.0
                        valid_ratios.append(valid_ratio)

                        if aligned.empty:
                            continue

                        validation = validator.validate_factor(
                            factor_values=aligned["factor"],
                            return_values=aligned["return"],
                        )

                        ic_values.append(abs(validation["ic_validation"]["ic"]))
                        ir_values.append(validation["ir_validation"]["ir"])

                    avg_ic = float(np.mean(ic_values)) if ic_values else 0.0
                    avg_ir = float(np.mean(ir_values)) if ir_values else 0.0
                    avg_valid_ratio = float(np.mean(valid_ratios)) if valid_ratios else 0.0
                    passed = (
                        avg_ic >= request.ic_threshold and
                        avg_ir >= request.ir_threshold and
                        avg_valid_ratio >= request.min_valid_ratio
                    )

                    detail = {
                        "factor": factor_identifier,
                        "code": factor_code,
                        "avg_ic": avg_ic,
                        "avg_ir": avg_ir,
                        "avg_valid_ratio": avg_valid_ratio,
                        "passed": passed,
                        "sample_size": len(valid_ratios),
                    }

                    if passed:
                        selected_factors.append(factor_identifier)
                except Exception as factor_error:
                    detail = {
                        "factor": factor_identifier,
                        "code": factor_identifier,
                        "avg_ic": 0.0,
                        "avg_ir": 0.0,
                        "avg_valid_ratio": 0.0,
                        "passed": False,
                        "sample_size": 0,
                        "error": str(factor_error),
                    }

                details.append(detail)
        finally:
            db.close()

        return {
            "success": True,
            "data": {
                "total": len(request.factors),
                "selected": len(selected_factors),
                "factors": selected_factors,
                "details": details,
                "sample_stock_codes": sample_stock_codes,
                "evaluation_period": {
                    "start_date": settings.DEFAULT_START_DATE,
                    "end_date": settings.DEFAULT_END_DATE,
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_factor(request: dict):
    """验证因子表达式"""
    try:
        code = request.get("code", "")
        formula_type = request.get("formula_type", "auto")

        if not code:
            return {
                "success": False,
                "message": "因子表达式不能为空"
            }

        # 字符检查：确保只包含合法字符
        import re
        # 使用更宽松的检查：只禁止控制字符，允许所有可打印字符（包括中文）
        if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', code):
            return {
                "success": False,
                "message": "因子表达式包含非法控制字符"
            }

        resolved_type = normalize_formula_type(formula_type, code)

        # 调用真正的验证逻辑：执行代码来测试
        is_valid, message = factor_service.validate_factor_code(code, formula_type=resolved_type)

        if not is_valid:
            return {
                "success": False,
                "message": message
            }

        return {
            "success": True,
            "data": {
                "code": code,
                "formula_type": resolved_type,
                "valid": True
            },
            "message": message
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


@router.post("/{factor_id}/copy")
async def copy_factor(factor_id: int):
    """复制因子"""
    try:
        # 获取原因子信息
        factors = factor_service.get_all_factors()
        original_factor = next((f for f in factors if f.get("id") == factor_id), None)

        if not original_factor:
            raise HTTPException(status_code=404, detail="因子不存在")

        # 生成新的因子名称（名称_数字）
        base_name = original_factor.get("name", "")
        new_name = base_name

        # 查找已存在的同名副本数量
        existing_copies = [
            f for f in factors
            if f.get("source") == "user" and f.get("name", "").startswith(base_name + "_")
        ]

        # 提取已有的数字后缀
        suffix_numbers = []
        for f in existing_copies:
            name = f.get("name", "")
            if name.startswith(base_name + "_"):
                suffix = name[len(base_name) + 1:]
                if suffix.isdigit():
                    suffix_numbers.append(int(suffix))

        # 生成新的数字后缀
        if suffix_numbers:
            new_suffix = max(suffix_numbers) + 1
        else:
            new_suffix = 1

        new_name = f"{base_name}_{new_suffix}"

        # 创建新因子（作为用户自定义因子）
        new_factor = factor_service.create_factor(
            name=new_name,
            code=original_factor.get("code", ""),
            category=original_factor.get("category", ""),
            description=original_factor.get("description", ""),
            formula_type=original_factor.get("formula_type", "auto")
        )

        return {
            "success": True,
            "data": new_factor,
            "message": f"因子已复制为 {new_name}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
