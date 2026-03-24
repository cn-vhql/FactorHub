"""
因子分析API路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import sys
from pathlib import Path


def safe_numeric_value(value):
    """安全处理数值，将NaN和无穷大转换为None"""
    if value is None:
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return float(value)


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.analysis_service import analysis_service
from backend.services.factor_stability_service import factor_stability_service
from backend.services.enhanced_analysis_service import enhanced_analysis_service
from backend.services.factor_exposure_service import factor_exposure_service
from backend.services.factor_effectiveness_service import factor_effectiveness_service
from backend.services.factor_attribution_service import factor_attribution_service
from backend.services.factor_monitoring_service import factor_monitoring_service

router = APIRouter()


# ========== 数据模型 ==========

class CalculateRequest(BaseModel):
    """计算因子值请求"""
    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str


class ICAnalysisRequest(BaseModel):
    """IC分析请求"""
    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str


class StabilityRequest(BaseModel):
    """稳定性检验请求"""
    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str


class MultiPeriodRequest(BaseModel):
    """多周期分析请求"""
    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str


# ========== API端点 ==========

@router.post("/calculate")
async def calculate_factor(request: CalculateRequest):
    """计算因子值"""
    import logging
    import traceback
    logger = logging.getLogger(__name__)

    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session

        # 获取因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factor = repo.get_by_name(request.factor_name)
        db.close()

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        logger.info(f"开始计算因子: {request.factor_name}, 代码: {factor.code}")

        # 获取数据并计算因子
        result_data = {}
        errors = []

        for stock_code in request.stock_codes:
            try:
                logger.info(f"获取股票数据: {stock_code}, 时间范围: {request.start_date} - {request.end_date}")
                data = data_service.get_stock_data(
                    stock_code,
                    request.start_date,
                    request.end_date
                )

                if data is None or len(data) == 0:
                    logger.warning(f"股票 {stock_code} 未获取到数据")
                    errors.append(f"股票 {stock_code} 未获取到数据")
                    continue

                logger.info(f"股票 {stock_code} 获取到 {len(data)} 条数据")

                # 使用 calculator 计算因子
                logger.info(f"开始计算因子值，因子代码: {factor.code}")
                factor_series = factor_service.calculator.calculate(data, factor.code)

                if factor_series is None:
                    logger.warning(f"股票 {stock_code} 因子计算返回 None")
                    errors.append(f"股票 {stock_code} 因子计算失败")
                    continue

                logger.info(f"因子计算完成，有效值数量: {factor_series.notna().sum()}/{len(factor_series)}")

                # 检查是否全为 NaN（因子不可用，如 TimesFM 在无 torch 环境下）
                if factor_series.notna().sum() == 0:
                    logger.warning(f"股票 {stock_code} 因子 {request.factor_name} 全部为 NaN，可能是该因子依赖的运行环境不满足（如 TimesFM 需要 torch）")
                    errors.append(f"因子 '{request.factor_name}' 返回全空值，该因子可能需要额外依赖（如 torch/TimesFM 模型）才能运行")
                    continue

                # 将因子值添加到数据中
                data[request.factor_name] = factor_series

                # 过滤掉因子值为 NaN 的行，确保 dates 和 factor_values 一一对应
                valid_data = data[[request.factor_name]].dropna()
                valid_dates = valid_data.index.strftime('%Y-%m-%d').tolist()
                valid_factor_values = valid_data[request.factor_name].tolist()

                # 额外检查：确保所有值都是有效的数字，转换 NaN 和 inf 为 None
                valid_factor_values = [safe_numeric_value(v) for v in valid_factor_values]

                # 移除值为 None 的项
                filtered_dates = []
                filtered_values = []
                for d, v in zip(valid_dates, valid_factor_values):
                    if v is not None:
                        filtered_dates.append(d)
                        filtered_values.append(v)

                valid_dates = filtered_dates
                valid_factor_values = filtered_values

                logger.info(f"股票 {stock_code}: 有效数据范围 {valid_dates[0] if valid_dates else '无'} 到 {valid_dates[-1] if valid_dates else '无'}, 共 {len(valid_dates)} 行")

                # 验证数据完整性
                if len(valid_dates) != len(valid_factor_values):
                    logger.error(f"数据长度不一致! dates={len(valid_dates)}, values={len(valid_factor_values)}")
                    errors.append(f"股票 {stock_code} 数据长度不一致")
                    continue

                # 转换为字典格式返回
                result_data[stock_code] = {
                    "dates": valid_dates,
                    "factor_values": valid_factor_values,
                    "statistics": {
                        "mean": safe_numeric_value(factor_series.mean()) if len(factor_series) > 0 else None,
                        "std": safe_numeric_value(factor_series.std()) if len(factor_series) > 0 else None,
                        "min": safe_numeric_value(factor_series.min()) if len(factor_series) > 0 else None,
                        "max": safe_numeric_value(factor_series.max()) if len(factor_series) > 0 else None,
                        "count": int(factor_series.count())
                    }
                }
                logger.info(f"股票 {stock_code} 因子计算成功")

            except Exception as e:
                logger.error(f"股票 {stock_code} 因子计算失败: {str(e)}\n{traceback.format_exc()}")
                errors.append(f"股票 {stock_code} 计算失败: {str(e)}")
                continue

        if not result_data:
            error_msg = f"因子计算失败或无有效数据。详情: {'; '.join(errors) if errors else '未知错误'}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        if errors:
            logger.warning(f"因子计算完成，但有部分错误: {'; '.join(errors)}")

        return {
            "success": True,
            "data": result_data,
            "warnings": errors if errors else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"因子计算异常: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"因子计算失败: {str(e)}")


@router.post("/ic")
async def calculate_ic(request: ICAnalysisRequest):
    """计算IC/IR"""
    import logging
    import traceback
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"开始IC分析: {request.factor_name}, 股票: {request.stock_codes}, 时间: {request.start_date} - {request.end_date}")

        # 先尝试使用缓存
        result = analysis_service.analyze(
            stock_codes=request.stock_codes,
            factor_names=[request.factor_name],
            start_date=request.start_date,
            end_date=request.end_date,
            use_cache=True,
            rolling_window=252
        )

        logger.info(f"IC分析原始结果: {result}")

        # 提取 IC/IR 相关数据并简化返回格式
        ic_ir_data = result.get("ic_ir", {})
        ic_stats = ic_ir_data.get("ic_stats", {})

        logger.info(f"提取的ic_stats: {ic_stats}")

        # 如果缓存中的数据无效，重新计算
        if not ic_stats or len(ic_stats) == 0:
            logger.warning("缓存中的数据无效，重新计算IC分析")
            result = analysis_service.analyze(
                stock_codes=request.stock_codes,
                factor_names=[request.factor_name],
                start_date=request.start_date,
                end_date=request.end_date,
                use_cache=False,  # 不使用缓存
                rolling_window=252
            )

            # 重新提取
            ic_ir_data = result.get("ic_ir", {})
            ic_stats = ic_ir_data.get("ic_stats", {})
            logger.info(f"重新计算的ic_stats: {ic_stats}")

        simplified_result = {
            "metadata": result.get("metadata", {}),
            "ic_stats": ic_stats,
        }

        # 检查是否有有效数据
        if not ic_stats or len(ic_stats) == 0:
            logger.warning("IC分析未返回有效统计数据")
            return {
                "success": True,
                "data": simplified_result,
                "message": "IC分析未返回有效统计数据，可能原因：股票数据不足或因子计算失败"
            }

        return {
            "success": True,
            "data": simplified_result
        }
    except Exception as e:
        logger.error(f"IC分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"IC分析失败: {str(e)}")


@router.post("/stability")
async def stability_test(request: StabilityRequest):
    """稳定性检验"""
    try:
        # 调用稳定性服务
        result = factor_stability_service.comprehensive_stability_test(
            factor_name=request.factor_name,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date
        )

        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-period")
async def multi_period_analysis(request: MultiPeriodRequest):
    """多周期分析"""
    try:
        # 调用增强分析服务
        result = enhanced_analysis_service.analyze_multi_period_ic(
            factor_name=request.factor_name,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date
        )

        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decay")
async def decay_analysis(request: ICAnalysisRequest):
    """因子衰减分析"""
    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session
        import pandas as pd
        import numpy as np

        # 获取因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factor = repo.get_by_name(request.factor_name)
        db.close()

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 计算因子在不同周期的IC（衰减分析）
        decay_periods = [1, 3, 5, 10, 20]  # 1日, 3日, 5日, 10日, 20日
        decay_results = []

        for period in decay_periods:
            all_ics = []
            for stock_code in request.stock_codes:
                data = data_service.get_stock_data(
                    stock_code,
                    request.start_date,
                    request.end_date
                )
                if data is not None and len(data) > 0:
                    # 计算因子
                    factor_series = factor_service.calculator.calculate(data, factor.code)
                    if factor_series is not None:
                        # 计算未来收益率
                        future_returns = data["close"].pct_change(period).shift(-period)
                        # 计算IC
                        ic = factor_series.rolling(20).corr(future_returns)
                        if not ic.empty and ic.dropna().count() > 0:
                            all_ics.append(ic.dropna().mean())

            if all_ics:
                mean_ic = np.mean(all_ics)
                decay_results.append({
                    "period": f"{period}日",
                    "ic_mean": float(mean_ic),
                    "period_days": period
                })

        result = {
            "factor_name": request.factor_name,
            "decay_analysis": decay_results
        }

        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exposure")
async def exposure_analysis(request: CalculateRequest):
    """因子暴露度分析"""
    import logging
    import traceback
    logger = logging.getLogger(__name__)

    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session

        logger.info(f"开始因子暴露度分析: {request.factor_name}, 股票: {request.stock_codes}")

        # 获取因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factor = repo.get_by_name(request.factor_name)
        db.close()

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 获取因子数据
        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes,
            [request.factor_name],
            request.start_date,
            request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        # 调用暴露度分析服务
        result = factor_exposure_service.calculate_exposure_metrics(
            factor_data=factor_data,
            factor_name=request.factor_name,
            window=20
        )

        return {
            "success": True,
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"因子暴露度分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"因子暴露度分析失败: {str(e)}")


@router.post("/effectiveness")
async def effectiveness_analysis(request: ICAnalysisRequest):
    """因子有效性分析"""
    import logging
    import traceback
    logger = logging.getLogger(__name__)

    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session

        logger.info(f"开始因子有效性分析: {request.factor_name}, 股票: {request.stock_codes}")

        # 获取因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factor = repo.get_by_name(request.factor_name)
        db.close()

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 获取因子数据
        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes,
            [request.factor_name],
            request.start_date,
            request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        # 调用有效性分析服务
        result = factor_effectiveness_service.analyze_effectiveness(
            factor_data=factor_data,
            factor_name=request.factor_name,
            future_periods=[1, 5, 10, 20]
        )

        return {
            "success": True,
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"因子有效性分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"因子有效性分析失败: {str(e)}")


@router.post("/attribution")
async def attribution_analysis(request: ICAnalysisRequest):
    """因子贡献度分解"""
    import logging
    import traceback
    logger = logging.getLogger(__name__)

    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session

        logger.info(f"开始因子贡献度分解: {request.factor_name}, 股票: {request.stock_codes}")

        # 获取因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factor = repo.get_by_name(request.factor_name)
        db.close()

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 获取因子数据
        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes,
            [request.factor_name],
            request.start_date,
            request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        # 调用贡献度分解服务
        result = factor_attribution_service.analyze_attribution(
            factor_data=factor_data,
            factor_name=request.factor_name,
            benchmark_data=None
        )

        return {
            "success": True,
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"因子贡献度分解失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"因子贡献度分解失败: {str(e)}")


@router.post("/monitoring")
async def monitoring_analysis(request: ICAnalysisRequest):
    """时间序列动态监测"""
    import logging
    import traceback
    logger = logging.getLogger(__name__)

    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session

        logger.info(f"开始时间序列动态监测: {request.factor_name}, 股票: {request.stock_codes}")

        # 获取因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factor = repo.get_by_name(request.factor_name)
        db.close()

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 获取因子数据
        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes,
            [request.factor_name],
            request.start_date,
            request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        # 调用动态监测服务
        result = factor_monitoring_service.monitor_dynamics(
            factor_data=factor_data,
            factor_name=request.factor_name
        )

        return {
            "success": True,
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"时间序列动态监测失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"时间序列动态监测失败: {str(e)}")
