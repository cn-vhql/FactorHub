"""
AI 模型配置与因子生成 API。
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.ai_factor_service import ai_factor_service
from backend.services.ai_analysis_service import ai_factor_analysis_service


router = APIRouter()


class AIModelConfigRequest(BaseModel):
    """AI 模型配置请求。"""

    base_url: str
    model: str
    api_key: str = ""
    request_path: str = "/chat/completions"


class AIFactorGenerateRequest(BaseModel):
    """AI 生成因子请求。"""

    requirement: str
    formula_type: str = "auto"
    suggested_name: str = ""
    description_hint: str = ""
    max_rounds: int = 4


class AIFactorInterpretRequest(BaseModel):
    """AI 因子流式解读请求。"""

    factor: Dict[str, Any]
    stock_code: str
    start_date: str
    end_date: str
    chart_period: Optional[str] = None
    analysis_context: Dict[str, Any]


@router.get("/model-config")
async def get_model_config():
    """获取当前 AI 模型配置。"""
    try:
        return {
            "success": True,
            "data": ai_factor_service.get_config_view(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/model-config")
async def save_model_config(request: AIModelConfigRequest):
    """保存 AI 模型配置。"""
    try:
        result = ai_factor_service.save_config(request.model_dump())
        return {
            "success": True,
            "data": result,
            "message": "模型配置已保存",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/model-config/validate")
async def validate_model_config(request: Optional[AIModelConfigRequest] = None):
    """验证 AI 模型配置是否生效。"""
    try:
        payload = request.model_dump() if request is not None else None
        result = ai_factor_service.validate_config(payload)
        return {
            "success": True,
            "data": result,
            "message": "模型配置验证通过",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/generate-factor")
async def generate_factor_with_ai(request: AIFactorGenerateRequest):
    """使用 AI 根据自然语言生成因子表达式。"""
    try:
        result = ai_factor_service.generate_factor(request.model_dump())
        return {
            "success": True,
            "data": result,
            "message": f"AI 已在第 {result['attempts']} 轮生成可执行因子",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/interpret-factor/stream")
async def stream_ai_factor_interpretation(request: AIFactorInterpretRequest):
    """流式生成 AI 因子分析报告。"""
    try:
        generator = ai_factor_analysis_service.stream_analysis_report(request.model_dump())
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
