"""
FastAPI主应用
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
import sys
from pathlib import Path
import numpy as np
import json

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.settings import settings
from backend.core.database import init_db
from backend.services.factor_service import factor_service
from backend.services.data_service import data_service

# 导入路由
from .routers import (
    factors,
    analysis,
    mining,
    portfolio,
    backtest,
    data,
    scoring,
    search,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("启动FastAPI服务...")
    init_db()
    factor_service.load_preset_factors()
    print("数据库和预置因子加载完成")

    yield

    # 关闭时清理
    print("关闭FastAPI服务...")


# 自定义JSON编码器来处理numpy浮点数值
class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# 配置JSON编码器
def jsonable_encoder_with_numpy(obj, *args, **kwargs):
    """处理numpy类型的JSON编码器"""
    try:
        return jsonable_encoder(obj, *args, **kwargs, custom_serializer=lambda x: NumpyJSONEncoder().default(x))
    except:
        return jsonable_encoder(obj, *args, **kwargs)


# 创建FastAPI应用
app = FastAPI(
    title="FactorFlow API",
    description="股票因子分析系统 REST API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=JSONResponse
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],  # 允许的前端来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# 注册路由
app.include_router(factors.router, prefix="/api/factors", tags=["因子管理"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["因子分析"])
app.include_router(mining.router, prefix="/api/mining", tags=["因子挖掘"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["组合分析"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["策略回测"])
app.include_router(data.router, prefix="/api/data", tags=["数据管理"])
app.include_router(scoring.router, prefix="/api/scoring", tags=["时序打分"])
app.include_router(search.router, prefix="/api/search", tags=["策略搜索"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "FactorFlow API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    print(f"[ERROR] 请求错误: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": str(exc),
            "detail": "服务器内部错误"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
