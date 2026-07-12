"""
FastAPI主应用
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
import sys
from pathlib import Path
import numpy as np
import json
import os

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.settings import settings
from backend.core.database import init_db
from backend.services.factor_service import factor_service
from backend.services.data_service import data_service

# 导入路由
from .routers import (
    ai,
    factors,
    analysis,
    mining,
    portfolio,
    backtest,
    data
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("启动FastAPI服务...")
    init_db()
    factor_service.load_preset_factors()
    cleanup_summary = factor_service.cleanup_legacy_generated_factors()
    app.json_encoder = NumpyJSONEncoder
    if cleanup_summary["migrated"] or cleanup_summary["deleted"]:
        print(
            "历史旧包装因子已清理: "
            f"迁移 {cleanup_summary['migrated']} 个, 删除 {cleanup_summary['deleted']} 个"
        )
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

# ============================================
# Static File Serving (for production)
# ============================================
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "react-antd" / "dist"

if FRONTEND_DIST.exists():
    print(f"Serving static files from: {FRONTEND_DIST}")
    # Mount static assets directory (js, css, images, etc.)
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.exception_handler(404)
async def spa_fallback(request, exc):
    """SPA fallback - return index.html for 404 errors (non-API routes)"""
    # Only handle non-API routes for HTML requests
    if FRONTEND_DIST.exists() and not request.url.path.startswith(("/api", "/docs", "/redoc", "/openapi.json")):
        return FileResponse(FRONTEND_DIST / "index.html")
    return JSONResponse(
        status_code=404,
        content={"detail": "Not Found"}
    )


# 注册路由
app.include_router(factors.router, prefix="/api/factors", tags=["因子管理"])
app.include_router(ai.router, prefix="/api/ai", tags=["AI 因子"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["因子分析"])
app.include_router(mining.router, prefix="/api/mining", tags=["因子挖掘"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["组合分析"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["策略回测"])
app.include_router(data.router, prefix="/api/data", tags=["数据管理"])


@app.get("/api")
async def api_root():
    """API 根路径"""
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


# 全局异常处理
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
