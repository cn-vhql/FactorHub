"""
使用 autoresearch venv 启动 FactorHub API（支持 TimesFM 因子）。

用法：
    ../autoresearch/.venv/bin/python start_api_with_timesfm.py
"""
import sys
import os
from pathlib import Path

# 把 autoresearch 加入 sys.path，使 timesfm_cn_forecast 可被导入
_AR_ROOT = Path(__file__).resolve().parent.parent / "autoresearch"
if str(_AR_ROOT) not in sys.path:
    sys.path.insert(0, str(_AR_ROOT))

# 把 FactorHub 根目录加入 sys.path
_FH_ROOT = Path(__file__).resolve().parent
if str(_FH_ROOT) not in sys.path:
    sys.path.insert(0, str(_FH_ROOT))

_RUNTIME_CACHE_ROOT = _FH_ROOT / "data" / ".runtime-cache"
_MPL_CACHE_DIR = _RUNTIME_CACHE_ROOT / "matplotlib"
_XDG_CACHE_DIR = _RUNTIME_CACHE_ROOT / "xdg"

for cache_dir in (_MPL_CACHE_DIR, _XDG_CACHE_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_DIR))

import uvicorn

if __name__ == "__main__":
    from backend.services.timesfm_factor_service import timesfm_factor_service

    status = timesfm_factor_service.get_status()

    print("=" * 50)
    print("启动 FactorFlow API 服务（含 TimesFM 支持）...")
    print(f"autoresearch 路径: {_AR_ROOT}")
    print(f"Python解释器: {sys.executable}")
    print(f"TimesFM状态: {'可用' if status.get('available') else '不可用'}")
    print(f"TimesFM模型目录: {status.get('model_dir')}")
    print("API地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # autoresearch venv 下关闭 reload 避免路径扫描问题
        log_level="info",
    )
