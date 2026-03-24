"""
TimesFM 预测因子服务

将 autoresearch/timesfm_cn_forecast 的单股预测能力封装为 FactorHub 因子。

因子逻辑：
  对每个时间点 t，用 [t-context_len, t) 的历史收盘价预测 t+horizon 步后的价格，
  计算预期收益率 = (pred_price - close[t]) / close[t]，写入 result[t]。

  这样 result[t] 表示"在 t 日收盘时，模型预测未来 horizon 日的涨跌幅"，
  与分析层的 future_return_N = close.pct_change(N).shift(-N) 对齐（同为 t 日视角）。

使用方式（在 FactorHub 因子代码框中）：
  def calculate_factor(df):
      from backend.services.timesfm_factor_service import timesfm_factor_service
      return timesfm_factor_service.compute(df, horizon=1, context_len=60)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# autoresearch 项目根目录（相对于 FactorHub）
_AUTORESEARCH_ROOT = Path(__file__).resolve().parents[3] / "autoresearch"
_AUTORESEARCH_SRC = str(_AUTORESEARCH_ROOT)


def _ensure_autoresearch_in_path() -> bool:
    """将 autoresearch 加入 sys.path，使其模块可被导入。"""
    if _AUTORESEARCH_SRC not in sys.path:
        if _AUTORESEARCH_ROOT.exists():
            sys.path.insert(0, _AUTORESEARCH_SRC)
            return True
        else:
            logger.warning(f"autoresearch 目录不存在: {_AUTORESEARCH_ROOT}")
            return False
    return True


def _resolve_model_dir(model_dir: Optional[str]) -> str:
    """将 None 解析为默认路径，保证比较时使用同一字符串。"""
    if model_dir is not None:
        return model_dir
    return str(_AUTORESEARCH_ROOT / "local_timesfm_model")


class TimesFMFactorService:
    """
    TimesFM 预测因子服务（单例，懒加载模型）。

    模型只在第一次调用（或切换 model_dir）时加载，后续复用。
    """

    def __init__(self):
        self._model = None
        self._loaded_model_dir: Optional[str] = None  # 已加载的解析后路径

    def get_status(self, model_dir: Optional[str] = None) -> dict:
        """返回当前运行状态，便于启动脚本和前端提示 TimesFM 是否就绪。"""
        resolved_model_dir = _resolve_model_dir(model_dir)
        status = {
            "available": False,
            "python_executable": sys.executable,
            "autoresearch_root": str(_AUTORESEARCH_ROOT),
            "autoresearch_exists": _AUTORESEARCH_ROOT.exists(),
            "model_dir": resolved_model_dir,
            "model_exists": Path(resolved_model_dir).exists(),
            "reason": None,
        }

        if not _ensure_autoresearch_in_path():
            status["reason"] = "autoresearch_not_found"
            return status

        try:
            from timesfm_cn_forecast.modeling import TimesFM_2p5_200M_torch  # noqa: F401
        except ImportError as exc:
            status["reason"] = f"import_error: {exc}"
            return status

        if not status["model_exists"]:
            status["reason"] = "model_not_found"
            return status

        status["available"] = True
        status["reason"] = "ready"
        return status

    def _load_model(self, model_dir: Optional[str] = None):
        """懒加载 TimesFM 模型。先解析路径再比较，避免 None vs 字符串不等的问题。"""
        resolved = _resolve_model_dir(model_dir)

        # 已加载且路径相同 → 直接复用
        if self._model is not None and resolved == self._loaded_model_dir:
            return self._model

        if not _ensure_autoresearch_in_path():
            raise ImportError(
                "无法找到 autoresearch 目录，请确认路径: " + str(_AUTORESEARCH_ROOT)
            )

        try:
            from timesfm_cn_forecast.modeling import 加载模型
        except ImportError as e:
            raise ImportError(
                f"无法导入 timesfm_cn_forecast，请确认 autoresearch 环境已配置: {e}"
            ) from e

        logger.info(f"正在加载 TimesFM 模型: {resolved}")
        self._model = 加载模型(resolved)
        self._loaded_model_dir = resolved
        logger.info("TimesFM 模型加载完成")
        return self._model

    def compute(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        context_len: int = 60,
        model_dir: Optional[str] = None,
    ) -> pd.Series:
        """
        计算 TimesFM 预测收益率因子（精确逐日版）。

        对每个时间点 t（从 context_len 开始），用 close[t-context_len:t] 预测
        horizon 步后的价格，计算 (pred - close[t]) / close[t] 写入 result[t]。

        Args:
            df: FactorHub 标准 OHLCV DataFrame（index 为 DatetimeIndex）
            horizon: 预测步长，1=预测明日，5=预测5日后
            context_len: 输入上下文长度（历史窗口），同时也是最小起始 bar 数
            model_dir: 模型目录，None 则使用默认路径

        Returns:
            pd.Series，index 与 df 相同，值为预期收益率
        """
        if "close" not in df.columns:
            raise ValueError("df 必须包含 'close' 列")

        if not self.is_available():
            logger.warning(
                "TimesFM 不可用（torch 未安装或模型目录不存在），"
                "返回全 NaN 序列。请使用 autoresearch venv 运行以启用此因子。"
            )
            return pd.Series(np.nan, index=df.index, name=f"timesfm_return_h{horizon}")

        model = self._load_model(model_dir)

        try:
            from timesfm_cn_forecast.modeling import 运行预测
        except ImportError as e:
            raise ImportError(f"无法导入 timesfm_cn_forecast: {e}") from e

        close = df["close"].values.astype(np.float32)
        n = len(close)
        result = np.full(n, np.nan, dtype=np.float64)

        # 从 context_len 开始，保证每次都有完整的 context_len 根历史 bar
        for i in range(context_len, n):
            context = close[i - context_len : i]  # shape: (context_len,)
            current_price = float(close[i])
            if current_price <= 0:
                continue
            try:
                点预测, _ = 运行预测(model, context, context_len, horizon)
                pred_price = float(点预测[horizon - 1])
                # result[i] = 在 t=i 时预测未来 horizon 步的涨跌幅
                result[i] = (pred_price - current_price) / current_price
            except Exception as e:
                logger.debug(f"第 {i} 个时间点预测失败: {e}")

        return pd.Series(result, index=df.index, name=f"timesfm_return_h{horizon}")

    def compute_fast(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        context_len: int = 60,
        model_dir: Optional[str] = None,
        step: int = 5,
    ) -> pd.Series:
        """
        快速版本：每隔 step 天预测一次，中间用线性插值填充。

        时间对齐与 compute() 相同：result[i] 表示 t=i 时对未来 horizon 步的预测。
        """
        if "close" not in df.columns:
            raise ValueError("df 必须包含 'close' 列")

        if not self.is_available():
            logger.warning(
                "TimesFM 不可用（torch 未安装或模型目录不存在），"
                "返回全 NaN 序列。请使用 autoresearch venv 运行以启用此因子。"
            )
            return pd.Series(np.nan, index=df.index, name=f"timesfm_return_h{horizon}_fast")

        model = self._load_model(model_dir)

        try:
            from timesfm_cn_forecast.modeling import 运行预测
        except ImportError as e:
            raise ImportError(f"无法导入 timesfm_cn_forecast: {e}") from e

        close = df["close"].values.astype(np.float32)
        n = len(close)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(context_len, n, step):
            context = close[i - context_len : i]
            current_price = float(close[i])
            if current_price <= 0:
                continue
            try:
                点预测, _ = 运行预测(model, context, context_len, horizon)
                pred_price = float(点预测[horizon - 1])
                result[i] = (pred_price - current_price) / current_price
            except Exception:
                continue

        s = pd.Series(result, index=df.index)
        s = s.interpolate(method="linear", limit_direction="forward").ffill()
        return s.rename(f"timesfm_return_h{horizon}_fast")

    def is_available(self) -> bool:
        """检查 TimesFM 是否可用（不加载模型）。"""
        return self.get_status().get("available", False)


# 全局单例
timesfm_factor_service = TimesFMFactorService()
