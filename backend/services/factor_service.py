"""
因子服务模块 - 因子计算与管理
"""
import numpy as np
import pandas as pd
import talib
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import logging

from backend.core.database import get_db_session
from backend.core.settings import settings
from backend.formula_engine import FormulaExecutionResult, formula_engine_manager
from backend.formula_engine.code_normalizer import normalize_formula_code
from backend.formula_engine.runtime import FormulaExecutionError, infer_formula_type, normalize_formula_type
from backend.models.factor import FactorModel
from backend.repositories.factor_repository import FactorRepository
from backend.services.data_service import data_service
from backend.services.factor_version_service import factor_version_service

# 配置日志
logger = logging.getLogger(__name__)


class FactorCalculator:
    """因子计算器 - 执行因子计算逻辑"""

    def __init__(self):
        self.engine = formula_engine_manager

    def infer_formula_type(self, factor_code: str, formula_type: Optional[str] = None) -> str:
        stripped = (factor_code or "").strip()
        return normalize_formula_type(formula_type, stripped)

    def _prepare_execution_code(self, factor_code: str, formula_type: Optional[str] = None) -> str:
        """执行前仅规范化 Python 代码，麦语言保留原程序以支持多输出。"""
        stripped = (factor_code or "").strip()
        if not stripped:
            return stripped

        resolved_type = normalize_formula_type(formula_type, stripped)
        if resolved_type == "python":
            return normalize_formula_code(stripped, formula_type="python")
        return stripped

    def calculate(self, df: pd.DataFrame, factor_code: str, formula_type: Optional[str] = None) -> pd.Series:
        """
        计算单个因子

        Args:
            df: 包含OHLCV数据的DataFrame
            factor_code: 因子计算代码
            formula_type: 公式类型（mylanguage / python / auto）

        Returns:
            因子值的Series
        """
        resolved_type = self.infer_formula_type(factor_code, formula_type)
        prepared_code = self._prepare_execution_code(factor_code, resolved_type)
        try:
            return self.engine.execute(df, prepared_code, resolved_type)
        except FormulaExecutionError as exc:
            logger.error("因子计算失败 [%s]: %s", resolved_type, prepared_code, exc_info=True)
            raise ValueError(f"{resolved_type} 因子计算失败: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("因子计算异常 [%s]: %s", resolved_type, prepared_code, exc_info=True)
            raise ValueError(f"{resolved_type} 因子计算失败: {exc}") from exc

    def calculate_with_metadata(
        self,
        df: pd.DataFrame,
        factor_code: str,
        formula_type: Optional[str] = None,
    ) -> FormulaExecutionResult:
        """计算因子并返回主输出与绘图输出。"""
        resolved_type = self.infer_formula_type(factor_code, formula_type)
        prepared_code = self._prepare_execution_code(factor_code, resolved_type)
        try:
            result = self.engine.execute_with_metadata(df, prepared_code, resolved_type)
            if result.formula_type != resolved_type:
                return FormulaExecutionResult(
                    primary=result.primary,
                    plots=result.plots,
                    formula_type=resolved_type,
                )
            return result
        except FormulaExecutionError as exc:
            logger.error("因子计算失败 [%s]: %s", resolved_type, prepared_code, exc_info=True)
            raise ValueError(f"{resolved_type} 因子计算失败: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("因子计算异常 [%s]: %s", resolved_type, prepared_code, exc_info=True)
            raise ValueError(f"{resolved_type} 因子计算失败: {exc}") from exc

    def calculate_multiple(
        self, df: pd.DataFrame, factors: List[FactorModel]
    ) -> pd.DataFrame:
        """
        计算多个因子

        Args:
            df: 包含OHLCV数据的DataFrame
            factors: 因子模型列表

        Returns:
            包含所有因子值的DataFrame
        """
        result = pd.DataFrame(index=df.index)

        for factor in factors:
            try:
                factor_values = self.calculate(df, factor.code, getattr(factor, "formula_type", None))
                result[factor.name] = factor_values
            except Exception as e:
                logger.warning(f"计算因子 {factor.name} 失败: {e}")
                result[factor.name] = np.nan

        return result

    def rolling_standardize(self, df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        滚动窗口标准化

        Args:
            df: 因子数据DataFrame
            window: 滚动窗口大小

        Returns:
            标准化后的DataFrame
        """
        result = df.copy()
        for col in df.columns:
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            # 避免除以0，当标准差为0或接近0时，返回0而不是inf
            rolling_std_safe = rolling_std.replace(0, np.nan).fillna(1e-10)
            result[col] = (df[col] - rolling_mean) / rolling_std_safe
            # 将无穷大值替换为NaN
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
        return result

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时间特征

        Args:
            df: 包含日期索引的DataFrame

        Returns:
            添加时间特征后的DataFrame
        """
        result = df.copy()
        if isinstance(result.index, pd.DatetimeIndex):
            result["day_of_week"] = result.index.dayofweek
            result["month"] = result.index.month
            result["quarter"] = result.index.quarter
        return result


class FactorService:
    """因子服务类"""

    def __init__(self):
        self.calculator = FactorCalculator()

    def load_preset_factors(self) -> None:
        """从配置文件加载预置因子"""
        config_path = settings.CONFIG_DIR / "factors.yaml"

        if not config_path.exists():
            # 如果配置文件不存在，创建默认预置因子
            self._create_default_preset_factors()
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 如果配置文件为空或加载失败，创建默认因子
        if config is None:
            self._create_default_preset_factors()
            return

        self._sync_preset_factors(config)

    def _create_default_preset_factors(self) -> None:
        """创建默认预置因子"""
        self._sync_preset_factors(self._get_default_factors())

    def _sync_preset_factors(self, preset_factors: Dict[str, List[Dict]]) -> None:
        """将预置因子定义同步到数据库。"""
        db = get_db_session()
        repo = FactorRepository(db)

        try:
            for category, factors in preset_factors.items():
                for factor_data in factors:
                    resolved_type = factor_data.get("formula_type", infer_formula_type(factor_data["code"]))
                    existing = repo.get_by_name(factor_data["name"], include_inactive=True)

                    if existing is None:
                        factor = FactorModel(
                            name=factor_data["name"],
                            code=factor_data["code"],
                            formula_type=resolved_type,
                            description=factor_data.get("description", ""),
                            source="preset",
                            category=category,
                            is_active=1,
                        )
                        repo.create(factor)
                        continue

                    if existing.source != "preset":
                        logger.warning("跳过预置因子同步，名称被用户因子占用: %s", factor_data["name"])
                        continue

                    changed = False
                    desired_values = {
                        "code": factor_data["code"],
                        "formula_type": resolved_type,
                        "description": factor_data.get("description", ""),
                        "category": category,
                        "is_active": 1,
                    }
                    for field, desired in desired_values.items():
                        if getattr(existing, field) != desired:
                            setattr(existing, field, desired)
                            changed = True

                    if changed:
                        repo.update(existing)
        finally:
            db.close()

    def _get_default_factors(self) -> Dict[str, List[Dict]]:
        """获取默认预置因子定义"""
        return {
            "价格收益率": [
                {
                    "name": "log_return_1",
                    "code": "np.log(close / close.shift(1))",
                    "description": "日对数收益率",
                },
                {
                    "name": "log_return_5",
                    "code": "np.log(close / close.shift(5))",
                    "description": "5日累计收益",
                },
                {
                    "name": "price_vs_sma20",
                    "code": "close / SMA(close, timeperiod=20)",
                    "description": "相对20日均线位置",
                },
                {
                    "name": "price_vs_sma60",
                    "code": "close / SMA(close, timeperiod=60)",
                    "description": "相对60日均线位置",
                },
                {
                    "name": "sma20_vs_sma60",
                    "code": "SMA(close, timeperiod=20) / SMA(close, timeperiod=60)",
                    "description": "短期vs长期趋势方向",
                },
                {
                    "name": "high_low_ratio",
                    "code": "(high - low) / open",
                    "description": "日内波动幅度",
                },
                {
                    "name": "close_open_ratio",
                    "code": "close / open",
                    "description": "收盘相对开盘强度",
                },
            ],
            "动量趋势": [
                {
                    "name": "rsi_14",
                    "code": "RSI(close, timeperiod=14)",
                    "description": "RSI(14) 超买超卖指标",
                },
                {
                    "name": "macd_line",
                    "code": "MACD(close, fastperiod=12, slowperiod=26)[0]",
                    "description": "MACD差值线",
                },
                {
                    "name": "macd_signal",
                    "code": "MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[1]",
                    "description": "MACD信号线",
                },
                {
                    "name": "macd_hist",
                    "code": "MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[2]",
                    "description": "MACD柱状图",
                },
                {
                    "name": "adx_14",
                    "code": "ADX(high, low, close, timeperiod=14)",
                    "description": "趋势强度指标",
                },
                {
                    "name": "cci_20",
                    "code": "CCI(high, low, close, timeperiod=20)",
                    "description": "通道突破信号",
                },
                {
                    "name": "roc_10",
                    "code": "(close - close.shift(10)) / close.shift(10)",
                    "description": "10日变化率",
                },
            ],
            "波动率风险": [
                {
                    "name": "atr_14",
                    "code": "ATR(high, low, close, timeperiod=14)",
                    "description": "平均真实波幅",
                },
                {
                    "name": "atr_norm",
                    "code": "ATR(high, low, close, timeperiod=14) / close",
                    "description": "波动率相对价格水平",
                },
                {
                    "name": "volatility_10",
                    "code": "np.log(close / close.shift(1)).rolling(window=10).std()",
                    "description": "近10日收益率标准差",
                },
                {
                    "name": "bollinger_bandwidth",
                    "code": "(BBANDS(close, timeperiod=20)[0] - BBANDS(close, timeperiod=20)[2]) / BBANDS(close, timeperiod=20)[1]",
                    "description": "布林带宽度",
                },
                {
                    "name": "bollinger_position",
                    "code": "(close - BBANDS(close, timeperiod=20)[2]) / (BBANDS(close, timeperiod=20)[0] - BBANDS(close, timeperiod=20)[2])",
                    "description": "价格在布林带中的相对位置",
                },
            ],
            "成交量资金流": [
                {
                    "name": "volume_ma_ratio",
                    "code": "volume / SMA(volume, timeperiod=10)",
                    "description": "当日量能vs近期均量",
                },
                {
                    "name": "obv",
                    "code": "OBV(close, volume)",
                    "description": "累积量能趋势",
                },
                {
                    "name": "obv_slope",
                    "code": "OBV(close, volume) - OBV(close, volume).shift(5)",
                    "description": "OBV近5日斜率",
                },
            ],
            "结构模式": [
                {
                    "name": "is_bullish_candle",
                    "code": "(close > open).astype(int)",
                    "description": "是否阳线",
                },
                {
                    "name": "regime_volatility",
                    "code": "(np.log(close / close.shift(1)).rolling(window=20).std() > np.log(close / close.shift(1)).rolling(window=20).std().shift(1).expanding().max()).astype(int)",
                    "description": "高波动regime标记（当前波动大于历史最大值）",
                },
                {
                    "name": "regime_trend",
                    "code": "((ADX(high, low, close, timeperiod=14) > 25) & (SMA(close, timeperiod=20) > SMA(close, timeperiod=60))).astype(int)",
                    "description": "趋势市标记",
                },
            ],
            "动量加速度": [
                {
                    "name": "momentum_20",
                    "code": "close / close.shift(20) - 1",
                    "description": "20日动量（20日收益率）",
                },
                {
                    "name": "momentum_60",
                    "code": "close / close.shift(60) - 1",
                    "description": "60日动量（60日收益率）",
                },
                {
                    "name": "momentum_acceleration",
                    "code": "(close / close.shift(10) - 1) - (close.shift(10) / close.shift(20) - 1)",
                    "description": "动量加速度（近期动量减去前期动量）",
                },
                {
                    "name": "price_momentum_strength",
                    "code": "(SMA(close, timeperiod=5) / SMA(close, timeperiod=20) - 1) * 100",
                    "description": "价格动量强度（短期均线相对长期均线的百分比）",
                },
            ],
            "反转信号": [
                {
                    "name": "reversal_5",
                    "code": "-(close / close.shift(5) - 1)",
                    "description": "5日反转因子（负收益率，用于捕捉短期反转）",
                },
                {
                    "name": "reversal_10",
                    "code": "-(close / close.shift(10) - 1)",
                    "description": "10日反转因子",
                },
                {
                    "name": "deviation_from_ma20",
                    "code": "(close - SMA(close, timeperiod=20)) / SMA(close, timeperiod=20)",
                    "description": "价格偏离20日均线的程度",
                },
                {
                    "name": "deviation_from_ma60",
                    "code": "(close - SMA(close, timeperiod=60)) / SMA(close, timeperiod=60)",
                    "description": "价格偏离60日均线的程度",
                },
                {
                    "name": "stochastic_k",
                    "code": "(close - LLV(low, 14)) / (HHV(high, 14) - LLV(low, 14)) * 100",
                    "description": "随机指标K值（衡量价格在近期区间的相对位置）",
                },
                {
                    "name": "stochastic_d",
                    "code": "SMA((close - LLV(low, 14)) / (HHV(high, 14) - LLV(low, 14)) * 100, timeperiod=3)",
                    "description": "随机指标D值（K值的3日平滑）",
                },
            ],
            "技术形态": [
                {
                    "name": "three_rising_candles",
                    "code": "EVERY(close > open, 3).astype(int)",
                    "description": "三连阳形态（连续3日收阳）",
                },
                {
                    "name": "three_falling_candles",
                    "code": "EVERY(close < open, 3).astype(int)",
                    "description": "三连阴形态（连续3日收阴）",
                },
                {
                    "name": "golden_cross",
                    "code": "CROSS(SMA(close, timeperiod=5), SMA(close, timeperiod=20)).astype(int)",
                    "description": "金叉信号（5日均线上穿20日均线）",
                },
                {
                    "name": "death_cross",
                    "code": "CROSS(SMA(close, timeperiod=20), SMA(close, timeperiod=5)).astype(int)",
                    "description": "死叉信号（5日均线下穿20日均线）",
                },
                {
                    "name": "new_high_20",
                    "code": "(close >= HHV(high, 20)).astype(int)",
                    "description": "触及20日新高",
                },
                {
                    "name": "new_low_20",
                    "code": "(close <= LLV(low, 20)).astype(int)",
                    "description": "触及20日新低",
                },
                {
                    "name": "gap_up",
                    "code": "(low > REF(high, 1)).astype(int)",
                    "description": "向上跳空（今日最低价大于昨日最高价）",
                },
                {
                    "name": "gap_down",
                    "code": "(high < REF(low, 1)).astype(int)",
                    "description": "向下跳空（今日最高价小于昨日最低价）",
                },
            ],
            "市场情绪": [
                {
                    "name": "price_change_1",
                    "code": "(close - close.shift(1)) / close.shift(1)",
                    "description": "1日涨跌幅",
                },
                {
                    "name": "price_change_5",
                    "code": "(close - close.shift(5)) / close.shift(5)",
                    "description": "5日涨跌幅",
                },
                {
                    "name": "volatility_change",
                    "code": "np.log(close / close.shift(1)).rolling(window=10).std() - np.log(close / close.shift(1)).rolling(window=10).std().shift(5)",
                    "description": "波动率变化（当前10日波动率减去5日前波动率）",
                },
                {
                    "name": "volume_surge",
                    "code": "CROSS(volume, SMA(volume, timeperiod=20) * 1.5).astype(int)",
                    "description": "放量信号（成交量突破20日均量的1.5倍）",
                },
                {
                    "name": "volume_shrink",
                    "code": "CROSS(SMA(volume, timeperiod=20) * 0.7, volume).astype(int)",
                    "description": "缩量信号（成交量低于20日均量的0.7倍）",
                },
            ],
            "风险指标": [
                {
                    "name": "downside_risk",
                    "code": "np.log(close / close.shift(1)).clip(upper=0).rolling(window=20).std()",
                    "description": "下行风险（仅计算负收益的标准差）",
                },
                {
                    "name": "skewness_20",
                    "code": "np.log(close / close.shift(1)).rolling(window=20).skew()",
                    "description": "20日收益率偏度（衡量分布不对称性）",
                },
                {
                    "name": "kurtosis_20",
                    "code": "np.log(close / close.shift(1)).rolling(window=20).kurt()",
                    "description": "20日收益率峰度（衡量尾部风险）",
                },
                {
                    "name": "max_drawdown_20",
                    "code": "MAXDRAWDOWN(close, 20)",
                    "description": "20日最大回撤",
                },
                {
                    "name": "var_95_20",
                    "code": "np.log(close / close.shift(1)).rolling(window=20).quantile(0.05)",
                    "description": "20日95% VaR（在险价值）",
                },
            ],
            "均线系统": [
                {
                    "name": "ma5",
                    "code": "SMA(close, timeperiod=5)",
                    "description": "5日均线",
                },
                {
                    "name": "ma10",
                    "code": "SMA(close, timeperiod=10)",
                    "description": "10日均线",
                },
                {
                    "name": "ma20",
                    "code": "SMA(close, timeperiod=20)",
                    "description": "20日均线",
                },
                {
                    "name": "ma60",
                    "code": "SMA(close, timeperiod=60)",
                    "description": "60日均线",
                },
                {
                    "name": "ma120",
                    "code": "SMA(close, timeperiod=120)",
                    "description": "120日均线",
                },
                {
                    "name": "ema12",
                    "code": "EMA(close, timeperiod=12)",
                    "description": "12日指数移动平均",
                },
                {
                    "name": "ema26",
                    "code": "EMA(close, timeperiod=26)",
                    "description": "26日指数移动平均",
                },
                {
                    "name": "ma_bias_5_20",
                    "code": "(SMA(close, timeperiod=5) - SMA(close, timeperiod=20)) / SMA(close, timeperiod=20)",
                    "description": "5日均线乖离率（相对20日均线）",
                },
                {
                    "name": "ma_bias_10_60",
                    "code": "(SMA(close, timeperiod=10) - SMA(close, timeperiod=60)) / SMA(close, timeperiod=60)",
                    "description": "10日均线乖离率（相对60日均线）",
                },
                {
                    "name": "ma_multi_align",
                    "code": "((SMA(close, timeperiod=5) > SMA(close, timeperiod=10)).astype(int) + (SMA(close, timeperiod=10) > SMA(close, timeperiod=20)).astype(int) + (SMA(close, timeperiod=20) > SMA(close, timeperiod=60)).astype(int))",
                    "description": "均线多头排列得分（短中长期均线的多头排列程度）",
                },
            ],
            "价格位置": [
                {
                    "name": "percentile_20",
                    "code": "RANGEPOS(close, 20)",
                    "description": "20日价格分位数（当前价格在20日区间中的位置）",
                },
                {
                    "name": "percentile_60",
                    "code": "RANGEPOS(close, 60)",
                    "description": "60日价格分位数（当前价格在60日区间中的位置）",
                },
                {
                    "name": "distance_to_high_20",
                    "code": "(HHV(high, 20) - close) / HHV(high, 20)",
                    "description": "距离20日高点的幅度",
                },
                {
                    "name": "distance_to_low_20",
                    "code": "(close - LLV(low, 20)) / LLV(low, 20)",
                    "description": "距离20日低点的幅度",
                },
                {
                    "name": "price_range_ratio_20",
                    "code": "(close - LLV(low, 20)) / (HHV(high, 20) - LLV(low, 20))",
                    "description": "价格在20日高低区间的相对位置",
                },
            ],
            "资金流动": [
                {
                    "name": "force_index",
                    "code": "(close - close.shift(1)) * volume",
                    "description": "强力指数（价格变化方向与成交量的结合）",
                },
                {
                    "name": "force_index_ma",
                    "code": "SMA((close - close.shift(1)) * volume, timeperiod=13)",
                    "description": "13日强力指数均值",
                },
                {
                    "name": "money_flow",
                    "code": "IF(close > open, (close + open + high + low) / 4 * volume, -(close + open + high + low) / 4 * volume)",
                    "description": "资金流（阳线为正，阴线为负）",
                },
                {
                    "name": "money_flow_ma",
                    "code": "SMA(IF(close > open, (close + open + high + low) / 4 * volume, -(close + open + high + low) / 4 * volume), timeperiod=5)",
                    "description": "5日资金流均值",
                },
                {
                    "name": "vwma_20",
                    "code": "SUM(close * volume, 20) / SUM(volume, 20)",
                    "description": "20日成交量加权均线",
                },
                {
                    "name": "price_vwma_ratio",
                    "code": "close / (SUM(close * volume, 20) / SUM(volume, 20))",
                    "description": "价格相对VWMA的位置",
                },
            ],
        }

    def get_all_factors(self) -> List[Dict]:
        """获取所有因子"""
        db = get_db_session()
        repo = FactorRepository(db)
        factors = repo.get_all(active_only=True)
        db.close()
        return [f.to_dict() for f in factors]

    def get_factor_stats(self) -> Dict:
        """获取因子统计信息"""
        db = get_db_session()
        repo = FactorRepository(db)

        # 获取缓存统计
        from backend.services.cache_service import cache_service
        from backend.services.strategy_registry import strategy_registry
        cache_stats = cache_service.get_stats()
        stock_cache_count = cache_stats.get("total_count", 0)

        # 检查AKShare健康状态
        akshare_healthy = True
        try:
            import akshare as ak
            # 使用用户指定的接口验证连接
            stock_zh_a_daily_qfq_df = ak.stock_zh_a_daily(
                symbol="sz000001",
                start_date="20230903",
                end_date="20231027",
                adjust="qfq"
            )
        except Exception:
            akshare_healthy = False

        stats = {
            "preset_count": repo.get_preset_count(),
            "user_count": repo.get_user_count(),
            "total_count": repo.get_preset_count() + repo.get_user_count(),
            "strategy_count": len(strategy_registry.list_strategies()),
            "stock_cache_count": stock_cache_count,
            "akshare_healthy": akshare_healthy,
        }
        db.close()
        return stats

    def cleanup_legacy_generated_factors(self) -> Dict[str, Any]:
        """清理历史自动生成的旧版 Python 包装因子。"""
        db = get_db_session()
        repo = FactorRepository(db)
        summary = {
            "scanned": 0,
            "migrated": 0,
            "deleted": 0,
            "skipped": 0,
            "migrated_names": [],
            "deleted_names": [],
        }

        try:
            for factor in repo.get_all(source="user", active_only=False):
                summary["scanned"] += 1

                if not self._is_legacy_generated_python_factor(factor):
                    summary["skipped"] += 1
                    continue

                migrated_expression = self._extract_generated_expression(factor.code)
                if migrated_expression:
                    factor.code = migrated_expression
                    factor.formula_type = infer_formula_type(migrated_expression)
                    repo.update(factor)
                    summary["migrated"] += 1
                    summary["migrated_names"].append(factor.name)
                    continue

                repo.delete(factor.id)
                summary["deleted"] += 1
                summary["deleted_names"].append(factor.name)
        finally:
            db.close()

        return summary

    def _is_legacy_generated_python_factor(self, factor: FactorModel) -> bool:
        code = (factor.code or "").strip()
        if factor.source != "user":
            return False

        if factor.category in {"遗传挖掘", "组合因子"}:
            if "def calculate_factor" in code:
                return True
            if re.fullmatch(r"factor_\d+", code):
                return True

        if not code.startswith("def calculate_factor"):
            return False

        legacy_markers = (
            "遗传算法挖掘因子",
            "组合因子 - ",
            "通过遗传算法挖掘的因子",
        )
        return any(marker in code or marker in (factor.description or "") for marker in legacy_markers)

    def _extract_generated_expression(self, code: str) -> Optional[str]:
        match = re.search(r"^\s*表达式:\s*(.+?)\s*$", code, flags=re.MULTILINE)
        if not match:
            return None
        expression = match.group(1).strip()
        return expression or None

    def repair_stored_factor_codes(self) -> Dict[str, Any]:
        """扫描并修复库中历史坏因子的持久化代码内容。"""
        db = get_db_session()
        repo = FactorRepository(db)
        summary = {
            "scanned": 0,
            "repaired": 0,
            "unchanged": 0,
            "failed": 0,
            "repaired_items": [],
            "failed_items": [],
        }

        try:
            for factor in repo.get_all(source=None, active_only=False):
                summary["scanned"] += 1
                previous_formula_type = factor.formula_type
                try:
                    repaired = self._repair_factor_storage_record(factor)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("修复历史因子失败: id=%s name=%s", factor.id, factor.name)
                    summary["failed"] += 1
                    summary["failed_items"].append(
                        {
                            "id": factor.id,
                            "name": factor.name,
                            "category": factor.category,
                            "formula_type": factor.formula_type,
                            "error": str(exc),
                        }
                    )
                    continue

                if repaired is None:
                    summary["unchanged"] += 1
                    continue

                factor.code = repaired["code"]
                factor.formula_type = repaired["formula_type"]
                repo.update(factor)
                summary["repaired"] += 1
                summary["repaired_items"].append(
                    {
                        "id": factor.id,
                        "name": factor.name,
                        "category": factor.category,
                        "old_formula_type": previous_formula_type,
                        "new_formula_type": repaired["formula_type"],
                        "reasons": repaired["reasons"],
                    }
                )
        finally:
            db.close()

        return summary

    def _repair_factor_storage_record(self, factor: FactorModel) -> Optional[Dict[str, Any]]:
        """修复单个因子的库内存储内容，保持合法麦语言程序不被压缩。"""
        original_code = (factor.code or "").strip()
        original_formula_type = (factor.formula_type or "").strip() or "auto"
        repaired_code = original_code
        repaired_formula_type = original_formula_type
        reasons: List[str] = []

        if self._is_legacy_generated_python_factor(factor):
            migrated_expression = self._extract_generated_expression(original_code)
            if migrated_expression:
                repaired_code = normalize_formula_code(migrated_expression, formula_type="python")
                repaired_formula_type = normalize_formula_type("auto", repaired_code)
                reasons.append("legacy_generated_expression")

        normalized_code = self._prepare_factor_code_for_storage(
            repaired_code,
            formula_type=repaired_formula_type,
        )
        if normalized_code != repaired_code:
            repaired_code = normalized_code
            reasons.append("normalized_storage_code")

        resolved_formula_type = self._resolve_stored_formula_type(
            repaired_code,
            formula_type=repaired_formula_type,
        )
        if resolved_formula_type != repaired_formula_type:
            repaired_formula_type = resolved_formula_type
            reasons.append("normalized_formula_type")

        is_valid, message = self.validate_factor_code(repaired_code, formula_type=repaired_formula_type)
        if not is_valid:
            raise ValueError(message)

        if repaired_code == original_code and repaired_formula_type == original_formula_type:
            return None

        return {
            "code": repaired_code,
            "formula_type": repaired_formula_type,
            "reasons": reasons,
        }

    def create_factor(
        self, name: str, code: str, description: str = "",
        category: str = "自定义", formula_type: str = "auto"
    ) -> Dict:
        """创建用户自定义因子"""
        db = get_db_session()
        repo = FactorRepository(db)

        # 检查名称是否已存在
        existing_factor = repo.get_by_name(name, include_inactive=True)

        if existing_factor:
            # 如果因子已存在
            if existing_factor.is_active == 1:
                # 活跃因子，不能创建
                db.close()
                raise ValueError(f"因子名称 '{name}' 已存在")
            else:
                # 已软删除的因子，硬删除旧记录后创建新记录
                logger.info(f"因子 '{name}' 已存在但已删除，将替换为新记录")
                from sqlalchemy import delete
                stmt = delete(FactorModel).where(FactorModel.id == existing_factor.id)
                db.execute(stmt)
                db.commit()

        stored_code = self._prepare_factor_code_for_storage(code, formula_type=formula_type)
        resolved_type = self._resolve_stored_formula_type(stored_code, formula_type=formula_type)
        self._ensure_factor_code_is_valid(stored_code, resolved_type)

        factor = FactorModel(
            name=name,
            code=stored_code,
            formula_type=resolved_type,
            description=description,
            source="user",
            category=category,
            is_active=1,
        )
        result = repo.create(factor)
        db.close()
        return result.to_dict()

    def update_factor(
        self, factor_id: int, name: str = None, code: str = None, description: str = None,
        category: str = None, formula_type: str = None, create_version: bool = True, change_reason: str = ""
    ) -> Dict:
        """
        更新因子

        Args:
            factor_id: 因子ID
            name: 新名称（可选）
            code: 新代码（可选）
            description: 新描述（可选）
            create_version: 是否创建版本快照（默认True）
            change_reason: 变更原因（可选）

        Returns:
            更新后的因子信息
        """
        db = get_db_session()
        repo = FactorRepository(db)
        factor = repo.get_by_id(factor_id)

        if not factor:
            db.close()
            raise ValueError(f"因子ID {factor_id} 不存在")

        if factor.source == "preset" and (name or code):
            db.close()
            raise ValueError("预置因子的名称和代码不能修改")

        if name and name != factor.name:
            existing_factor = repo.get_by_name(name, include_inactive=True)
            if existing_factor and existing_factor.id != factor_id:
                db.close()
                raise ValueError(f"因子名称 '{name}' 已存在")

        # 如果需要创建版本且代码有变化，先保存版本
        if create_version and code and code != factor.code:
            try:
                factor_version_service.create_version(
                    factor_id=factor_id,
                    code=factor.code,
                    description=factor.description,
                    change_reason=change_reason or "更新前自动保存",
                    auto_increment=True,
                )
            except Exception as e:
                logger.warning(f"创建版本快照失败: {e}")

        proposed_code = code if code is not None else factor.code
        stored_code = self._prepare_factor_code_for_storage(proposed_code, formula_type=formula_type)
        if formula_type is not None:
            proposed_formula_type = self._resolve_stored_formula_type(stored_code, formula_type=formula_type)
        elif code is not None:
            proposed_formula_type = self._resolve_stored_formula_type(stored_code, formula_type=None)
        else:
            proposed_formula_type = factor.formula_type or self._resolve_stored_formula_type(stored_code, formula_type=None)

        if code is not None or formula_type is not None:
            self._ensure_factor_code_is_valid(stored_code, proposed_formula_type)

        # 更新因子
        if name:
            factor.name = name
        if code:
            factor.code = stored_code
        if description is not None:
            factor.description = description
        if category:
            factor.category = category
        if formula_type:
            factor.formula_type = self._resolve_stored_formula_type(
                stored_code if code is not None else factor.code,
                formula_type=formula_type,
            )
        elif code:
            factor.formula_type = self._resolve_stored_formula_type(stored_code, formula_type=None)

        result = repo.update(factor)
        db.close()
        return result.to_dict()

    def get_factor_versions(self, factor_id: int) -> List[Dict]:
        """获取因子的版本历史"""
        return factor_version_service.get_version_history(factor_id)

    def rollback_factor_version(self, factor_id: int, version_code: str) -> bool:
        """回滚因子到指定版本"""
        return factor_version_service.rollback_to_version(factor_id, version_code)

    def delete_factor(self, factor_id: int) -> bool:
        """删除因子"""
        db = get_db_session()
        repo = FactorRepository(db)
        try:
            result = repo.delete(factor_id)
            db.close()
            return result
        except ValueError as e:
            db.close()
            raise e

    def validate_factor_code(self, code: str, formula_type: Optional[str] = None) -> tuple[bool, str]:
        """验证因子代码"""
        # 使用logging记录调试信息（可通过配置关闭）
        logger.debug(f"Validating factor code, length: {len(code)}")

        # 创建更真实的测试数据（避免全相同值）
        import numpy as np
        test_df = pd.DataFrame({
            "open": np.linspace(10.0, 11.0, 100),
            "high": np.linspace(11.0, 12.0, 100),
            "low": np.linspace(9.0, 10.0, 100),
            "close": np.linspace(10.5, 11.5, 100),
            "volume": np.linspace(1000000, 1100000, 100),
            "amount": np.linspace(10000000, 11000000, 100),
        })
        test_df.index = pd.date_range("2024-01-01", periods=len(test_df), freq="D")

        try:
            prepared_code = self._prepare_factor_code(code, formula_type=formula_type)
            resolved_type = normalize_formula_type(formula_type, prepared_code)
            calculator = FactorCalculator()
            result = calculator.calculate(test_df, prepared_code, resolved_type)

            # 检查结果
            if result is None or len(result) == 0:
                return False, "代码未返回任何结果"

            # 检查是否包含 NaN
            if result.isna().all():
                return False, "计算结果全部为NaN，请检查公式"

            # 检查是否包含 Inf
            if np.isinf(result).any():
                return False, "计算结果包含无穷大值，请检查公式"

            # 检查是否所有值都相同（可能不是有效的因子）
            # 先排除 NaN 值再检查
            valid_result = result.dropna()
            if len(valid_result) > 0 and valid_result.nunique() == 1:
                # 对于常量值，我们只警告但仍然允许通过
                logger.warning(f"Factor result has only one unique value: {valid_result.iloc[0]}")
                # 不返回错误，只记录警告，因为有些有效的因子可能确实是常量

            return True, f"验证通过（类型: {resolved_type}）"

        except ValueError as e:
            # 捕获因子计算错误
            logger.debug(f"Factor code validation failed: {str(e)}", exc_info=True)
            resolved_type = normalize_formula_type(formula_type, (code or "").strip())
            return False, self._friendly_validation_error(str(e), resolved_type, (code or "").strip())
        except Exception as e:
            # 捕获其他错误（如 NameError、SyntaxError 等）
            logger.debug(f"Factor code validation failed: {str(e)}", exc_info=True)
            # 提供更友好的错误信息
            error_msg = str(e)
            resolved_type = normalize_formula_type(formula_type, (code or "").strip())

            # 检查常见错误模式
            if "is not defined" in error_msg:
                # 提取未定义的变量名
                import re
                match = re.search(r"name '(\w+)' is not defined", error_msg)
                if match:
                    undefined_name = match.group(1)
                    # 提供友好的建议
                    suggestions = []

                    # 检查是否是常见变量名的拼写错误
                    common_vars = {'close', 'open', 'high', 'low', 'volume', 'amount', 'np', 'pd'}
                    for var in common_vars:
                        if undefined_name.lower() == var.lower() or undefined_name.lower() in var:
                            suggestions.append(f"变量名：{var}")

                    # 检查是否是常见函数的拼写错误
                    common_funcs = {
                        # TALib 函数
                        'SMA': 'SMA (简单移动平均)',
                        'MA': 'SMA 或 MA (简单移动平均)',
                        'EMA': 'EMA (指数移动平均)',
                        'RSI': 'RSI (相对强弱指标)',
                        'MACD': 'MACD (移动平均收敛散度)',
                        'ATR': 'ATR (平均真实波幅)',
                        'BBANDS': 'BBANDS (布林带)',
                        'OBV': 'OBV (能量潮)',
                        # 麦语言函数
                        'REF': 'REF (引用n日前的值)',
                        'HHV': 'HHV (n日内最高值)',
                        'LLV': 'LLV (n日内最低值)',
                        'SUM': 'SUM (n日总和)',
                        'AVE': 'AVE (n日平均值)',
                        'STD': 'STD (n日标准差)',
                        'COUNT': 'COUNT (n日内满足条件的次数)',
                        'EVERY': 'EVERY (n日内是否一直满足条件)',
                        'EXIST': 'EXIST (n日内是否存在满足条件)',
                        'CROSS': 'CROSS (金叉：x上穿y)',
                        'LONGCROSS': 'LONGCROSS (n日内金叉)',
                        'UP': 'UP (上涨：今日大于n日前)',
                        'DOWN': 'DOWN (下跌：今日小于n日前)',
                        'IF': 'IF (条件选择函数)',
                        'BETWEEN': 'BETWEEN (区间判断)',
                        'MAX': 'MAX (最大值)',
                        'MIN': 'MIN (最小值)',
                        'BARSLAST': 'BARSLAST (上一次满足条件到当前的周期数)',
                        'CONST': 'CONST (常量序列)'
                    }

                    for func, desc in common_funcs.items():
                        if undefined_name.upper() == func or func in undefined_name.upper():
                            suggestions.append(f"函数：{desc}")

                    if suggestions:
                        return False, f"未定义的名称 '{undefined_name}'。您是否想使用：{', '.join(suggestions)}？"

                    return False, f"未定义的名称 '{undefined_name}'，请检查拼写。常见变量名：close, open, high, low, volume"

            return False, self._friendly_validation_error(f"验证失败: {error_msg}", resolved_type, (code or "").strip())

    def _friendly_validation_error(self, error_message: str, resolved_type: str, code: str) -> str:
        if resolved_type == "python":
            if "Python 指标不支持 AST 节点: Lambda" in error_message:
                return "当前 Python 因子不支持 lambda 表达式，请改写为系统函数或普通表达式"
            if "RollingWindow 不允许访问属性 apply" in error_message:
                return "当前 Python 因子不支持 rolling(...).apply(...)，请改写为系统函数或显式表达式写法"
            if "Python 指标不支持 AST 节点: ListComp" in error_message or "Python 指标不支持 AST 节点: DictComp" in error_message or "Python 指标不支持 AST 节点: SetComp" in error_message or "Python 指标不支持 AST 节点: GeneratorExp" in error_message:
                return "当前 Python 因子不支持推导式或生成器表达式，请改写为普通表达式"
            if "Python 指标不支持语句类型: For" in error_message or "Python 指标不支持语句类型: While" in error_message:
                return "当前 Python 因子不支持 for / while 循环，请改写为向量化表达式"
            if "Python 指标不支持语句类型: With" in error_message:
                return "当前 Python 因子不支持 with 语句"
            if "Python 指标暂不支持链式比较" in error_message:
                return "当前 Python 因子不支持链式比较，请拆成两个独立比较后再组合"
            if "Python 指标不允许使用 from ... import ..." in error_message:
                return "当前 Python 因子不支持 from ... import ...，如需导入仅可使用 import pandas / import numpy"
            if "Python 指标仅支持简单变量赋值" in error_message or "Python 指标仅支持简单变量增强赋值" in error_message:
                return "当前 Python 因子仅支持简单变量赋值，不支持解构赋值或复杂目标赋值"
            if "Series 不允许访问属性" in error_message or "RollingWindow 不允许访问属性" in error_message or "ExpandingWindow 不允许访问属性" in error_message:
                return (
                    "当前 Python 因子使用了未开放的方法。支持的序列方法主要包括 "
                    "shift/diff/pct_change/abs/round/fillna/clip/astype/mean/std/max/min/"
                    "median/sum/skew/kurt/quantile/rank/rolling/expanding/ffill/bfill/replace"
                )

        if resolved_type == "mylanguage":
            if "." in code and ("麦语言词法错误" in error_message or "麦语言语法错误" in error_message):
                return "当前麦语言不支持对象点号语法，请改用 MA(CLOSE, 20)、REF(CLOSE, 1) 这类函数式写法"

        return error_message

    def _ensure_factor_code_is_valid(self, code: str, formula_type: Optional[str] = None) -> str:
        """在写库前强制校验因子代码。"""
        prepared_code = self._prepare_factor_code(code, formula_type=formula_type)
        resolved_type = normalize_formula_type(formula_type, prepared_code)
        is_valid, message = self.validate_factor_code(prepared_code, formula_type=resolved_type)
        if not is_valid:
            raise ValueError(f"因子代码校验失败: {message}")
        return resolved_type

    def _prepare_factor_code(self, code: str, formula_type: Optional[str] = None) -> str:
        """执行和校验前统一规范化公式代码。"""
        stripped = (code or "").strip()
        if not stripped:
            return stripped

        resolved_type = normalize_formula_type(formula_type, stripped)
        if resolved_type == "python":
            prepared_code = normalize_formula_code(stripped, formula_type="python")
        else:
            prepared_code = stripped

        if prepared_code != (code or "").strip():
            logger.info("因子代码已规范化: %s -> %s", (code or "").strip(), prepared_code)
        return prepared_code

    def _prepare_factor_code_for_storage(self, code: str, formula_type: Optional[str] = None) -> str:
        """写库前仅修复历史坏模式，不压缩合法麦语言程序。"""
        stripped = (code or "").strip()
        if not stripped:
            return stripped

        resolved_type = self._resolve_stored_formula_type(stripped, formula_type=formula_type)
        if resolved_type == "python":
            prepared_code = normalize_formula_code(stripped, formula_type="python")
        else:
            prepared_code = stripped

        if prepared_code != stripped:
            logger.info("因子代码已修复入库格式: %s -> %s", stripped, prepared_code)
        return prepared_code

    def _resolve_stored_formula_type(self, code: str, formula_type: Optional[str] = None) -> str:
        """推断写库时应持久化的公式类型。"""
        stripped = (code or "").strip()
        if not stripped:
            return normalize_formula_type(formula_type, stripped)

        declared_type = (formula_type or "").strip().lower()
        if declared_type in {"mylanguage", "python"}:
            return declared_type
        return infer_formula_type(stripped)

    def calculate_factors_for_stock(
        self,
        stock_code: str,
        factor_names: List[str],
        start_date: str,
        end_date: str,
        rolling_window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        为单个股票计算因子

        Args:
            stock_code: 股票代码
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            rolling_window: 滚动标准化窗口大小

        Returns:
            包含因子值的DataFrame
        """
        # 获取股票数据
        df = data_service.get_stock_data(stock_code, start_date, end_date)

        # 获取因子定义
        db = get_db_session()
        repo = FactorRepository(db)
        factors = []
        for name in factor_names:
            factor = repo.get_by_name(name)
            if factor:
                factors.append(factor)
        db.close()

        if not factors:
            raise ValueError("未找到有效的因子")

        # 计算因子
        factor_df = self.calculator.calculate_multiple(df, factors)

        # 滚动标准化
        if rolling_window:
            factor_df = self.calculator.rolling_standardize(factor_df, rolling_window)

        # 添加时间特征
        factor_df = self.calculator.add_time_features(factor_df)

        # 合并原始数据
        result = pd.concat([df, factor_df], axis=1)

        # 最终清理：将所有无穷大值替换为NaN
        for col in result.select_dtypes(include=[np.number]).columns:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)

        return result

    def calculate_factors_for_stocks(
        self,
        stock_codes: List[str],
        factor_names: List[str],
        start_date: str,
        end_date: str,
        rolling_window: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """为多个股票计算因子"""
        results = {}
        for code in stock_codes:
            try:
                result = self.calculate_factors_for_stock(
                    code, factor_names, start_date, end_date, rolling_window
                )
                results[code] = result
            except Exception as e:
                logger.warning(f"为股票 {code} 计算因子失败: {e}")
        return results


# 全局因子服务实例
factor_service = FactorService()
