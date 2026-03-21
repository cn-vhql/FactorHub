"""
因子服务模块 - 因子计算与管理
"""
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import logging

from backend.core.database import get_db_session
from backend.core.settings import settings
from backend.models.factor import FactorModel
from backend.repositories.factor_repository import FactorRepository
from backend.services.data_service import data_service
from backend.services.factor_version_service import factor_version_service

# 配置日志
logger = logging.getLogger(__name__)


class FactorCalculator:
    """因子计算器 - 执行因子计算逻辑"""

    def __init__(self):
        # TALib 函数（注意：SMA 在 mylanguage_funcs 中定义，支持命名参数）
        self.talib_funcs = {
            # SMA 不在这里定义，改用 mylanguage_funcs 中的版本
            "EMA": talib.EMA,
            "RSI": talib.RSI,
            "MACD": talib.MACD,
            "ADX": talib.ADX,
            "CCI": talib.CCI,
            "ATR": talib.ATR,
            "BBANDS": talib.BBANDS,
            "OBV": talib.OBV,
            "STOCH": talib.STOCH,
            "STOCHRSI": talib.STOCHRSI,
            "WILLR": talib.WILLR,
            "KAMA": talib.KAMA,
            "ROC": talib.ROC,
            "MOM": talib.MOM,
        }

        # 麦语言（MyLanguage）函数
        self.mylanguage_funcs = self._create_mylanguage_funcs()

    def _create_mylanguage_funcs(self):
        """创建麦语言兼容函数"""

        # SMA 包装函数，支持命名参数
        def SMA(series, timeperiod=30, **kwargs):
            """简单移动平均"""
            import talib
            if isinstance(series, pd.Series):
                result = talib.SMA(series.values, timeperiod=timeperiod, **kwargs)
                return pd.Series(result, index=series.index)
            return talib.SMA(series, timeperiod=timeperiod, **kwargs)

        # MA 作为 SMA 的别名
        def MA(series, timeperiod=30, **kwargs):
            """移动平均（SMA别名）"""
            return SMA(series, timeperiod=timeperiod, **kwargs)

        def REF(series, n=1):
            """引用n日前的值"""
            if isinstance(series, pd.Series):
                return series.shift(n)
            return pd.Series(series).shift(n)

        def HHV(series, n=5):
            """n日内最高值"""
            if isinstance(series, pd.Series):
                return series.rolling(window=n, min_periods=1).max()
            return pd.Series(series).rolling(window=n, min_periods=1).max()

        def LLV(series, n=5):
            """n日内最低值"""
            if isinstance(series, pd.Series):
                return series.rolling(window=n, min_periods=1).min()
            return pd.Series(series).rolling(window=n, min_periods=1).min()

        def SUM(series, n=5):
            """n日总和"""
            if isinstance(series, pd.Series):
                return series.rolling(window=n, min_periods=1).sum()
            return pd.Series(series).rolling(window=n, min_periods=1).sum()

        def AVE(series, n=5):
            """n日平均值"""
            if isinstance(series, pd.Series):
                return series.rolling(window=n, min_periods=1).mean()
            return pd.Series(series).rolling(window=n, min_periods=1).mean()

        def STD(series, n=5):
            """n日标准差"""
            if isinstance(series, pd.Series):
                return series.rolling(window=n, min_periods=1).std()
            return pd.Series(series).rolling(window=n, min_periods=1).std()

        def COUNT(condition, n=5):
            """n日内满足条件的次数"""
            if isinstance(condition, pd.Series):
                return condition.rolling(window=n, min_periods=1).sum()
            return pd.Series(condition).rolling(window=n, min_periods=1).sum()

        def EVERY(condition, n=5):
            """n日内是否一直满足条件"""
            if isinstance(condition, pd.Series):
                return condition.rolling(window=n, min_periods=1).apply(lambda x: x.all(), raw=False)
            return pd.Series(condition).rolling(window=n, min_periods=1).apply(lambda x: x.all())

        def EXIST(condition, n=5):
            """n日内是否存在满足条件"""
            if isinstance(condition, pd.Series):
                return condition.rolling(window=n, min_periods=1).apply(lambda x: x.any(), raw=False)
            return pd.Series(condition).rolling(window=n, min_periods=1).apply(lambda x: x.any())

        def CROSS(x, y):
            """金叉：x上穿y"""
            if isinstance(x, pd.Series) and isinstance(y, pd.Series):
                return (x > y) & (x.shift(1) <= y.shift(1))
            return pd.Series(x > y) & pd.Series(x).shift(1) <= pd.Series(y).shift(1)

        def LONGCROSS(x, y, n=5):
            """n日内金叉"""
            if isinstance(x, pd.Series) and isinstance(y, pd.Series):
                cross = (x > y) & (x.shift(1) <= y.shift(1))
                return cross.rolling(window=n, min_periods=1).apply(lambda z: z.any(), raw=False)
            x_series = pd.Series(x)
            y_series = pd.Series(y)
            cross = (x_series > y_series) & (x_series.shift(1) <= y_series.shift(1))
            return cross.rolling(window=n, min_periods=1).apply(lambda z: z.any())

        def UP(series, n=1):
            """上涨：今日大于n日前"""
            if isinstance(series, pd.Series):
                return series > series.shift(n)
            return pd.Series(series) > pd.Series(series).shift(n)

        def DOWN(series, n=1):
            """下跌：今日小于n日前"""
            if isinstance(series, pd.Series):
                return series < series.shift(n)
            return pd.Series(series) < pd.Series(series).shift(n)

        def IF(condition, true_value, false_value=0):
            """条件选择函数"""
            if isinstance(condition, pd.Series):
                result = pd.Series(np.where(condition, true_value, false_value), index=condition.index)
                return result
            return np.where(condition, true_value, false_value)

        def BETWEEN(series, lower, upper):
            """区间判断"""
            if isinstance(series, pd.Series):
                return (series >= lower) & (series <= upper)
            return (pd.Series(series) >= lower) & (pd.Series(series) <= upper)

        def MAX(series1, series2):
            """最大值"""
            if isinstance(series1, pd.Series) and isinstance(series2, pd.Series):
                return series1.combine(series2, max)
            return np.maximum(series1, series2)

        def MIN(series1, series2):
            """最小值"""
            if isinstance(series1, pd.Series) and isinstance(series2, pd.Series):
                return series1.combine(series2, min)
            return np.minimum(series1, series2)

        def BARSLAST(condition):
            """上一次满足条件到当前的周期数"""
            if not isinstance(condition, pd.Series):
                condition = pd.Series(condition)

            result = pd.Series(0, index=condition.index)
            last_true_idx = -1

            for i in range(len(condition)):
                if condition.iloc[i]:
                    last_true_idx = i
                    result.iloc[i] = 0
                elif last_true_idx >= 0:
                    result.iloc[i] = i - last_true_idx
                else:
                    result.iloc[i] = len(condition)

            return result

        def CONST(value, length=100):
            """常量序列"""
            return pd.Series([value] * length)

        return {
            # 移动平均函数
            "SMA": SMA,
            "MA": MA,
            # 引用函数
            "REF": REF,
            # 极值函数
            "HHV": HHV,
            "LLV": LLV,
            # 统计函数
            "SUM": SUM,
            "AVE": AVE,
            "STD": STD,
            "COUNT": COUNT,
            # 逻辑函数
            "EVERY": EVERY,
            "EXIST": EXIST,
            "CROSS": CROSS,
            "LONGCROSS": LONGCROSS,
            "UP": UP,
            "DOWN": DOWN,
            # 条件函数
            "IF": IF,
            "BETWEEN": BETWEEN,
            # 数学函数
            "MAX": MAX,
            "MIN": MIN,
            # 其他函数
            "BARSLAST": BARSLAST,
            "CONST": CONST,
        }

    def calculate(self, df: pd.DataFrame, factor_code: str) -> pd.Series:
        """
        计算单个因子

        Args:
            df: 包含OHLCV数据的DataFrame
            factor_code: 因子计算代码（支持表达式或函数形式）

        Returns:
            因子值的Series
        """
        import pandas as pd

        # 检测是否是函数形式（去除注释和空行后再检查）
        lines = factor_code.strip().split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        is_function = len(code_lines) > 0 and code_lines[0].strip().startswith("def ")

        if is_function:
            # 函数形式：使用 exec 执行函数定义，然后调用函数
            # 提供完整的全局变量，包括 pandas, numpy 和常见函数
            global_vars = {
                "__builtins__": {
                    "__import__": __import__,
                    "abs": abs, "min": min, "max": max, "len": len,
                    "range": range, "float": float, "int": int, "bool": bool,
                    "list": list, "tuple": tuple, "dict": dict, "sum": sum,
                    "any": any, "all": all, "enumerate": enumerate, "zip": zip,
                    "round": round, "pow": pow, "divmod": divmod,
                },
                "pd": pd,
                "np": np,
                **self.talib_funcs,
                **self.mylanguage_funcs,
            }

            local_vars = {}

            try:
                # 执行函数定义
                # 注意：此处在受限环境中执行用户代码
                # global_vars已经限制了可用的内置函数
                exec(factor_code, global_vars, local_vars)

                # 调用函数（函数可能在 global_vars 或 local_vars 中）
                calc_func = local_vars.get("calculate_factor") or global_vars.get("calculate_factor")
                if calc_func is None:
                    raise ValueError("函数代码中未找到 'calculate_factor' 函数定义")

                result = calc_func(df)

                if isinstance(result, pd.DataFrame):
                    # 如果返回DataFrame，取第一列
                    result = result.iloc[:, 0]
                elif not isinstance(result, pd.Series):
                    raise ValueError(f"函数必须返回 pd.Series，实际返回了 {type(result)}")

                return result
            except Exception as e:
                import traceback
                # 记录详细错误信息用于调试
                error_msg = f"因子计算失败: {str(e)}\n"
                error_msg += f"因子代码类型: {'函数' if is_function else '表达式'}\n"
                error_msg += f"代码长度: {len(factor_code)} 字符"
                # 在开发环境可以添加traceback，生产环境使用日志
                import logging
                logger = logging.getLogger(__name__)
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg)
        else:
            # 表达式形式：使用 eval 执行（保持向后兼容）
            local_vars = {
                "df": df,
                "open": df["open"],
                "high": df["high"],
                "low": df["low"],
                "close": df["close"],
                "volume": df["volume"],
                # 麦语言价格别名（大写）
                "C": df["close"],  # CLOSE
                "O": df["open"],   # OPEN
                "H": df["high"],   # HIGH
                "L": df["low"],    # LOW
                "V": df["volume"], # VOLUME
                "CLOSE": df["close"],
                "OPEN": df["open"],
                "HIGH": df["high"],
                "LOW": df["low"],
                "VOL": df["volume"],
                # Python内置函数
                "int": int,
                "float": float,
                "bool": bool,
                "str": str,
                "list": list,
                "tuple": tuple,
                "dict": dict,
                "set": set,
                "len": len,
                "range": range,
                # NumPy
                "np": np,
                **self.talib_funcs,
                **self.mylanguage_funcs,
            }

            try:
                # 安全措施：
                # 1. 限制__builtins__为空字典
                # 2. 只提供预定义的local_vars
                # 3. 使用ast验证代码语法
                import ast
                try:
                    ast.parse(factor_code, mode='eval')
                except SyntaxError:
                    raise ValueError(f"因子表达式语法错误: {factor_code}")

                result = eval(factor_code, {"__builtins__": {}}, local_vars)

                # 处理不同类型的返回结果
                if isinstance(result, pd.DataFrame):
                    # 如果返回DataFrame，取第一列
                    result = result.iloc[:, 0]
                elif isinstance(result, (int, float)):
                    # 如果是标量值，转换为Series
                    result = pd.Series([result] * len(df), index=df.index)
                elif not isinstance(result, pd.Series):
                    # 其他类型尝试转换
                    result = pd.Series(result)

                return result
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"因子表达式计算失败: {factor_code}", exc_info=True)
                raise ValueError(f"因子表达式计算失败: {e}")

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
                factor_values = self.calculate(df, factor.code)
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

        db = get_db_session()
        repo = FactorRepository(db)

        for category, factors in config.items():
            for factor_data in factors:
                # 检查因子是否已存在
                existing = repo.get_by_name(factor_data["name"])
                if existing:
                    continue

                factor = FactorModel(
                    name=factor_data["name"],
                    code=factor_data["code"],
                    description=factor_data.get("description", ""),
                    source="preset",
                    category=category,
                    is_active=1,
                )
                repo.create(factor)

        db.close()

    def _create_default_preset_factors(self) -> None:
        """创建默认预置因子"""
        preset_factors = self._get_default_factors()
        db = get_db_session()
        repo = FactorRepository(db)

        for category, factors in preset_factors.items():
            for factor_data in factors:
                existing = repo.get_by_name(factor_data["name"])
                if existing:
                    continue
                factor = FactorModel(
                    name=factor_data["name"],
                    code=factor_data["code"],
                    description=factor_data.get("description", ""),
                    source="preset",
                    category=category,
                    is_active=1,
                )
                repo.create(factor)

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
                    "code": "close.rolling(window=20).apply(lambda x: (x - x.cummax()).min() / x.cummax().max())",
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
                    "code": "close.rolling(window=20).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))",
                    "description": "20日价格分位数（当前价格在20日区间中的位置）",
                },
                {
                    "name": "percentile_60",
                    "code": "close.rolling(window=60).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))",
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
            "strategy_count": 0,  # 暂时为0，后续可以根据实际情况添加
            "stock_cache_count": stock_cache_count,
            "akshare_healthy": akshare_healthy,
        }
        db.close()
        return stats

    def create_factor(
        self, name: str, code: str, description: str = "",
        category: str = "自定义", formula_type: str = "expression"
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

        factor = FactorModel(
            name=name,
            code=code,
            description=description,
            source="user",
            category=category,
            formula_type=formula_type,
            is_active=1,
        )
        result = repo.create(factor)
        db.close()
        return result.to_dict()

    def update_factor(
        self, factor_id: int, name: str = None, code: str = None, description: str = None,
        category: str = None, create_version: bool = True, change_reason: str = ""
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

        # 更新因子
        if name:
            factor.name = name
        if code:
            factor.code = code
        if description is not None:
            factor.description = description
        if category:
            factor.category = category

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

    def validate_factor_code(self, code: str) -> tuple[bool, str]:
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
        })

        try:
            calculator = FactorCalculator()
            result = calculator.calculate(test_df, code)

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

            return True, "验证通过"

        except ValueError as e:
            # 捕获因子计算错误
            logger.debug(f"Factor code validation failed: {str(e)}", exc_info=True)
            return False, str(e)
        except Exception as e:
            # 捕获其他错误（如 NameError、SyntaxError 等）
            logger.debug(f"Factor code validation failed: {str(e)}", exc_info=True)
            # 提供更友好的错误信息
            error_msg = str(e)

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
                    common_vars = {'close', 'open', 'high', 'low', 'volume', 'np'}
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

            return False, f"验证失败: {error_msg}"

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
