"""
高级统计分析服务
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 只忽略特定类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*invalid value.*")


class StatisticsService:
    """高级统计分析服务"""

    # ==================== 因子有效性检验 ====================

    def t_test_ic(self, ic_series: pd.Series, confidence_level: float = 0.95) -> Dict:
        """
        IC序列的t检验，判断因子是否显著

        Args:
            ic_series: IC序列
            confidence_level: 置信水平

        Returns:
            Dict: t检验结果
        """
        ic_clean = ic_series.dropna()

        if len(ic_clean) == 0:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "mean_ic": 0.0,
                "std_ic": 0.0,
                "confidence_interval": (0.0, 0.0),
            }

        # t检验
        t_stat, p_value = stats.ttest_1samp(ic_clean, 0)

        # 计算置信区间
        alpha = 1 - confidence_level
        df = len(ic_clean) - 1
        se = ic_clean.std() / np.sqrt(len(ic_clean))
        ci = stats.t.interval(confidence_level, df, loc=ic_clean.mean(), scale=se)

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": p_value < alpha,
            "mean_ic": ic_clean.mean(),
            "std_ic": ic_clean.std(),
            "confidence_interval": ci,
        }

    def test_monotonicity(
        self, quantile_returns: Dict[str, pd.Series], alternative: str = "increasing"
    ) -> Dict:
        """
        检验因子分层的单调性

        Args:
            quantile_returns: 各分层收益字典，如 {"Q1": returns1, "Q2": returns2, ...}
            alternative: 备择假设方向，"increasing"或"decreasing"

        Returns:
            Dict: 单调性检验结果
        """
        # 计算各层的平均收益
        layer_means = []
        layer_names = sorted(quantile_returns.keys())

        for name in layer_names:
            returns = quantile_returns[name].dropna()
            if len(returns) > 0:
                layer_means.append(returns.mean())
            else:
                layer_means.append(0.0)

        # Spearman秩相关检验
        n_layers = len(layer_means)
        layer_ranks = np.arange(n_layers)

        if alternative == "increasing":
            # 正相关
            correlation, p_value = stats.spearmanr(layer_ranks, layer_means, alternative="greater")
            expected_direction = "正相关"
        else:
            # 负相关
            correlation, p_value = stats.spearmanr(
                layer_ranks, layer_means, alternative="less"
            )
            expected_direction = "负相关"

        return {
            "correlation": correlation,
            "p_value": p_value,
            "is_monotonic": p_value < 0.05,
            "layer_means": layer_means,
            "layer_names": layer_names,
            "expected_direction": expected_direction,
        }

    def calculate_factor_decay(
        self, df: pd.DataFrame, factor_name: str, max_periods: int = 10
    ) -> Dict:
        """
        计算因子衰减（预测能力随时间的变化）

        Args:
            df: 包含因子和价格数据的DataFrame
            factor_name: 因子名称
            max_periods: 最大的向前期数

        Returns:
            Dict: 各期的IC值
        """
        df = df.copy()

        # 计算未来收益率
        decay_results = {}
        for period in range(1, max_periods + 1):
            df[f"future_return_{period}"] = df["close"].pct_change(period).shift(-period)

            # 计算IC
            factor_clean = df[factor_name].dropna()
            return_clean = df[f"future_return_{period}"].dropna()

            # 对齐数据
            aligned_idx = factor_clean.index.intersection(return_clean.index)
            if len(aligned_idx) > 10:  # 至少需要10个样本
                ic = factor_clean.loc[aligned_idx].corr(return_clean.loc[aligned_idx])
                decay_results[f"period_{period}"] = ic
            else:
                decay_results[f"period_{period}"] = np.nan

        return decay_results

    # ==================== 因子稳定性分析 ====================

    def calculate_periodic_ic(
        self, factor_data: pd.DataFrame, return_data: pd.DataFrame, period: str = "M"
    ) -> Dict[str, float]:
        """
        计算不同周期的IC，分析因子稳定性

        Args:
            factor_data: 因子数据，索引为日期
            return_data: 收益数据，索引为日期
            period: 周期类型，"M"月度，"Q"季度，"Y"年度

        Returns:
            Dict: 各周期的IC值
        """
        # 确保索引是datetime
        if not isinstance(factor_data.index, pd.DatetimeIndex):
            factor_data = factor_data.copy()
            factor_data.index = pd.to_datetime(factor_data.index)

        if not isinstance(return_data.index, pd.DatetimeIndex):
            return_data = return_data.copy()
            return_data.index = pd.to_datetime(return_data.index)

        # 按周期重采样
        periodic_ic = {}

        normalized_period = self._normalize_resample_period(period)

        for period_name, factor_group in factor_data.resample(normalized_period):
            if len(factor_group) == 0:
                continue

            # 获取对应区间的收益数据
            start_idx = factor_group.index[0]
            end_idx = factor_group.index[-1]

            return_mask = (return_data.index >= start_idx) & (return_data.index <= end_idx)
            return_group = return_data[return_mask]

            # 对齐并计算IC
            if len(return_group) > 0:
                ic = self._calculate_aligned_correlation(factor_group, return_group)
                if ic is not None:
                    periodic_ic[str(period_name.date())] = ic

        return periodic_ic

    def calculate_rolling_ic_stability(
        self, ic_series: pd.Series, windows: List[int] = [20, 60, 120, 252]
    ) -> Dict:
        """
        计算不同滚动窗口下的IC统计量，评估因子稳定性

        Args:
            ic_series: IC时间序列
            windows: 滚动窗口列表

        Returns:
            Dict: 各窗口下的IC统计量
        """
        results = {}

        for window in windows:
            min_periods = max(1, window // 4)
            rolling_mean = ic_series.rolling(window=window, min_periods=min_periods).mean()
            rolling_std = ic_series.rolling(window=window, min_periods=min_periods).std()
            rolling_ir = rolling_mean / rolling_std.replace(0, np.nan)

            results[f"window_{window}"] = {
                "mean_ic": rolling_mean.iloc[-1] if len(rolling_mean) > 0 else np.nan,
                "std_ic": rolling_std.iloc[-1] if len(rolling_std) > 0 else np.nan,
                "ir": rolling_ir.iloc[-1] if len(rolling_ir) > 0 else np.nan,
            }

        return results

    def calculate_market_regime_ic(
        self, factor_data: Dict[str, pd.DataFrame], return_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        计算不同市场环境下的IC（牛市、熊市、震荡市）

        Args:
            factor_data: 按市场环境分类的因子数据
            return_data: 按市场环境分类的收益数据

        Returns:
            Dict: 各市场环境下的IC
        """
        results = {}

        for regime in factor_data.keys():
            factor_df = factor_data[regime]
            return_df = return_data.get(regime)

            if return_df is not None:
                ic = self._calculate_aligned_correlation(factor_df, return_df)
                results[regime] = ic if ic is not None else np.nan

        return results

    # ==================== 因子交互效应分析 ====================

    def analyze_factor_interactions(
        self, df: pd.DataFrame, factor_names: List[str], degree: int = 2
    ) -> Dict:
        """
        分析因子交互效应

        Args:
            df: 包含因子和收益数据的DataFrame
            factor_names: 因子名称列表
            degree: 多项式阶数，默认2（包含平方项和交互项）

        Returns:
            Dict: 交互效应分析结果
        """
        # 提取因子数据
        factor_df = df[factor_names].copy()

        # 删除缺失值
        factor_df = factor_df.dropna()

        target_column = self._resolve_target_column(df, factor_names)
        if target_column is None:
            return {
                "interaction_features": [],
                "feature_importance": {},
                "error": "缺少目标收益列（如 future_return、return、future_return_1）",
            }

        interaction_frame = factor_df.copy()
        interaction_frame[target_column] = df[target_column]
        interaction_frame = interaction_frame.dropna()

        if len(interaction_frame) < 10:
            return {
                "interaction_features": [],
                "feature_importance": {},
                "error": "有效样本不足，无法分析交互效应",
            }

        # 创建多项式特征（包含交互项）
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(interaction_frame[factor_names])

        # 获取特征名称
        feature_names = poly.get_feature_names_out(factor_names)

        y = interaction_frame[target_column].astype(float)
        feature_matrix = pd.DataFrame(
            poly_features,
            columns=feature_names,
            index=interaction_frame.index,
        )

        model = LinearRegression()
        model.fit(feature_matrix, y)

        coefficient_series = pd.Series(model.coef_, index=feature_names, dtype=float)
        standardized_features = (
            feature_matrix - feature_matrix.mean()
        ) / feature_matrix.std(ddof=0).replace(0, np.nan)
        standardized_coefficients = coefficient_series * standardized_features.std(ddof=0).fillna(0.0)

        interaction_results = {}
        for i, feat_name in enumerate(feature_names):
            feature_series = feature_matrix[feat_name]
            correlation = feature_series.corr(y)
            interaction_results[feat_name] = {
                "index": i,
                "is_interaction": " " in feat_name and "^2" not in feat_name,
                "is_squared": "^2" in feat_name,
                "coefficient": float(coefficient_series.loc[feat_name]),
                "standardized_coefficient": float(standardized_coefficients.loc[feat_name]),
                "correlation_with_target": float(correlation) if pd.notna(correlation) else np.nan,
            }

        importance_series = standardized_coefficients.abs().sort_values(ascending=False)

        return {
            "interaction_features": feature_names.tolist(),
            "feature_info": interaction_results,
            "feature_importance": {
                feature_name: float(importance)
                for feature_name, importance in importance_series.items()
            },
            "target_column": target_column,
            "r_squared": float(model.score(feature_matrix, y)),
        }

    def calculate_factor_correlation_matrix(
        self, df: pd.DataFrame, factor_names: List[str]
    ) -> pd.DataFrame:
        """
        计算因子相关性矩阵

        Args:
            df: 包含因子数据的DataFrame
            factor_names: 因子名称列表

        Returns:
            pd.DataFrame: 相关性矩阵
        """
        factor_df = df[factor_names].copy()
        factor_df = factor_df.dropna()

        if len(factor_df) == 0:
            return pd.DataFrame()

        corr_matrix = factor_df.corr()
        return corr_matrix

    # ==================== 因子拥挤度分析 ====================

    def calculate_factor_crowding(
        self, df: pd.DataFrame, factor_name: str, window: int = 20
    ) -> pd.Series:
        """
        计算因子拥挤度（因子值的标准差，越小越拥挤）

        Args:
            df: 包含因子数据的DataFrame
            factor_name: 因子名称
            window: 滚动窗口

        Returns:
            pd.Series: 拥挤度指标
        """
        factor_values = df[factor_name].dropna()

        # 计算滚动标准差（反向：拥挤度 = 1 / std）
        rolling_std = factor_values.rolling(window=window, min_periods=1).std()

        # 拥挤度指标（标准化到0-1）
        crowding = 1 / (1 + rolling_std)

        return crowding

    def calculate_turnover(
        self, signals: pd.Series, lag: int = 1
    ) -> Dict:
        """
        计算因子换手率

        Args:
            signals: 信号序列（0或1）
            lag: 滞后期

        Returns:
            Dict: 换手率统计
        """
        if len(signals) == 0:
            return {"turnover_rate": 0.0, "avg_turnover": 0.0}

        # 计算信号变化
        signal_changes = signals.diff(lag).abs()

        # 换手率 = 信号变化 / 总信号数
        turnover_rate = signal_changes.mean()

        return {
            "turnover_rate": turnover_rate,
            "avg_turnover": signal_changes.mean(),
        }

    # ==================== 因子分层收益分析 ====================

    def analyze_quantile_returns(
        self, quantile_returns: Dict[str, pd.Series], annual_trading_days: int = 252
    ) -> Dict:
        """
        分析各分层收益的统计特性

        Args:
            quantile_returns: 各分层收益字典
            annual_trading_days: 年化交易日数

        Returns:
            Dict: 各分层收益统计
        """
        results = {}

        for quantile_name, returns in quantile_returns.items():
            returns_clean = returns.dropna()

            if len(returns_clean) == 0:
                results[quantile_name] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "annual_return": 0.0,
                    "sharpe": 0.0,
                    "win_rate": 0.0,
                }
                continue

            mean = returns_clean.mean()
            std = returns_clean.std()
            annual_return = mean * annual_trading_days
            sharpe = mean / std if std > 0 else 0.0
            win_rate = (returns_clean > 0).mean()

            results[quantile_name] = {
                "mean": mean,
                "std": std,
                "annual_return": annual_return,
                "sharpe": sharpe,
                "win_rate": win_rate,
            }

        return results

    # ==================== 因子IC预测能力分析 ====================

    def calculate_ic_predictability(
        self, ic_series: pd.Series, lag: int = 5
    ) -> Dict:
        """
        计算IC的自相关性和可预测性

        Args:
            ic_series: IC时间序列
            lag: 最大滞后阶数

        Returns:
            Dict: 自相关分析结果
        """
        autocorrs = []
        for l in range(1, lag + 1):
            autocorr = ic_series.autocorr(lag=l)
            autocorrs.append(autocorr)

        return {
            "autocorrelations": autocorrs,
            "mean_abs_autocorr": np.mean(np.abs(autocorrs)),
        }

    def _calculate_aligned_correlation(
        self,
        factor_values: pd.Series | pd.DataFrame,
        return_values: pd.Series | pd.DataFrame,
    ) -> Optional[float]:
        """对齐后计算单一因子与单一收益序列的相关系数。"""
        factor_series = self._coerce_to_series(factor_values, "factor")
        return_series = self._coerce_to_series(return_values, "return")

        aligned = pd.DataFrame(
            {
                "factor": factor_series,
                "return": return_series,
            }
        ).dropna()

        if len(aligned) <= 10:
            return None

        ic = aligned["factor"].corr(aligned["return"])
        if pd.isna(ic):
            return None
        return float(ic)

    def _coerce_to_series(
        self,
        values: pd.Series | pd.DataFrame,
        default_name: str,
    ) -> pd.Series:
        """将 Series 或单列 DataFrame 转为 Series。"""
        if isinstance(values, pd.Series):
            return values.rename(default_name)
        if isinstance(values, pd.DataFrame):
            if values.shape[1] == 0:
                return pd.Series(dtype=float, name=default_name)
            if values.shape[1] == 1:
                return values.iloc[:, 0].rename(values.columns[0])
            numeric_columns = values.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return pd.Series(dtype=float, name=default_name)
            return values[numeric_columns[0]].rename(numeric_columns[0])
        raise TypeError(f"不支持的数据类型: {type(values)}")

    def _resolve_target_column(self, df: pd.DataFrame, factor_names: List[str]) -> Optional[str]:
        """解析交互效应分析使用的目标列。"""
        preferred_columns = ["future_return", "future_return_1", "return", "target"]
        for column in preferred_columns:
            if column in df.columns and column not in factor_names:
                return column
        return None

    def _normalize_resample_period(self, period: str) -> str:
        """兼容 pandas 新版本的重采样频率写法。"""
        period_map = {
            "M": "ME",
            "Q": "QE",
            "Y": "YE",
        }
        return period_map.get(period, period)
