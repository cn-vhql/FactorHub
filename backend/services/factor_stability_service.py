"""
因子稳定性分析服务
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from backend.services.analysis_service import AnalysisService
from backend.services.factor_service import factor_service


class FactorStabilityService:
    """因子稳定性分析服务类"""

    def __init__(self):
        pass

    def comprehensive_stability_test(
        self,
        factor_name: str,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict:
        """对指定因子执行完整稳定性检验。"""
        factor_data_map = factor_service.calculate_factors_for_stocks(
            stock_codes=stock_codes,
            factor_names=[factor_name],
            start_date=start_date,
            end_date=end_date,
            rolling_window=None,
        )

        if not factor_data_map:
            raise ValueError("未能获取有效的因子数据")

        combined_segments = []
        per_stock = {}

        for stock_code, df in factor_data_map.items():
            if factor_name not in df.columns or "close" not in df.columns:
                continue

            stock_df = df.copy()
            stock_df["future_return"] = stock_df["close"].pct_change().shift(-1)
            stock_df = stock_df.replace([np.inf, -np.inf], np.nan)

            factor_series = stock_df[factor_name].dropna()
            ic_series = stock_df[factor_name].rolling(window=20, min_periods=10).corr(
                stock_df["future_return"]
            ).dropna()

            per_stock_result = {}

            if len(factor_series) >= 40:
                try:
                    window = min(252, max(20, len(factor_series) // 2))
                    per_stock_result["distribution_stability"] = self.calculate_distribution_stability(
                        factor_series=factor_series,
                        window=window,
                    )
                except Exception as exc:
                    per_stock_result["distribution_stability"] = {"error": str(exc)}

            if len(ic_series) >= 20:
                per_stock_result["time_series_stability"] = self.calculate_time_series_stability(ic_series)
                per_stock_result["coefficient_of_variation"] = self.calculate_coefficient_of_variation(ic_series)
            else:
                per_stock_result["time_series_stability"] = {"error": "IC序列长度不足，无法执行ADF检验"}
                per_stock_result["coefficient_of_variation"] = {"error": "IC序列长度不足"}

            per_stock_result["rolling_stability"] = self.calculate_rolling_stability(
                factor_data=stock_df,
                factor_name=factor_name,
                return_col="future_return",
            )

            try:
                per_stock_result["market_regime_performance"] = self.calculate_market_regime_performance(
                    factor_data=stock_df,
                    factor_name=factor_name,
                    return_col="future_return",
                )
            except Exception as exc:
                per_stock_result["market_regime_performance"] = {"error": str(exc)}

            per_stock[stock_code] = per_stock_result
            combined_segments.append(
                stock_df[[factor_name, "future_return", "close"]].assign(stock_code=stock_code)
            )

        if not combined_segments:
            raise ValueError("未能形成可用于稳定性分析的数据集")

        combined_df = pd.concat(combined_segments, axis=0).sort_index()
        combined_factor = combined_df[factor_name].dropna()
        combined_ic_series = combined_df[factor_name].rolling(window=20, min_periods=10).corr(
            combined_df["future_return"]
        ).dropna()

        aggregate = {}
        if len(combined_factor) >= 40:
            try:
                window = min(252, max(20, len(combined_factor) // 2))
                aggregate["distribution_stability"] = self.calculate_distribution_stability(
                    factor_series=combined_factor,
                    window=window,
                )
            except Exception as exc:
                aggregate["distribution_stability"] = {"error": str(exc)}
        else:
            aggregate["distribution_stability"] = {"error": "样本长度不足，无法执行分布稳定性检验"}

        if len(combined_ic_series) >= 20:
            aggregate["time_series_stability"] = self.calculate_time_series_stability(combined_ic_series)
            aggregate["coefficient_of_variation"] = self.calculate_coefficient_of_variation(combined_ic_series)
        else:
            aggregate["time_series_stability"] = {"error": "综合IC序列长度不足"}
            aggregate["coefficient_of_variation"] = {"error": "综合IC序列长度不足"}

        aggregate["rolling_stability"] = self.calculate_rolling_stability(
            factor_data=combined_df,
            factor_name=factor_name,
            return_col="future_return",
        )

        return {
            "factor_name": factor_name,
            "stock_codes": list(factor_data_map.keys()),
            "analysis_period": {
                "start_date": start_date,
                "end_date": end_date,
            },
            "aggregate": aggregate,
            "per_stock": per_stock,
        }

    def calculate_distribution_stability(
        self,
        factor_series: pd.Series,
        window: int = 252,
        method: str = "ks"
    ) -> Dict:
        """
        分布稳定性分析 - 使用KS检验比较不同窗口期的分布

        Args:
            factor_series: 因子值序列
            window: 窗口大小（交易日）
            method: 检验方法，"ks"（Kolmogorov-Smirnov）或 "ttest"

        Returns:
            稳定性分析结果
        """
        if len(factor_series) < window * 2:
            raise ValueError(f"数据长度不足，至少需要 {window * 2} 个数据点")

        results = {}
        p_values = []
        test_statistics = []

        # 分段数据
        n_windows = len(factor_series) // window
        segments = []
        for i in range(n_windows):
            start_idx = i * window
            end_idx = start_idx + window
            segments.append(factor_series.iloc[start_idx:end_idx].dropna())

        # 两两比较
        comparisons = []
        for i in range(len(segments) - 1):
            for j in range(i + 1, len(segments)):
                segment1 = segments[i]
                segment2 = segments[j]

                if method == "ks":
                    # Kolmogorov-Smirnov检验
                    statistic, p_value = stats.ks_2samp(segment1, segment2)
                elif method == "ttest":
                    # t检验
                    statistic, p_value = stats.ttest_ind(segment1, segment2)
                else:
                    raise ValueError(f"不支持的检验方法: {method}")

                comparisons.append({
                    "segment1": i,
                    "segment2": j,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                })
                p_values.append(p_value)
                test_statistics.append(statistic)

        # 汇总统计
        avg_p_value = np.mean(p_values)
        stable_ratio = sum(1 for p in p_values if p > 0.05) / len(p_values)

        results = {
            "method": method,
            "window": window,
            "n_comparisons": len(comparisons),
            "avg_p_value": float(avg_p_value),
            "stable_ratio": float(stable_ratio),
            "stability_score": float(stable_ratio),  # 稳定性得分
            "comparisons": comparisons[:10],  # 只返回前10个比较
        }

        return results

    def calculate_time_series_stability(
        self,
        ic_series: pd.Series,
        maxlag: int = 10
    ) -> Dict:
        """
        时间序列稳定性分析 - 使用ADF检验判断平稳性

        Args:
            ic_series: IC值序列
            maxlag: ADF检验的最大滞后阶数

        Returns:
            平稳性分析结果
        """
        # 移除缺失值
        ic_clean = ic_series.dropna()

        if len(ic_clean) < 20:
            raise ValueError("IC序列长度不足，至少需要20个数据点")

        # ADF检验
        try:
            result = adfuller(ic_clean, maxlag=maxlag)

            adf_statistic = float(result[0])
            p_value = float(result[1])
            used_lag = int(result[2])
            n_obs = int(result[3])
            critical_values = result[4]

            # 判断平稳性（p < 0.05）
            is_stationary = p_value < 0.05

            return {
                "is_stationary": is_stationary,
                "adf_statistic": adf_statistic,
                "p_value": p_value,
                "used_lag": used_lag,
                "n_obs": n_obs,
                "critical_values": {
                    "1%": float(critical_values['1%']),
                    "5%": float(critical_values['5%']),
                    "10%": float(critical_values['10%']),
                },
                "interpretation": (
                    "序列平稳，拒绝存在单位根的原假设" if is_stationary
                    else "序列不平稳，存在单位根"
                ),
            }

        except Exception as e:
            return {
                "error": str(e),
                "is_stationary": None,
            }

    def calculate_coefficient_of_variation(
        self,
        ic_series: pd.Series,
    ) -> Dict:
        """
        计算变异系数 - 衡量离散程度

        Args:
            ic_series: IC值序列

        Returns:
            变异系数统计
        """
        ic_clean = ic_series.dropna()

        if len(ic_clean) == 0:
            return {"error": "没有有效数据"}

        mean = ic_clean.mean()
        std = ic_clean.std()

        cv = std / mean if mean != 0 else np.nan

        return {
            "mean": float(mean),
            "std": float(std),
            "cv": float(cv),
            "interpretation": (
                "变异程度较低" if cv < 0.5 else
                "变异程度中等" if cv < 1.0 else
                "变异程度较高"
            ),
        }

    def calculate_rolling_stability(
        self,
        factor_data: pd.DataFrame,
        factor_name: str,
        return_col: str = "future_return",
        windows: List[int] = [20, 60, 120, 252]
    ) -> Dict:
        """
        滚动窗口稳定性分析 - 在不同窗口下计算IC

        Args:
            factor_data: 包含因子和收益率的数据框
            factor_name: 因子列名
            return_col: 收益率列名
            windows: 窗口大小列表

        Returns:
            各窗口的稳定性统计
        """
        results = {}

        for window in windows:
            if len(factor_data) < window * 2:
                continue

            # 计算滚动IC
            rolling_ic = []
            for i in range(window, len(factor_data)):
                window_data = factor_data.iloc[i-window:i]
                if factor_name in window_data.columns and return_col in window_data.columns:
                    ic = window_data[factor_name].corr(window_data[return_col])
                    if not np.isnan(ic):
                        rolling_ic.append(ic)

            if rolling_ic:
                ic_series = pd.Series(rolling_ic)
                results[f"window_{window}"] = {
                    "window": window,
                    "mean_ic": float(ic_series.mean()),
                    "std_ic": float(ic_series.std()),
                    "ir": float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else np.nan,
                    "cv": float(ic_series.std() / ic_series.mean()) if ic_series.mean() != 0 else np.nan,
                }

        return results

    def calculate_market_regime_performance(
        self,
        factor_data: pd.DataFrame,
        factor_name: str,
        return_col: str = "future_return",
        price_col: str = "close",
        bull_threshold: float = 0.05,
        bear_threshold: float = -0.05
    ) -> Dict:
        """
        不同市场环境下的表现分析

        Args:
            factor_data: 因子数据
            factor_name: 因子列名
            return_col: 收益率列名
            price_col: 价格列名（用于判断市场环境）
            bull_threshold: 牛市阈值
            bear_threshold: 熊市阈值

        Returns:
            各市场环境下的IC统计
        """
        if price_col not in factor_data.columns:
            raise ValueError(f"数据框中缺少价格列: {price_col}")

        # 计算市场累计收益率
        factor_data = factor_data.copy()
        factor_data['market_return'] = factor_data[price_col].pct_change()

        # 划分市场环境
        regimes = []
        current_regime = "unknown"
        regime_start = 0

        for i in range(len(factor_data)):
            if i == 0:
                continue

            # 简单的滚动判断：过去20天的累计收益率
            if i >= 20:
                recent_return = factor_data['market_return'].iloc[i-20:i].sum()

                if recent_return > bull_threshold:
                    current_regime = "bull"
                elif recent_return < bear_threshold:
                    current_regime = "bear"
                else:
                    current_regime = "flat"

                # 记录环境变化
                if i == len(factor_data) - 1:
                    regimes.append({
                        "start": regime_start,
                        "end": i,
                        "regime": current_regime
                    })
                elif len(regimes) > 0 and regimes[-1]["regime"] != current_regime:
                    regimes[-1]["end"] = regime_start
                    regimes.append({
                        "start": regime_start,
                        "end": i,
                        "regime": current_regime
                    })
                    regime_start = i

        # 计算各环境下的IC
        regime_performance = {}
        for regime_info in regimes:
            regime = regime_info["regime"]
            start_idx = regime_info["start"]
            end_idx = regime_info["end"]

            regime_data = factor_data.iloc[start_idx:end_idx]

            if factor_name in regime_data.columns and return_col in regime_data.columns:
                ic = regime_data[factor_name].corr(regime_data[return_col])

                if regime not in regime_performance:
                    regime_performance[regime] = []

                regime_performance[regime].append(ic if not np.isnan(ic) else 0)

        # 汇总
        results = {}
        for regime, ics in regime_performance.items():
            results[regime] = {
                "mean_ic": float(np.mean(ics)),
                "std_ic": float(np.std(ics)),
                "n_periods": len(ics),
            }

        return results


# 全局因子稳定性分析服务实例
factor_stability_service = FactorStabilityService()
