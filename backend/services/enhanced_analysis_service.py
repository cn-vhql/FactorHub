"""
阶段3增强分析服务 - 扩展AnalysisService功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from scipy import stats

from backend.services.factor_neutralization_service import factor_neutralization_service
from backend.services.factor_stability_service import factor_stability_service
from backend.services.factor_summary_service import factor_summary_service
from backend.services.factor_service import factor_service


class EnhancedAnalysisService:
    """增强分析服务 - 集成阶段3新功能"""

    def __init__(self):
        pass

    def analyze_multi_period_ic(
        self,
        factor_name: str,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        periods: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """分析因子在多预测周期下的IC表现。"""
        if periods is None:
            periods = [1, 5, 10, 20]

        factor_data_map = factor_service.calculate_factors_for_stocks(
            stock_codes=stock_codes,
            factor_names=[factor_name],
            start_date=start_date,
            end_date=end_date,
            rolling_window=None,
        )

        if not factor_data_map:
            raise ValueError("未能获取有效的因子数据")

        period_results = []
        per_stock_results = {stock_code: {} for stock_code in factor_data_map.keys()}

        for period in periods:
            ic_values = []
            ic_by_stock = {}

            for stock_code, df in factor_data_map.items():
                if factor_name not in df.columns or "close" not in df.columns:
                    continue

                future_returns = df["close"].pct_change(period).shift(-period)
                aligned = pd.DataFrame({
                    "factor": df[factor_name],
                    "return": future_returns,
                }).replace([np.inf, -np.inf], np.nan).dropna()

                if len(aligned) < 10:
                    continue

                ic = aligned["factor"].corr(aligned["return"])
                if pd.notna(ic):
                    ic_float = float(ic)
                    ic_values.append(ic_float)
                    ic_by_stock[stock_code] = ic_float
                    per_stock_results[stock_code][f"{period}d"] = {
                        "ic": ic_float,
                        "sample_size": int(len(aligned)),
                    }

            if ic_values:
                period_results.append({
                    "period_days": period,
                    "period_label": f"{period}日",
                    "mean_ic": float(np.mean(ic_values)),
                    "std_ic": float(np.std(ic_values)),
                    "positive_ratio": float(np.mean([value > 0 for value in ic_values])),
                    "sample_size": len(ic_values),
                    "per_stock_ic": ic_by_stock,
                })
            else:
                period_results.append({
                    "period_days": period,
                    "period_label": f"{period}日",
                    "mean_ic": None,
                    "std_ic": None,
                    "positive_ratio": None,
                    "sample_size": 0,
                    "per_stock_ic": {},
                })

        valid_periods = [item for item in period_results if item["mean_ic"] is not None]
        decay_trend = None
        if len(valid_periods) >= 2:
            x = np.array([item["period_days"] for item in valid_periods], dtype=float)
            y = np.array([item["mean_ic"] for item in valid_periods], dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            decay_trend = {
                "slope": float(slope),
                "intercept": float(intercept),
                "interpretation": "IC随周期延长衰减" if slope < 0 else "IC未呈现明显衰减",
            }

        return {
            "factor_name": factor_name,
            "stock_codes": list(factor_data_map.keys()),
            "analysis_period": {
                "start_date": start_date,
                "end_date": end_date,
            },
            "period_results": period_results,
            "decay_trend": decay_trend,
            "per_stock_results": per_stock_results,
        }

    def calculate_ic_significance(
        self,
        factor_values: pd.Series,
        return_values: pd.Series,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        计算IC的显著性t检验

        Args:
            factor_values: 因子值序列
            return_values: 收益率序列
            confidence_level: 置信水平（默认95%）

        Returns:
            IC显著性检验结果
        """
        # 移除缺失值
        valid_data = pd.DataFrame({
            "factor": factor_values,
            "return": return_values
        }).dropna()

        if len(valid_data) < 10:
            return {
                "error": "有效数据不足（至少需要10个数据点）",
                "n_samples": len(valid_data)
            }

        # 计算IC
        ic = valid_data["factor"].corr(valid_data["return"])

        # t检验：检验IC是否显著不为0
        n = len(valid_data)
        # t统计量计算
        # t = r * sqrt(n-2) / sqrt(1-r^2)
        if abs(ic) >= 1:
            t_statistic = 0
            p_value = 1.0
        else:
            t_statistic = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2)
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n - 2))

        # 计算置信区间
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 2)

        # 标准误差
        se = np.sqrt((1 - ic**2) / (n - 2))
        ci_lower = ic - t_critical * se
        ci_upper = ic + t_critical * se

        return {
            "ic": float(ic),
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "is_significant": p_value < 0.05,
            "significance_level": (
                "极高显著性 (p<0.01)" if p_value < 0.01 else
                "显著性 (p<0.05)" if p_value < 0.05 else
                "不显著 (p>=0.05)"
            ),
            "confidence_interval": {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "level": confidence_level,
            },
            "n_samples": n,
            "interpretation": (
                f"IC在{confidence_level*100:.0f}%置信区间为[{ci_lower:.4f}, {ci_upper:.4f}]"
            ),
        }

    def analyze_enhanced(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_names: List[str],
        enable_neutralization: bool = False,
        enable_stability: bool = False,
        enable_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        增强版因子分析 - 集成所有阶段3新功能

        Args:
            factor_data: 因子数据字典
            factor_names: 因子名称列表
            enable_neutralization: 是否启用中性化
            enable_stability: 是否启用稳定性分析
            enable_summary: 是否生成摘要统计

        Returns:
            增强的分析结果
        """
        results = {
            "factors": {},
            "neutralization": {},
            "stability": {},
            "summary": {},
        }

        for factor_name in factor_names:
            if factor_name not in factor_data:
                continue

            df = factor_data[factor_name]

            # 基础IC分析
            if "close" in df.columns and "future_return" in df.columns:
                # 计算未来收益率（假设已存在）
                ic_significance = self.calculate_ic_significance(
                    df[factor_name],
                    df["future_return"]
                )

                results["factors"][factor_name] = {
                    "ic_significance": ic_significance,
                }

            # 中性化处理
            if enable_neutralization:
                try:
                    # 市值中性化
                    mc_neutralized = factor_neutralization_service.neutralize_market_cap(
                        df, factor_name, "market_cap"
                    )

                    # 计算中性化后的IC
                    if "future_return" in df.columns:
                        ic_after_mc = mc_neutralized.corr(df["future_return"])
                        results["neutralization"][f"{factor_name}_mc"] = {
                            "method": "市值中性化",
                            "ic_before": results["factors"][factor_name]["ic_significance"]["ic"],
                            "ic_after": float(ic_after_mc),
                            "improvement": float(ic_after_mc - results["factors"][factor_name]["ic_significance"]["ic"]),
                        }

                    # 行业中性化
                    industry_neutralized = factor_neutralization_service.neutralize_industry(
                        df, factor_name, "industry"
                    )

                    if "future_return" in df.columns:
                        ic_after_ind = industry_neutralized.corr(df["future_return"])
                        results["neutralization"][f"{factor_name}_ind"] = {
                            "method": "行业中性化",
                            "ic_before": results["factors"][factor_name]["ic_significance"]["ic"],
                            "ic_after": float(ic_after_ind),
                            "improvement": float(ic_after_ind - results["factors"][factor_name]["ic_significance"]["ic"]),
                        }

                except Exception as e:
                    results["neutralization"][factor_name] = {
                        "error": str(e)
                    }

            # 稳定性分析
            if enable_stability:
                try:
                    # 分布稳定性
                    dist_stability = factor_stability_service.calculate_distribution_stability(
                        df[factor_name]
                    )

                    # 时间序列稳定性（如果有IC序列）
                    if "ic_series" in df.columns:
                        ts_stability = factor_stability_service.calculate_time_series_stability(
                            df["ic_series"]
                        )
                    else:
                        ts_stability = None

                    # 滚动窗口稳定性
                    rolling_stability = factor_stability_service.calculate_rolling_stability(
                        df, factor_name
                    )

                    results["stability"][factor_name] = {
                        "distribution_stability": dist_stability,
                        "time_series_stability": ts_stability,
                        "rolling_stability": rolling_stability,
                    }

                except Exception as e:
                    results["stability"][factor_name] = {
                        "error": str(e)
                    }

            # 生成摘要
            if enable_summary:
                try:
                    # 准备分析数据
                    ic_analysis = results.get("factors", {})
                    stability_analysis = results.get("stability", {})

                    summary = factor_summary_service.generate_factor_summary(
                        factor_name,
                        df,
                        ic_analysis,
                        stability_analysis
                    )

                    results["summary"][factor_name] = summary

                except Exception as e:
                    results["summary"][factor_name] = {
                        "error": str(e)
                    }

        return results


# 全局增强分析服务实例
enhanced_analysis_service = EnhancedAnalysisService()
