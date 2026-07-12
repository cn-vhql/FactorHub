"""
因子摘要统计服务 - 自动生成因子统计摘要
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


class FactorSummaryService:
    """因子摘要统计服务类"""

    def __init__(self):
        pass

    def generate_factor_summary(
        self,
        factor_name: str,
        factor_data: pd.DataFrame,
        ic_analysis: Dict,
        stability_analysis: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        生成因子统计摘要

        Args:
            factor_name: 因子名称
            factor_data: 因子数据
            ic_analysis: IC分析结果
            stability_analysis: 稳定性分析结果（可选）

        Returns:
            因子摘要报告
        """
        summary = {
            "factor_name": factor_name,
            "basic_stats": self._calculate_basic_stats(factor_data),
            "ic_summary": self._summarize_ic_analysis(ic_analysis),
            "stability_summary": self._summarize_stability(stability_analysis) if stability_analysis else None,
            "quality_score": None,
            "grade": None,
        }

        # 计算质量得分
        summary["quality_score"] = self._calculate_quality_score(summary)
        summary["grade"] = self._get_grade(summary["quality_score"])

        return summary

    def _calculate_basic_stats(self, factor_data: pd.DataFrame) -> Dict:
        """计算基础统计"""
        # 假设因子数据已经标准化
        stats = {}

        for col in factor_data.columns:
            if col == factor_data.index.name:
                continue

            series = factor_data[col].dropna()
            if len(series) > 0:
                stats[col] = {
                    "count": int(len(series)),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "missing_count": int(factor_data[col].isna().sum()),
                }

        return stats

    def _summarize_ic_analysis(self, ic_analysis: Dict) -> Dict:
        """汇总IC分析"""
        if "ic_stats" not in ic_analysis:
            return {"error": "无IC分析数据"}

        ic_stats = ic_analysis["ic_stats"]
        summary = {}

        for factor_name, stats in ic_stats.items():
            if isinstance(stats, dict) and "IC均值" in stats:
                summary[factor_name] = {
                    "ic_mean": float(stats.get("IC均值", 0)),
                    "ic_std": float(stats.get("IC标准差", 0)),
                    "ir": float(stats.get("IR", 0)),
                    "ic_positive_ratio": float(stats.get("IC>0占比", 0)),
                    "ic_abs_mean": float(stats.get("IC绝对值均值", 0)),
                }

        return summary

    def _summarize_stability(self, stability_analysis: Dict) -> Dict:
        """汇总稳定性分析"""
        summary = {}

        # 分布稳定性
        if "distribution_stability" in stability_analysis:
            dist_stability = stability_analysis["distribution_stability"]
            summary["distribution"] = {
                "stability_score": float(dist_stability.get("stability_score", 0)),
                "stable_ratio": float(dist_stability.get("stable_ratio", 0)),
            }

        # 时间序列稳定性
        if "time_series_stability" in stability_analysis:
            ts_stability = stability_analysis["time_series_stability"]
            summary["time_series"] = {
                "is_stationary": bool(ts_stability.get("is_stationary", False)),
                "p_value": float(ts_stability.get("p_value", 1)),
            }

        # 滚动窗口稳定性
        if "rolling_stability" in stability_analysis:
            rolling_stability = stability_analysis["rolling_stability"]
            summary["rolling"] = rolling_stability

        return summary

    def _calculate_quality_score(self, summary: Dict) -> float:
        """
        计算因子质量得分（0-100）

        评分维度：
        - IC均值（40%）：绝对值越大越好
        - IR（30%）：越大越好
        - 稳定性（20%）：综合稳定性得分
        - IC>0占比（10%）：方向准确性
        """
        score = 0.0

        ic_summary = summary.get("ic_summary", {})
        factor_metrics = [value for value in ic_summary.values() if isinstance(value, dict)]
        if factor_metrics:
            avg_ic_mean = float(np.mean([abs(item.get("ic_mean", 0.0)) for item in factor_metrics]))
            avg_ir = float(np.mean([item.get("ir", 0.0) for item in factor_metrics]))
            avg_positive_ratio = float(np.mean([item.get("ic_positive_ratio", 0.0) for item in factor_metrics]))

            score += min(avg_ic_mean * 400, 40)
            score += min(avg_ir * 10, 30)
        else:
            avg_positive_ratio = 0.0

        # 稳定性得分（0-20分）
        stability_summary = summary.get("stability_summary")
        if stability_summary and "distribution" in stability_summary:
            stability_score = stability_summary["distribution"].get("stability_score", 0)
            score += stability_score * 20

        # IC>0占比得分（0-10分）
        score += avg_positive_ratio * 10

        return round(score, 2)

    def _get_grade(self, score: float) -> str:
        """根据得分返回评级"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C+"
        elif score >= 40:
            return "C"
        else:
            return "D"

    def generate_multi_factor_summary(
        self,
        factors_summary: List[Dict]
    ) -> Dict:
        """
        生成多因子对比摘要

        Args:
            factors_summary: 多个因子的摘要列表

        Returns:
            对比摘要报告
        """
        comparison = {
            "factors": [],
            "ranking": [],
        }

        for summary in factors_summary:
            factor_name = summary.get("factor_name", "Unknown")
            quality_score = summary.get("quality_score", 0)
            grade = summary.get("grade", "D")

            comparison["factors"].append({
                "name": factor_name,
                "score": quality_score,
                "grade": grade,
            })

        # 排序
        comparison["factors"].sort(key=lambda x: x["score"], reverse=True)
        comparison["ranking"] = [f["name"] for f in comparison["factors"]]

        return comparison

    def generate_report_text(self, summary: Dict) -> str:
        """
        生成文本格式的因子报告

        Args:
            summary: 因子摘要

        Returns:
            文本报告
        """
        factor_name = summary.get("factor_name", "Unknown")
        score = summary.get("quality_score", 0)
        grade = summary.get("grade", "N/A")

        report = f"""
# {factor_name} 因子分析报告

## 质量评级
**得分**: {score}/100
**等级**: {grade}

## IC分析摘要
"""

        # 添加IC统计
        ic_summary = summary.get("ic_summary", {})
        if ic_summary:
            for factor, stats in ic_summary.items():
                if isinstance(stats, dict):
                    report += f"""
**因子**: {factor}
- IC均值: {stats.get('ic_mean', 0):.4f}
- IC标准差: {stats.get('ic_std', 0):.4f}
- 信息比率(IR): {stats.get('ir', 0):.4f}
- IC>0占比: {stats.get('ic_positive_ratio', 0):.2%}
"""

        # 添加稳定性分析
        stability_summary = summary.get("stability_summary")
        if stability_summary:
            report += "\n## 稳定性分析\n"

            if "distribution" in stability_summary:
                dist = stability_summary["distribution"]
                report += f"""
**分布稳定性**
- 稳定性得分: {dist.get('stability_score', 0):.2f}
- 稳定比例: {dist.get('stable_ratio', 0):.2%}
"""

            if "time_series" in stability_summary:
                ts = stability_summary["time_series"]
                is_stationary = ts.get("is_stationary", False)
                report += f"""
**时间序列稳定性**
- 平稳性: {'是' if is_stationary else '否'}
- P值: {ts.get('p_value', 1):.4f}
"""

        return report


# 全局因子摘要统计服务实例
factor_summary_service = FactorSummaryService()
