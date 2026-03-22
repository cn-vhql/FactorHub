"""
单票历史因子研究服务。

目标：
1. 对单只股票在给定历史窗口内评估因子有效性；
2. 输出可直接用于后续「近期因子筛选 / 策略搜索」的历史因子排名；
3. 保持与 recent_factor_mining_service 的评价口径尽量一致。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backend.services.recent_factor_mining_service import recent_factor_mining_service


class SingleStockFactorResearchService:
    """单票历史因子研究服务。"""

    @staticmethod
    def _pct_rank(series: pd.Series) -> pd.Series:
        if len(series) <= 1:
            return pd.Series([1.0] * len(series), index=series.index)
        return series.rank(method="average", pct=True)

    def research_factors(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        factor_names: list[str] | None = None,
        horizons: list[int] | None = None,
        lookback_days: int = 252,
        top_k: int = 20,
        min_samples: int = 60,
        correlation_threshold: float = 0.85,
    ) -> dict[str, Any]:
        """研究单票在历史窗口内的有效因子。"""
        if horizons is None:
            horizons = [1, 5, 10, 20]

        if not stock_code:
            raise ValueError("stock_code 不能为空")
        if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
            raise ValueError("start_date 必须早于 end_date")
        if any(h <= 0 for h in horizons):
            raise ValueError("horizons 必须全部为正整数")
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        if min_samples < 10:
            raise ValueError("min_samples 不能小于 10")

        all_factor_defs = recent_factor_mining_service._load_active_factor_definitions()
        if not all_factor_defs:
            raise ValueError("未找到任何启用因子")

        if factor_names:
            factor_defs = {
                name: all_factor_defs[name]
                for name in factor_names
                if name in all_factor_defs
            }
            missing = sorted(set(factor_names) - set(factor_defs))
            if not factor_defs:
                raise ValueError("指定的 factor_names 在当前启用因子中均不存在")
        else:
            factor_defs = all_factor_defs
            missing = []

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        fetch_start = (
            start_ts - timedelta(days=lookback_days + max(horizons) + 30)
        ).strftime("%Y-%m-%d")

        panels = recent_factor_mining_service._build_factor_panels(
            stock_codes=[stock_code],
            factor_defs=factor_defs,
            start_date=fetch_start,
            end_date=end_date,
        )
        panel = panels.get(stock_code)
        if panel is None or panel.empty:
            raise ValueError(f"无法为股票 {stock_code} 构建历史因子面板")

        evaluation = recent_factor_mining_service._evaluate_stock_factors(
            stock_code=stock_code,
            panel=panel,
            factor_names=list(factor_defs.keys()),
            eval_start=start_ts,
            horizons=horizons,
            min_samples=min_samples,
        )

        records = evaluation.get("records", [])
        if not records:
            raise ValueError("指定时间窗口内未得到有效因子评估结果")

        records_df = pd.DataFrame(records)
        records_df["historical_score"] = 100 * (
            0.45 * self._pct_rank(records_df["abs_ic_mean"])
            + 0.25 * self._pct_rank(records_df["abs_ir"])
            + 0.15 * self._pct_rank(records_df["sign_consistency"])
            + 0.15 * self._pct_rank(records_df["regime_abs_ic_mean"])
        )
        records_df = records_df.sort_values(
            "historical_score", ascending=False
        ).reset_index(drop=True)

        research_df = panel[(panel.index >= start_ts) & (panel.index <= end_ts)].copy()
        ordered_factors = records_df["factor_name"].tolist()
        if ordered_factors:
            corr_source = research_df[ordered_factors].replace([np.inf, -np.inf], np.nan)
            corr_matrix = corr_source.corr().abs()
        else:
            corr_matrix = pd.DataFrame()

        selected, redundant_map = recent_factor_mining_service._corr_prune(
            candidates=ordered_factors,
            corr_matrix=corr_matrix,
            threshold=correlation_threshold,
        )
        selected_set = set(selected)

        all_factor_scores: list[dict[str, Any]] = []
        for row in records_df.itertuples(index=False):
            redundant_info = redundant_map.get(row.factor_name)
            all_factor_scores.append(
                {
                    "factor_name": row.factor_name,
                    "historical_score": round(float(row.historical_score), 4),
                    "recommended_direction": int(row.recommended_direction),
                    "predictive_corr_mean": round(float(row.ic_mean), 6),
                    "abs_predictive_corr_mean": round(float(row.abs_ic_mean), 6),
                    "horizon_stability": round(float(row.abs_ir), 6),
                    "regime_predictive_corr_mean": round(
                        float(row.regime_ic_mean), 6
                    ),
                    "regime_abs_predictive_corr_mean": round(
                        float(row.regime_abs_ic_mean), 6
                    ),
                    "sign_consistency": round(float(row.sign_consistency), 6),
                    "selected": row.factor_name in selected_set,
                    "redundant_with": (
                        redundant_info.get("redundant_with")
                        if redundant_info
                        else None
                    ),
                    "redundancy_correlation": (
                        round(float(redundant_info["correlation"]), 6)
                        if redundant_info
                        else None
                    ),
                    "abs_ic_mean": round(float(row.abs_ic_mean), 6),
                    "ic_mean": round(float(row.ic_mean), 6),
                    "abs_ir": round(float(row.abs_ir), 6),
                    "regime_abs_ic_mean": round(float(row.regime_abs_ic_mean), 6),
                }
            )

        top_factors = [item for item in all_factor_scores if item["selected"]][:top_k]
        if len(top_factors) < top_k:
            for item in all_factor_scores:
                if item in top_factors:
                    continue
                top_factors.append(item)
                if len(top_factors) >= top_k:
                    break

        actual_end_date = panel.index.max().strftime("%Y-%m-%d")
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "stock_code": stock_code,
                "analysis_start_date": start_date,
                "analysis_end_date": end_date,
                "actual_end_date": actual_end_date,
                "current_regime": evaluation.get("current_regime", "unknown"),
                "lookback_days": lookback_days,
                "horizons": horizons,
                "min_samples": min_samples,
                "factor_count_total": len(factor_defs),
                "factor_count_scored": len(all_factor_scores),
                "selected_factor_count": len(selected_set),
                "missing_requested_factors": missing,
            },
            "top_factors": top_factors,
            "all_factor_scores": all_factor_scores,
        }


single_stock_factor_research_service = SingleStockFactorResearchService()
