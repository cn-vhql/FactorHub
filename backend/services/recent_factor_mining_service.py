"""
最近阶段有效因子挖掘服务。

目标：
1. 从因子库自动拉取活跃因子；
2. 在最近窗口内评估多周期 IC/IR；
3. 基于“当前趋势状态（上升/下降/震荡）”计算 regime 匹配强度；
4. 对高相关因子做去重，输出可落地的 Top 因子列表。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backend.core.database import get_db_session
from backend.repositories.factor_repository import FactorRepository
from backend.services.data_service import data_service
from backend.services.factor_service import factor_service

logger = logging.getLogger(__name__)


class RecentFactorMiningService:
    """从现有因子库中自动挖掘最近阶段有效因子的服务。"""

    def _load_active_factor_definitions(self) -> dict[str, str]:
        """加载全部启用因子的名称与表达式。"""
        db = get_db_session()
        repo = FactorRepository(db)
        try:
            factors = repo.get_all(active_only=True)
            return {factor.name: factor.code for factor in factors}
        finally:
            db.close()

    @staticmethod
    def _prepare_external_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        规范化外部特征 DataFrame。

        支持两种格式：
        1. index 为日期；
        2. 含 date 列（会自动设为索引）。
        """
        ext = df.copy()
        if "date" in ext.columns:
            ext["date"] = pd.to_datetime(ext["date"])
            ext = ext.set_index("date")
        if not isinstance(ext.index, pd.DatetimeIndex):
            ext.index = pd.to_datetime(ext.index)
        return ext.sort_index()

    def _build_factor_panels(
        self,
        stock_codes: list[str],
        factor_defs: dict[str, str],
        start_date: str,
        end_date: str,
        external_features: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        为每只股票构建包含 OHLCV + 因子值的面板数据。

        external_features 可选，按股票代码合并外部数据列，便于未来扩展新数据源。
        """
        panels: dict[str, pd.DataFrame] = {}

        for code in stock_codes:
            try:
                df = data_service.get_stock_data(code, start_date, end_date)
            except Exception as exc:
                logger.warning(f"股票 {code} 拉取行情失败，已跳过: {exc}")
                continue

            if df is None or df.empty or "close" not in df.columns:
                logger.warning(f"股票 {code} 无有效行情数据，已跳过")
                continue

            panel = df.sort_index().copy()

            if external_features and code in external_features:
                try:
                    ext = self._prepare_external_features(external_features[code])
                    panel = panel.join(ext, how="left")
                except Exception as exc:
                    logger.warning(f"股票 {code} 合并外部特征失败，继续仅用行情数据: {exc}")

            for factor_name, expr in factor_defs.items():
                try:
                    panel[factor_name] = factor_service.calculator.calculate(panel, expr)
                except Exception as exc:
                    logger.debug(f"股票 {code} 计算因子 {factor_name} 失败: {exc}")
                    panel[factor_name] = np.nan

            panels[code] = panel

        return panels

    @staticmethod
    def _calc_regime_series(close: pd.Series) -> pd.Series:
        """
        生成简单趋势状态序列：
        - uptrend: 价格 > SMA20 > SMA60 且 20 日动量 > 0
        - downtrend: 价格 < SMA20 < SMA60 且 20 日动量 < 0
        - range: 其他
        """
        sma20 = close.rolling(20, min_periods=20).mean()
        sma60 = close.rolling(60, min_periods=60).mean()
        mom20 = close.pct_change(20)

        regime = pd.Series("range", index=close.index, dtype="object")
        uptrend_mask = (close > sma20) & (sma20 > sma60) & (mom20 > 0)
        downtrend_mask = (close < sma20) & (sma20 < sma60) & (mom20 < 0)

        regime.loc[uptrend_mask] = "uptrend"
        regime.loc[downtrend_mask] = "downtrend"
        return regime

    @staticmethod
    def _safe_corr(x: pd.Series, y: pd.Series, min_samples: int) -> float | None:
        """在有效样本数满足阈值时计算相关系数。"""
        aligned = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(aligned) < min_samples:
            return None
        corr = aligned["x"].corr(aligned["y"])
        if pd.isna(corr) or np.isinf(corr):
            return None
        return float(corr)

    @staticmethod
    def _pct_rank(series: pd.Series) -> pd.Series:
        """百分位排名，单元素时返回 1.0，避免 0 除。"""
        if len(series) <= 1:
            return pd.Series([1.0] * len(series), index=series.index)
        return series.rank(method="average", pct=True)

    @staticmethod
    def _corr_prune(
        candidates: list[str],
        corr_matrix: pd.DataFrame,
        threshold: float,
    ) -> tuple[list[str], dict[str, dict[str, float | str]]]:
        """
        基于相关矩阵执行贪心去重。

        candidates 需按得分从高到低排序。
        """
        selected: list[str] = []
        redundant_map: dict[str, dict[str, float | str]] = {}

        if corr_matrix.empty:
            return candidates, redundant_map

        for factor in candidates:
            if factor not in corr_matrix.columns:
                selected.append(factor)
                continue

            hit_factor: str | None = None
            hit_corr = 0.0

            for chosen in selected:
                if chosen not in corr_matrix.columns:
                    continue
                corr_val = corr_matrix.loc[factor, chosen]
                if pd.isna(corr_val):
                    continue
                corr_val = float(corr_val)
                if corr_val >= threshold and corr_val > hit_corr:
                    hit_factor = chosen
                    hit_corr = corr_val

            if hit_factor is None:
                selected.append(factor)
            else:
                redundant_map[factor] = {
                    "redundant_with": hit_factor,
                    "correlation": hit_corr,
                }

        return selected, redundant_map

    def _evaluate_stock_factors(
        self,
        stock_code: str,
        panel: pd.DataFrame,
        factor_names: list[str],
        eval_start: pd.Timestamp,
        horizons: list[int],
        min_samples: int,
    ) -> dict[str, Any]:
        """评估单只股票在最近窗口内各因子的有效性。"""
        if panel.empty:
            return {"stock_code": stock_code, "current_regime": "unknown", "records": []}

        eval_df = panel[panel.index >= eval_start].copy()
        if eval_df.empty:
            return {"stock_code": stock_code, "current_regime": "unknown", "records": []}

        full_regime = self._calc_regime_series(panel["close"])
        eval_df["_regime"] = full_regime.reindex(eval_df.index).fillna("range")
        current_regime = str(eval_df["_regime"].iloc[-1])

        for h in horizons:
            eval_df[f"_fwd_ret_{h}"] = eval_df["close"].shift(-h) / eval_df["close"] - 1

        records: list[dict[str, Any]] = []
        regime_min_samples = max(10, min_samples // 2)

        for factor_name in factor_names:
            if factor_name not in eval_df.columns:
                continue

            factor_values = eval_df[factor_name].replace([np.inf, -np.inf], np.nan)
            horizon_ic: list[float] = []
            regime_horizon_ic: list[float] = []

            for h in horizons:
                ret_col = eval_df[f"_fwd_ret_{h}"]
                ic = self._safe_corr(factor_values, ret_col, min_samples=min_samples)
                if ic is not None:
                    horizon_ic.append(ic)

                regime_mask = eval_df["_regime"] == current_regime
                regime_ic = self._safe_corr(
                    factor_values[regime_mask],
                    ret_col[regime_mask],
                    min_samples=regime_min_samples,
                )
                if regime_ic is not None:
                    regime_horizon_ic.append(regime_ic)

            if not horizon_ic:
                continue

            ic_array = np.array(horizon_ic, dtype=float)
            ic_mean = float(ic_array.mean())
            abs_ic_mean = float(np.abs(ic_array).mean())
            ic_std = float(ic_array.std())
            abs_ir = float(abs(ic_mean) / (ic_std + 1e-8))

            dominant_sign = 1 if ic_mean >= 0 else -1
            non_zero_signs = [int(np.sign(v)) for v in ic_array if abs(v) > 1e-12]
            if non_zero_signs:
                sign_consistency = float(np.mean(np.array(non_zero_signs) == dominant_sign))
            else:
                sign_consistency = 0.5

            if regime_horizon_ic:
                regime_ic_mean = float(np.mean(regime_horizon_ic))
                regime_abs_ic_mean = float(np.mean(np.abs(regime_horizon_ic)))
                regime_sign_match = 1.0 if int(np.sign(regime_ic_mean)) == dominant_sign else 0.0
            else:
                regime_ic_mean = 0.0
                regime_abs_ic_mean = 0.0
                regime_sign_match = 0.0

            records.append(
                {
                    "stock_code": stock_code,
                    "factor_name": factor_name,
                    # 语义准确名称：时序 Pearson 相关（非横截面 IC）
                    "predictive_corr_mean": ic_mean,
                    "abs_predictive_corr_mean": abs_ic_mean,
                    "horizon_stability": abs_ir,          # |mean| / (std + ε)，衡量多周期稳定性
                    "sign_consistency": sign_consistency,
                    "regime_predictive_corr_mean": regime_ic_mean,
                    "regime_abs_predictive_corr_mean": regime_abs_ic_mean,
                    "regime_sign_match": regime_sign_match,
                    "recommended_direction": dominant_sign,
                    "horizon_count": len(horizon_ic),
                    # 向后兼容别名（保留旧字段名，避免破坏现有调用方）
                    "ic_mean": ic_mean,
                    "abs_ic_mean": abs_ic_mean,
                    "abs_ir": abs_ir,
                    "regime_ic_mean": regime_ic_mean,
                    "regime_abs_ic_mean": regime_abs_ic_mean,
                }
            )

        return {
            "stock_code": stock_code,
            "current_regime": current_regime,
            "records": records,
        }

    def mine_recent_valuable_factors(
        self,
        stock_codes: list[str],
        end_date: str,
        eval_days: int = 90,
        lookback_days: int = 240,
        horizons: list[int] | None = None,
        top_k: int = 20,
        min_samples: int = 25,
        correlation_threshold: float = 0.85,
        external_features: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, Any]:
        """
        自动挖掘最近阶段的高价值因子。

        Args:
            stock_codes: 已筛选股票列表
            end_date: 评估结束日期（YYYY-MM-DD）
            eval_days: 最近评估窗口（自然日）
            lookback_days: 因子预热窗口（自然日）
            horizons: 未来收益周期列表（如 [1, 5, 10]）
            top_k: 返回 TopK 因子
            min_samples: 计算相关性最小样本数
            correlation_threshold: 因子去重相关阈值
            external_features: 可选外部特征（按股票代码映射 DataFrame）

        Returns:
            包含全局与单票因子排序结果的字典。
        """
        if not stock_codes:
            raise ValueError("stock_codes 不能为空")

        if horizons is None:
            horizons = [1, 5, 10]

        if any(h <= 0 for h in horizons):
            raise ValueError("horizons 必须全部为正整数")

        # 参数范围校验（服务层防御性检查，与 API 层 Pydantic 校验互补）
        if eval_days < 10:
            raise ValueError(f"eval_days={eval_days} 过小，最小值为 10")
        if lookback_days < 30:
            raise ValueError(f"lookback_days={lookback_days} 过小，最小值为 30")
        if top_k <= 0:
            raise ValueError(f"top_k={top_k} 必须为正整数")
        if min_samples < 5:
            raise ValueError(f"min_samples={min_samples} 过小，最小值为 5")
        if not (0.0 <= correlation_threshold <= 1.0):
            raise ValueError(
                f"correlation_threshold={correlation_threshold} 超出范围，须在 [0.0, 1.0] 内"
            )

        factor_defs = self._load_active_factor_definitions()
        if not factor_defs:
            raise ValueError("未找到任何启用因子")

        end_ts = pd.Timestamp(end_date)
        warmup_days = lookback_days + eval_days + max(horizons) + 30
        start_ts = end_ts - timedelta(days=warmup_days)
        start_date = start_ts.strftime("%Y-%m-%d")

        panels = self._build_factor_panels(
            stock_codes=stock_codes,
            factor_defs=factor_defs,
            start_date=start_date,
            end_date=end_date,
            external_features=external_features,
        )
        if not panels:
            raise ValueError("无法构建任何股票的因子面板，请检查股票代码或数据源")

        eval_start = end_ts - timedelta(days=eval_days)
        factor_names = list(factor_defs.keys())

        stock_eval_results: dict[str, dict[str, Any]] = {}
        all_records: list[dict[str, Any]] = []

        for code, panel in panels.items():
            one_stock = self._evaluate_stock_factors(
                stock_code=code,
                panel=panel,
                factor_names=factor_names,
                eval_start=eval_start,
                horizons=horizons,
                min_samples=min_samples,
            )
            if one_stock["records"]:
                stock_eval_results[code] = one_stock
                all_records.extend(one_stock["records"])

        if not all_records:
            raise ValueError("最近窗口内未得到有效因子评估结果，请放宽窗口或降低最小样本数")

        records_df = pd.DataFrame(all_records)
        valid_stock_count = max(1, len(stock_eval_results))

        agg_df = (
            records_df.groupby("factor_name", as_index=False)
            .agg(
                ic_mean=("ic_mean", "mean"),
                abs_ic_mean=("abs_ic_mean", "mean"),
                abs_ir=("abs_ir", "mean"),
                sign_consistency=("sign_consistency", "mean"),
                regime_abs_ic_mean=("regime_abs_ic_mean", "mean"),
                regime_sign_match=("regime_sign_match", "mean"),
                stock_coverage=("stock_code", "nunique"),
            )
        )

        agg_df["coverage_ratio"] = agg_df["stock_coverage"] / valid_stock_count
        agg_df["recommended_direction"] = np.where(agg_df["ic_mean"] >= 0, 1, -1)

        agg_df["score"] = 100 * (
            0.35 * self._pct_rank(agg_df["abs_ic_mean"])
            + 0.25 * self._pct_rank(agg_df["regime_abs_ic_mean"])
            + 0.15 * self._pct_rank(agg_df["abs_ir"])
            + 0.15 * self._pct_rank(agg_df["sign_consistency"])
            + 0.10 * self._pct_rank(agg_df["coverage_ratio"])
        )
        agg_df = agg_df.sort_values("score", ascending=False).reset_index(drop=True)

        # 构建最近窗口因子矩阵，做高相关去重
        eval_frames: list[pd.DataFrame] = []
        ordered_factors = agg_df["factor_name"].tolist()
        for panel in panels.values():
            one_eval = panel[panel.index >= eval_start]
            if one_eval.empty:
                continue
            cols = [f for f in ordered_factors if f in one_eval.columns]
            if not cols:
                continue
            eval_frames.append(one_eval[cols].replace([np.inf, -np.inf], np.nan))

        if eval_frames:
            pooled = pd.concat(eval_frames, axis=0, ignore_index=True)
            corr_matrix = pooled.corr().abs()
        else:
            corr_matrix = pd.DataFrame()

        selected, redundant_map = self._corr_prune(
            candidates=ordered_factors,
            corr_matrix=corr_matrix,
            threshold=correlation_threshold,
        )
        selected_set = set(selected)

        all_factor_scores: list[dict[str, Any]] = []
        for row in agg_df.itertuples(index=False):
            redundant_info = redundant_map.get(row.factor_name)
            all_factor_scores.append(
                {
                    "factor_name": row.factor_name,
                    "score": round(float(row.score), 4),
                    "recommended_direction": int(row.recommended_direction),
                    # 语义准确名称
                    "predictive_corr_mean": round(float(row.ic_mean), 6),
                    "abs_predictive_corr_mean": round(float(row.abs_ic_mean), 6),
                    "horizon_stability": round(float(row.abs_ir), 6),
                    "regime_predictive_corr_mean": round(float(row.regime_abs_ic_mean), 6),
                    "sign_consistency": round(float(row.sign_consistency), 6),
                    "regime_sign_match": round(float(row.regime_sign_match), 6),
                    "coverage_ratio": round(float(row.coverage_ratio), 6),
                    "selected": row.factor_name in selected_set,
                    "redundant_with": redundant_info.get("redundant_with") if redundant_info else None,
                    "redundancy_correlation": (
                        round(float(redundant_info["correlation"]), 6)
                        if redundant_info
                        else None
                    ),
                    # 向后兼容别名
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

        # 生成单票榜单
        per_stock_top_factors: dict[str, dict[str, Any]] = {}
        for code, result in stock_eval_results.items():
            stock_df = pd.DataFrame(result["records"])
            if stock_df.empty:
                continue

            stock_df["stock_score"] = 100 * (
                0.55 * self._pct_rank(stock_df["abs_ic_mean"])
                + 0.25 * self._pct_rank(stock_df["regime_abs_ic_mean"])
                + 0.20 * self._pct_rank(stock_df["abs_ir"])
            )
            stock_df = stock_df.sort_values("stock_score", ascending=False).reset_index(drop=True)

            stock_top = []
            for row in stock_df.head(top_k).itertuples(index=False):
                stock_top.append(
                    {
                        "factor_name": row.factor_name,
                        "stock_score": round(float(row.stock_score), 4),
                        "recommended_direction": int(row.recommended_direction),
                        # 语义准确名称
                        "predictive_corr_mean": round(float(row.ic_mean), 6),
                        "abs_predictive_corr_mean": round(float(row.abs_ic_mean), 6),
                        "horizon_stability": round(float(row.abs_ir), 6),
                        "regime_predictive_corr_mean": round(float(row.regime_abs_ic_mean), 6),
                        # 向后兼容别名
                        "abs_ic_mean": round(float(row.abs_ic_mean), 6),
                        "ic_mean": round(float(row.ic_mean), 6),
                        "abs_ir": round(float(row.abs_ir), 6),
                        "regime_abs_ic_mean": round(float(row.regime_abs_ic_mean), 6),
                    }
                )

            per_stock_top_factors[code] = {
                "current_regime": result["current_regime"],
                "factor_count": int(len(stock_df)),
                "top_factors": stock_top,
            }

        actual_end_ts = max(panel.index.max() for panel in panels.values())
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "requested_end_date": end_date,
            "actual_end_date": actual_end_ts.strftime("%Y-%m-%d"),
            "start_date": start_date,
            "eval_start_date": eval_start.strftime("%Y-%m-%d"),
            "stock_count_requested": len(stock_codes),
            "stock_count_with_result": len(stock_eval_results),
            "factor_count_total": len(factor_defs),
            "factor_count_scored": len(all_factor_scores),
            "eval_days": eval_days,
            "lookback_days": lookback_days,
            "horizons": horizons,
            "min_samples": min_samples,
            "correlation_threshold": correlation_threshold,
            "top_k": top_k,
        }

        return {
            "metadata": metadata,
            "top_factors": top_factors,
            "all_factor_scores": all_factor_scores,
            "per_stock_top_factors": per_stock_top_factors,
        }


recent_factor_mining_service = RecentFactorMiningService()

