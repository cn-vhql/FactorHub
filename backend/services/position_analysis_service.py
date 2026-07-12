"""
持仓分析服务 - 分析持仓统计信息
"""
from typing import Dict
import pandas as pd
import numpy as np


class PositionAnalysisService:
    """持仓分析服务"""

    def __init__(self):
        pass

    def analyze_positions(
        self,
        positions: pd.Series | pd.DataFrame,
        initial_capital: float = 1000000
    ) -> Dict:
        """
        分析持仓统计信息

        Args:
            positions: 持仓序列（权重）
            initial_capital: 初始资金

        Returns:
            持仓统计信息
        """
        position_frame = self._to_position_frame(positions)
        if position_frame.empty:
            return self._empty_stats()

        gross_exposure = position_frame.abs().sum(axis=1)
        non_zero_weights = position_frame.abs().replace(0, np.nan).stack()
        per_period_turnover = position_frame.diff().abs().sum(axis=1).fillna(0.0) * 0.5

        avg_position = gross_exposure.mean()
        max_position = position_frame.abs().max(axis=1).max()
        min_position = float(non_zero_weights.min()) if not non_zero_weights.empty else 0.0

        position_zero_ratio = (gross_exposure <= 1e-12).mean()
        position_full_ratio = (gross_exposure >= 0.9).mean()

        avg_position_change = per_period_turnover.mean()
        max_position_change = per_period_turnover.max()

        is_invested = gross_exposure > 0
        invested_periods = 0
        total_invested_days = 0

        if len(is_invested) > 0:
            current_period = 0
            for invested in is_invested:
                if invested:
                    current_period += 1
                    total_invested_days += 1
                else:
                    if current_period > 0:
                        invested_periods += 1
                    current_period = 0

            if current_period > 0:
                invested_periods += 1

        avg_holding_period = (
            total_invested_days / invested_periods if invested_periods > 0 else 0
        )

        turnover = per_period_turnover.sum()

        position_values = gross_exposure * initial_capital
        avg_position_value = position_values.mean()
        max_position_value = position_values.max()

        return {
            "basic_stats": {
                "avg_position": float(avg_position),
                "max_position": float(max_position),
                "min_position": float(min_position),
                "position_zero_ratio": float(position_zero_ratio),
                "position_full_ratio": float(position_full_ratio),
            },
            "position_changes": {
                "avg_position_change": float(avg_position_change),
                "max_position_change": float(max_position_change),
            },
            "holding_stats": {
                "invested_periods": int(invested_periods),
                "total_invested_days": int(total_invested_days),
                "avg_holding_period": float(avg_holding_period),
            },
            "turnover": float(turnover),
            "position_values": {
                "avg_position_value": float(avg_position_value),
                "max_position_value": float(max_position_value),
            },
        }

    def analyze_position_history(
        self,
        positions: pd.Series | pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        分析持仓历史（滚动窗口）

        Args:
            positions: 持仓序列
            window: 窗口大小

        Returns:
            持仓历史DataFrame
        """
        position_frame = self._to_position_frame(positions)
        gross_exposure = position_frame.abs().sum(axis=1)
        df = pd.DataFrame(index=position_frame.index)
        df["position"] = gross_exposure

        df["rolling_avg_position"] = gross_exposure.rolling(window=window).mean()
        df["rolling_max_position"] = gross_exposure.rolling(window=window).max()
        df["rolling_min_position"] = gross_exposure.rolling(window=window).min()

        df["position_change"] = position_frame.diff().abs().sum(axis=1).fillna(0.0) * 0.5

        return df

    def calculate_position_concentration(
        self,
        positions: pd.Series | pd.DataFrame
    ) -> Dict:
        """
        计算持仓集中度

        Args:
            positions: 持仓序列

        Returns:
            集中度指标
        """
        snapshot = self._extract_latest_snapshot(positions)
        if snapshot.empty:
            return {
                "concentration_ratio": 0.0,
                "herfindahl_index": 0.0,
                "gini_coefficient": 0.0,
            }

        total_weight = snapshot.sum()
        if total_weight <= 0:
            return {
                "concentration_ratio": 0.0,
                "herfindahl_index": 0.0,
                "gini_coefficient": 0.0,
            }

        normalized = snapshot / total_weight
        concentration_ratio = normalized.max()
        herfindahl_index = (normalized ** 2).sum()
        gini_coefficient = self._calculate_gini(normalized.values)

        return {
            "concentration_ratio": float(concentration_ratio),
            "herfindahl_index": float(herfindahl_index),
            "gini_coefficient": float(gini_coefficient),
        }

    def _to_position_frame(self, positions: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """将不同结构的持仓输入统一为按时间索引的权重矩阵。"""
        if isinstance(positions, pd.DataFrame):
            if {"date", "stock_code", "weight"}.issubset(positions.columns):
                frame = positions.copy()
                frame["date"] = pd.to_datetime(frame["date"])
                matrix = frame.pivot_table(
                    index="date",
                    columns="stock_code",
                    values="weight",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                return matrix.sort_index()

            if {"stock_code", "weight"}.issubset(positions.columns):
                snapshot = (
                    positions.groupby("stock_code")["weight"].sum().to_frame().T
                )
                snapshot.index = pd.Index(["snapshot"])
                return snapshot.fillna(0.0)

            numeric_df = positions.select_dtypes(include=[np.number]).copy()
            return numeric_df.fillna(0.0)

        series = positions.dropna()
        if series.empty:
            return pd.DataFrame()

        if isinstance(series.index, pd.MultiIndex) and "stock_code" in series.index.names:
            stock_level = series.index.names.index("stock_code")
            if "date" in series.index.names:
                frame = (
                    series.groupby(level=list(range(series.index.nlevels))).sum()
                    .unstack(level=stock_level)
                    .fillna(0.0)
                )
                frame.columns = frame.columns.get_level_values(-1)
                return frame.sort_index()

            snapshot = series.groupby(level=stock_level).sum().to_frame().T
            snapshot.index = pd.Index(["snapshot"])
            return snapshot.fillna(0.0)

        frame = pd.DataFrame({"asset_0": series.astype(float)})
        return frame.fillna(0.0)

    def _extract_latest_snapshot(self, positions: pd.Series | pd.DataFrame) -> pd.Series:
        """提取最近一期持仓快照并聚合为绝对权重。"""
        frame = self._to_position_frame(positions)
        if frame.empty:
            return pd.Series(dtype=float)

        latest = frame.iloc[-1].abs()
        latest = latest[latest > 0]
        return latest

    def _calculate_gini(self, values: np.ndarray) -> float:
        """计算基尼系数。"""
        if len(values) == 0:
            return 0.0

        sorted_values = np.sort(values)
        cumulative = np.cumsum(sorted_values)
        total = cumulative[-1]
        if total == 0:
            return 0.0

        n = len(sorted_values)
        return float((n + 1 - 2 * np.sum(cumulative) / total) / n)

    def _empty_stats(self) -> Dict:
        """返回空的统计信息"""
        return {
            "basic_stats": {
                "avg_position": 0.0,
                "max_position": 0.0,
                "min_position": 0.0,
                "position_zero_ratio": 0.0,
                "position_full_ratio": 0.0,
            },
            "position_changes": {
                "avg_position_change": 0.0,
                "max_position_change": 0.0,
            },
            "holding_stats": {
                "invested_periods": 0,
                "total_invested_days": 0,
                "avg_holding_period": 0.0,
            },
            "turnover": 0.0,
            "position_values": {
                "avg_position_value": 0.0,
                "max_position_value": 0.0,
            },
        }


# 全局持仓分析服务实例
position_analysis_service = PositionAnalysisService()
