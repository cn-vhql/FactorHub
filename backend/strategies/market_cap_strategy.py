"""
市值加权策略 - 按市值分配权重
"""
from typing import Dict
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class MarketCapStrategy(BaseStrategy):
    """
    市值加权策略

    逻辑：
    1. 信号为1时，按市值分配权重
    2. 市值大的股票权重高
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        market_cap_column: str = "market_cap",
        **kwargs
    ):
        """
        初始化市值加权策略

        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            market_cap_column: 市值列名
        """
        super().__init__(initial_capital, commission_rate, **kwargs)
        self.market_cap_column = market_cap_column

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        逻辑：对所有有市值数据的股票生买入信号

        Args:
            df: 数据

        Returns:
            信号序列
        """
        if self.market_cap_column not in df.columns:
            raise ValueError(f"市值加权策略需要列: {self.market_cap_column}")

        signals = pd.Series(0, index=df.index)
        mask = df[self.market_cap_column].notna() & (df[self.market_cap_column] > 0)
        signals[mask] = 1
        return signals

    def calculate_weights(
        self,
        df: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
        """
        按市值计算权重

        Args:
            df: 数据（必须包含market_cap列）
            signals: 信号

        Returns:
            权重序列
        """
        weights = pd.Series(0.0, index=df.index)

        if self.market_cap_column not in df.columns:
            raise ValueError(f"市值加权策略需要列: {self.market_cap_column}")

        # 计算市值权重 - 真正的市值加权实现
        # 对每个时间点计算横截面的市值权重

        # 确定索引结构
        if df.index.nlevels == 1:
            # 单级索引（日期）
            if df.index.name == "date":
                dates = df.index.unique()
                for date in dates:
                    date_mask = (df.index == date) & (signals == 1)
                    selected_stocks = df[date_mask]

                    if len(selected_stocks) > 0:
                        total_market_cap = selected_stocks[self.market_cap_column].sum()

                        if total_market_cap > 0:
                            for idx in selected_stocks.index:
                                mcap = selected_stocks.loc[idx, self.market_cap_column]
                                if pd.notna(mcap) and mcap > 0:
                                    weights[idx] = mcap / total_market_cap
        else:
            # 多级索引（可能包含日期和其他字段）
            level_names = list(df.index.names)

            if "date" in level_names:
                # 找到日期级别
                date_level = level_names.index("date")
                dates = df.index.get_level_values(date_level).unique()

                for date in dates:
                    date_mask = (df.index.get_level_values(date_level) == date) & (signals == 1)
                    selected_stocks = df[date_mask]

                    if len(selected_stocks) > 0:
                        total_market_cap = selected_stocks[self.market_cap_column].sum()

                        if total_market_cap > 0:
                            for idx in selected_stocks.index:
                                mcap = selected_stocks.loc[idx, self.market_cap_column]
                                if pd.notna(mcap) and mcap > 0:
                                    weights[idx] = mcap / total_market_cap
            else:
                # 如果没有明确的日期级别，假设索引的第一级是日期
                dates = df.index.get_level_values(0).unique()

                for date in dates:
                    date_mask = (df.index.get_level_values(0) == date) & (signals == 1)
                    selected_stocks = df[date_mask]

                    if len(selected_stocks) > 0:
                        total_market_cap = selected_stocks[self.market_cap_column].sum()

                        if total_market_cap > 0:
                            for idx in selected_stocks.index:
                                mcap = selected_stocks.loc[idx, self.market_cap_column]
                                if pd.notna(mcap) and mcap > 0:
                                    weights[idx] = mcap / total_market_cap

        return weights
