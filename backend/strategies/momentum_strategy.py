"""
动量策略 - 买入动量强的股票
"""
from typing import Dict
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    动量策略

    逻辑：
    1. 计算过去N天的收益率
    2. 收益率 > 阈值时买入
    3. 收益率 < -阈值时卖出
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        momentum_window: int = 20,
        buy_threshold: float = 0.03,
        sell_threshold: float = -0.03,
        **kwargs
    ):
        """
        初始化动量策略

        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            momentum_window: 动量窗口（天数）
            buy_threshold: 买入阈值（收益率）
            sell_threshold: 卖出阈值（收益率）
        """
        super().__init__(initial_capital, commission_rate, **kwargs)
        self.momentum_window = momentum_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        基于动量生成交易信号

        Args:
            df: 数据（必须包含close列）

        Returns:
            信号序列
        """
        signals = pd.Series(0, index=df.index)

        # 计算动量（过去N天的收益率）
        momentum = df["close"].pct_change(self.momentum_window)

        # 生成信号
        signals[momentum > self.buy_threshold] = 1  # 买入
        signals[momentum < self.sell_threshold] = -1  # 卖出

        return signals

    def calculate_weights(
        self,
        df: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
        """
        计算权重

        信号为1时满仓，信号为-1时空仓

        Args:
            df: 数据
            signals: 信号

        Returns:
            权重序列
        """
        weights = pd.Series(0.0, index=df.index)

        # 信号为1时满仓
        weights[signals == 1] = 1.0

        # 当前策略实现为多头/空仓，不启用做空仓位
        # weights[signals == -1] = -1.0

        return weights
