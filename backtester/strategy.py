"""
策略定义
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod

from utils.logger import logger
from utils.config import DEFAULT_CONFIG

class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def generate_signals(self,
                        data: pd.DataFrame,
                        date: pd.Timestamp) -> Dict[str, float]:
        """生成交易信号"""
        pass

class FactorStrategy(BaseStrategy):
    """因子策略"""

    def __init__(self,
                 name: str,
                 factor_names: List[str],
                 factor_weights: List[float] = None,
                 top_percent: float = 0.2,
                 bottom_percent: float = 0.2,
                 long_only: bool = True,
                 rebalance_frequency: str = "monthly"):
        super().__init__(name)
        self.factor_names = factor_names
        self.factor_weights = factor_weights or [1.0] * len(factor_names)
        self.top_percent = top_percent
        self.bottom_percent = bottom_percent
        self.long_only = long_only
        self.rebalance_frequency = rebalance_frequency

        if len(self.factor_names) != len(self.factor_weights):
            raise ValueError("因子数量和权重数量不匹配")

    def calculate_composite_factor(self, data: pd.DataFrame) -> pd.Series:
        """计算复合因子得分"""
        try:
            factor_scores = []
            for factor_name, weight in zip(self.factor_names, self.factor_weights):
                if factor_name in data.columns:
                    # 标准化因子值
                    factor_values = data[factor_name]
                    if factor_values.std() > 0:
                        normalized_factor = (factor_values - factor_values.mean()) / factor_values.std()
                    else:
                        normalized_factor = factor_values - factor_values.mean()

                    factor_scores.append(normalized_factor * weight)

            if factor_scores:
                composite_score = sum(factor_scores)
                return composite_score
            else:
                return pd.Series(np.nan, index=data.index)

        except Exception as e:
            logger.error(f"计算复合因子失败: {str(e)}")
            return pd.Series(np.nan, index=data.index)

    def generate_signals(self,
                        data: pd.DataFrame,
                        date: pd.Timestamp) -> Dict[str, float]:
        """生成交易信号"""
        try:
            date_data = data[data['date'] == date].copy()
            if len(date_data) < 10:  # 样本太少
                return {}

            # 计算复合因子得分
            date_data['composite_score'] = self.calculate_composite_factor(date_data)

            # 移除无效数据
            valid_data = date_data.dropna(subset=['composite_score'])
            if len(valid_data) < 5:
                return {}

            # 计算选股数量
            total_stocks = len(valid_data)
            top_n = max(1, int(total_stocks * self.top_percent))

            signals = {}

            if self.long_only:
                # 多头策略：选择因子值最高的股票
                top_stocks = valid_data.nlargest(top_n, 'composite_score')
                weight = 1.0 / len(top_stocks)
                for _, row in top_stocks.iterrows():
                    signals[row['symbol']] = weight
            else:
                # 多空策略
                bottom_n = max(1, int(total_stocks * self.bottom_percent))
                top_stocks = valid_data.nlargest(top_n, 'composite_score')
                bottom_stocks = valid_data.nsmallest(bottom_n, 'composite_score')

                # 等权重分配
                long_weight = 0.5 / len(top_stocks)
                short_weight = -0.5 / len(bottom_stocks)

                for _, row in top_stocks.iterrows():
                    signals[row['symbol']] = long_weight
                for _, row in bottom_stocks.iterrows():
                    signals[row['symbol']] = short_weight

            return signals

        except Exception as e:
            logger.error(f"生成交易信号失败: {str(e)}")
            return {}

class EqualWeightStrategy(BaseStrategy):
    """等权重策略"""

    def __init__(self, name: str, symbols: List[str]):
        super().__init__(name, "等权重策略")
        self.symbols = symbols

    def generate_signals(self,
                        data: pd.DataFrame,
                        date: pd.Timestamp) -> Dict[str, float]:
        """生成等权重信号"""
        try:
            date_data = data[data['date'] == date]
            available_symbols = set(date_data['symbol']) & set(self.symbols)

            if not available_symbols:
                return {}

            weight = 1.0 / len(available_symbols)
            return {symbol: weight for symbol in available_symbols}

        except Exception as e:
            logger.error(f"生成等权重信号失败: {str(e)}")
            return {}

class MarketCapStrategy(BaseStrategy):
    """市值加权策略"""

    def __init__(self, name: str, top_percent: float = 0.3):
        super().__init__(name, "市值加权策略")
        self.top_percent = top_percent

    def generate_signals(self,
                        data: pd.DataFrame,
                        date: pd.Timestamp) -> Dict[str, float]:
        """生成市值加权信号"""
        try:
            date_data = data[data['date'] == date].copy()

            if 'market_cap' not in date_data.columns:
                logger.warning("缺少市值数据")
                return {}

            # 按市值排序选择头部股票
            total_stocks = len(date_data)
            top_n = max(1, int(total_stocks * self.top_percent))

            top_stocks = date_data.nlargest(top_n, 'market_cap')
            total_market_cap = top_stocks['market_cap'].sum()

            signals = {}
            for _, row in top_stocks.iterrows():
                weight = row['market_cap'] / total_market_cap
                signals[row['symbol']] = weight

            return signals

        except Exception as e:
            logger.error(f"生成市值加权信号失败: {str(e)}")
            return {}

class MomentumStrategy(BaseStrategy):
    """动量策略"""

    def __init__(self,
                 name: str,
                 lookback_period: int = 20,
                 top_percent: float = 0.2):
        super().__init__(name, "动量策略")
        self.lookback_period = lookback_period
        self.top_percent = top_percent

    def calculate_momentum_score(self, data: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """计算动量得分"""
        try:
            date_data = data[data['date'] == date]
            momentum_scores = []

            for _, row in date_data.iterrows():
                symbol = row['symbol']
                symbol_data = data[data['symbol'] == symbol].sort_values('date')

                if len(symbol_data) >= self.lookback_period:
                    # 计算过去N日收益率
                    current_price = row['close']
                    past_price = symbol_data.iloc[-self.lookback_period-1]['close']
                    momentum = (current_price - past_price) / past_price
                    momentum_scores.append(momentum)
                else:
                    momentum_scores.append(np.nan)

            return pd.Series(momentum_scores, index=date_data.index)

        except Exception as e:
            logger.error(f"计算动量得分失败: {str(e)}")
            return pd.Series(np.nan, index=data.index)

    def generate_signals(self,
                        data: pd.DataFrame,
                        date: pd.Timestamp) -> Dict[str, float]:
        """生成动量交易信号"""
        try:
            date_data = data[data['date'] == date].copy()
            if len(date_data) < 10:
                return {}

            # 计算动量得分
            date_data['momentum_score'] = self.calculate_momentum_score(data, date)

            # 移除无效数据
            valid_data = date_data.dropna(subset=['momentum_score'])
            if len(valid_data) < 5:
                return {}

            # 选择动量最高的股票
            total_stocks = len(valid_data)
            top_n = max(1, int(total_stocks * self.top_percent))

            top_stocks = valid_data.nlargest(top_n, 'momentum_score')
            weight = 1.0 / len(top_stocks)

            signals = {}
            for _, row in top_stocks.iterrows():
                signals[row['symbol']] = weight

            return signals

        except Exception as e:
            logger.error(f"生成动量交易信号失败: {str(e)}")
            return {}

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""

    def __init__(self,
                 name: str,
                 lookback_period: int = 20,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5):
        super().__init__(name, "均值回归策略")
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.positions = {}  # 持仓记录

    def calculate_z_score(self, data: pd.DataFrame, symbol: str, date: pd.Timestamp) -> float:
        """计算Z得分"""
        try:
            symbol_data = data[data['symbol'] == symbol].sort_values('date')
            current_idx = symbol_data[symbol_data['date'] <= date].index

            if len(current_idx) < self.lookback_period:
                return np.nan

            lookback_data = symbol_data.iloc[-self.lookback_period:]
            prices = lookback_data['close']
            current_price = lookback_data.iloc[-1]['close']

            mean_price = prices.mean()
            std_price = prices.std()

            if std_price == 0:
                return np.nan

            z_score = (current_price - mean_price) / std_price
            return z_score

        except Exception as e:
            logger.error(f"计算Z得分失败: {str(e)}")
            return np.nan

    def generate_signals(self,
                        data: pd.DataFrame,
                        date: pd.Timestamp) -> Dict[str, float]:
        """生成均值回归信号"""
        try:
            date_data = data[data['date'] == date]
            signals = {}

            for _, row in date_data.iterrows():
                symbol = row['symbol']
                z_score = self.calculate_z_score(data, symbol, date)

                if np.isnan(z_score):
                    continue

                current_position = self.positions.get(symbol, 0)

                # 入场信号
                if current_position == 0:
                    if z_score < -self.entry_threshold:  # 超卖，买入
                        signals[symbol] = 1.0
                        self.positions[symbol] = 1.0
                    elif z_score > self.entry_threshold:  # 超买，卖出
                        signals[symbol] = -1.0
                        self.positions[symbol] = -1.0

                # 出场信号
                elif current_position > 0 and z_score > -self.exit_threshold:
                    signals[symbol] = 0.0  # 平仓
                    self.positions[symbol] = 0.0
                elif current_position < 0 and z_score < self.exit_threshold:
                    signals[symbol] = 0.0  # 平仓
                    self.positions[symbol] = 0.0
                else:
                    # 保持持仓
                    signals[symbol] = current_position

            return signals

        except Exception as e:
            logger.error(f"生成均值回归信号失败: {str(e)}")
            return {}

    def reset_positions(self):
        """重置持仓记录"""
        self.positions = {}