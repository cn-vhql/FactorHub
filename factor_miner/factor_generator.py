"""
因子生成器
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import random
import itertools
from abc import ABC, abstractmethod

from utils.logger import logger
from utils.config import DEFAULT_CONFIG
from factor_manager.factor_calculator import FactorCalculator

class FactorOperator(ABC):
    """因子操作基类"""

    def __init__(self, name: str, symbol: str, description: str = ""):
        self.name = name
        self.symbol = symbol
        self.description = description

    @abstractmethod
    def apply(self, *factors: pd.Series) -> pd.Series:
        """应用操作符"""
        pass

class AddOperator(FactorOperator):
    """加法操作符"""

    def __init__(self):
        super().__init__("加法", "+", "两个因子相加")

    def apply(self, factor1: pd.Series, factor2: pd.Series) -> pd.Series:
        return factor1 + factor2

class SubtractOperator(FactorOperator):
    """减法操作符"""

    def __init__(self):
        super().__init__("减法", "-", "两个因子相减")

    def apply(self, factor1: pd.Series, factor2: pd.Series) -> pd.Series:
        return factor1 - factor2

class MultiplyOperator(FactorOperator):
    """乘法操作符"""

    def __init__(self):
        super().__init__("乘法", "*", "两个因子相乘")

    def apply(self, factor1: pd.Series, factor2: pd.Series) -> pd.Series:
        return factor1 * factor2

class DivideOperator(FactorOperator):
    """除法操作符"""

    def __init__(self):
        super().__init__("除法", "/", "两个因子相除")

    def apply(self, factor1: pd.Series, factor2: pd.Series) -> pd.Series:
        # 避免除零错误
        result = factor1 / factor2.replace(0, np.nan)
        return result

class PowerOperator(FactorOperator):
    """幂运算操作符"""

    def __init__(self, power: float = 2):
        super().__init__(f"幂运算{power}", f"^{power}", f"因子{power}次方")
        self.power = power

    def apply(self, factor: pd.Series) -> pd.Series:
        return factor ** self.power

class LogOperator(FactorOperator):
    """对数操作符"""

    def __init__(self):
        super().__init__("对数", "log", "取自然对数")

    def apply(self, factor: pd.Series) -> pd.Series:
        # 避免对负数和零取对数
        return np.log(factor.replace([0, -np.inf], np.nan).clip(lower=1e-10))

class RankOperator(FactorOperator):
    """排名操作符"""

    def __init__(self):
        super().__init__("排名", "rank", "因子值排名")

    def apply(self, factor: pd.Series) -> pd.Series:
        return factor.rank(pct=True)

class MovingAverageOperator(FactorOperator):
    """移动平均操作符"""

    def __init__(self, window: int = 5):
        super().__init__(f"移动平均{window}", f"ma{window}", f"{window}日移动平均")
        self.window = window

    def apply(self, factor: pd.Series) -> pd.Series:
        return factor.rolling(window=self.window).mean()

class StdOperator(FactorOperator):
    """标准差操作符"""

    def __init__(self, window: int = 20):
        super().__init__(f"标准差{window}", f"std{window}", f"{window}日标准差")
        self.window = window

    def apply(self, factor: pd.Series) -> pd.Series:
        return factor.rolling(window=self.window).std()

class FactorGenerator:
    """因子生成器"""

    def __init__(self):
        self.logger = logger
        self.factor_calculator = FactorCalculator()
        self.operators = self._initialize_operators()
        self.base_factors = ['close', 'volume', 'high', 'low', 'open']

    def _initialize_operators(self) -> Dict[str, FactorOperator]:
        """初始化操作符"""
        return {
            'add': AddOperator(),
            'subtract': SubtractOperator(),
            'multiply': MultiplyOperator(),
            'divide': DivideOperator(),
            'power_2': PowerOperator(2),
            'power_3': PowerOperator(3),
            'log': LogOperator(),
            'rank': RankOperator(),
            'ma_5': MovingAverageOperator(5),
            'ma_10': MovingAverageOperator(10),
            'ma_20': MovingAverageOperator(20),
            'std_10': StdOperator(10),
            'std_20': StdOperator(20)
        }

    def generate_simple_factors(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """生成简单因子"""
        try:
            simple_factors = {}

            # 价格相关因子
            if 'close' in data.columns:
                simple_factors['close_ma5'] = data.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
                simple_factors['close_ma10'] = data.groupby('symbol')['close'].transform(lambda x: x.rolling(10).mean())
                simple_factors['close_ma20'] = data.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
                simple_factors['close_std20'] = data.groupby('symbol')['close'].transform(lambda x: x.rolling(20).std())

                # 价格变化率
                simple_factors['close_return_1d'] = data.groupby('symbol')['close'].transform(lambda x: x.pct_change())
                simple_factors['close_return_5d'] = data.groupby('symbol')['close'].transform(lambda x: x.pct_change(5))
                simple_factors['close_return_20d'] = data.groupby('symbol')['close'].transform(lambda x: x.pct_change(20))

                # 价格相对位置
                simple_factors['close_position_20d'] = data.groupby('symbol')['close'].transform(
                    lambda x: (x - x.rolling(20).min()) / (x.rolling(20).max() - x.rolling(20).min())
                )

            # 成交量相关因子
            if 'volume' in data.columns:
                simple_factors['volume_ma5'] = data.groupby('symbol')['volume'].transform(lambda x: x.rolling(5).mean())
                simple_factors['volume_std20'] = data.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).std())

                # 成交量变化率
                simple_factors['volume_ratio'] = data.groupby('symbol')['volume'].transform(
                    lambda x: x / x.rolling(20).mean()
                )

                # 价量关系
                if 'close' in data.columns:
                    simple_factors['price_volume'] = data.groupby('symbol').apply(
                        lambda x: x['close'] * x['volume']
                    ).reset_index(level=0, drop=True)

            # 高低价差因子
            if 'high' in data.columns and 'low' in data.columns:
                simple_factors['high_low_ratio'] = data.groupby('symbol').apply(
                    lambda x: (x['high'] - x['low']) / x['low']
                ).reset_index(level=0, drop=True)

                simple_factors['high_low_range_20d'] = data.groupby('symbol').apply(
                    lambda x: (x['high'].rolling(20).max() - x['low'].rolling(20).min()) / x['close']
                ).reset_index(level=0, drop=True)

            # 开盘收盘价关系
            if 'open' in data.columns and 'close' in data.columns:
                simple_factors['open_close_gap'] = data.groupby('symbol').apply(
                    lambda x: (x['open'] - x['close'].shift(1)) / x['close'].shift(1)
                ).reset_index(level=0, drop=True)

            self.logger.info(f"生成简单因子: {len(simple_factors)}个")
            return simple_factors

        except Exception as e:
            self.logger.error(f"生成简单因子失败: {str(e)}")
            return {}

    def generate_combination_factors(self,
                                   data: pd.DataFrame,
                                   base_factor_names: List[str],
                                   max_complexity: int = 3,
                                   max_combinations: int = 100) -> Dict[str, pd.Series]:
        """生成组合因子"""
        try:
            if not base_factor_names:
                return {}

            # 确保基础因子存在
            available_factors = [f for f in base_factor_names if f in data.columns]
            if len(available_factors) < 2:
                return {}

            combination_factors = {}
            combination_count = 0

            # 生成二元组合
            binary_operators = ['add', 'subtract', 'multiply', 'divide']
            for op_name in binary_operators:
                if combination_count >= max_combinations:
                    break

                operator = self.operators[op_name]
                for factor1, factor2 in itertools.combinations(available_factors, 2):
                    if combination_count >= max_combinations:
                        break

                    try:
                        factor1_data = data[factor1]
                        factor2_data = data[factor2]

                        if op_name == 'divide':
                            # 对于除法，检查分母的有效性
                            if (factor2_data == 0).any():
                                continue

                        combined_factor = operator.apply(factor1_data, factor2_data)
                        factor_name = f"{factor1}_{operator.symbol}_{factor2}"

                        # 检查因子质量
                        if self._validate_factor(combined_factor):
                            combination_factors[factor_name] = combined_factor
                            combination_count += 1

                    except Exception as e:
                        self.logger.debug(f"生成组合因子失败: {factor1} {operator.symbol} {factor2}, 错误: {str(e)}")
                        continue

            # 生成一元操作组合
            unary_operators = ['power_2', 'power_3', 'log', 'rank', 'ma_5', 'ma_10', 'std_10', 'std_20']
            for factor_name in available_factors:
                if combination_count >= max_combinations:
                    break

                factor_data = data[factor_name]

                for op_name in unary_operators:
                    if combination_count >= max_combinations:
                        break

                    try:
                        operator = self.operators[op_name]
                        if op_name in ['log'] and (factor_data <= 0).any():
                            continue

                        transformed_factor = operator.apply(factor_data)
                        new_factor_name = f"{op_name}_{factor_name}"

                        if self._validate_factor(transformed_factor):
                            combination_factors[new_factor_name] = transformed_factor
                            combination_count += 1

                    except Exception as e:
                        self.logger.debug(f"生成变换因子失败: {op_name}({factor_name}), 错误: {str(e)}")
                        continue

            self.logger.info(f"生成组合因子: {len(combination_factors)}个")
            return combination_factors

        except Exception as e:
            self.logger.error(f"生成组合因子失败: {str(e)}")
            return {}

    def generate_cross_sectional_factors(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """生成截面因子"""
        try:
            cross_sectional_factors = {}

            # 截面排名因子
            numeric_columns = [col for col in data.columns if col not in ['symbol', 'date'] and
                             data[col].dtype in ['float64', 'int64']]

            for col in numeric_columns:
                try:
                    rank_factor = data.groupby('date')[col].rank(pct=True)
                    factor_name = f"{col}_rank"
                    cross_sectional_factors[factor_name] = rank_factor

                    # 截面标准化
                    z_score_factor = data.groupby('date')[col].transform(
                        lambda x: (x - x.mean()) / x.std()
                    )
                    z_score_name = f"{col}_zscore"
                    cross_sectional_factors[z_score_name] = z_score_factor

                except Exception as e:
                    self.logger.debug(f"生成截面因子失败: {col}, 错误: {str(e)}")
                    continue

            # 行业中性化因子（如果有行业信息）
            if 'industry' in data.columns:
                for col in numeric_columns:
                    try:
                        # 行业内排名
                        industry_rank = data.groupby(['date', 'industry'])[col].rank(pct=True)
                        industry_rank.name = f"{col}_industry_rank"
                        cross_sectional_factors[industry_rank.name] = industry_rank

                    except Exception as e:
                        self.logger.debug(f"生成行业中性因子失败: {col}, 错误: {str(e)}")
                        continue

            self.logger.info(f"生成截面因子: {len(cross_sectional_factors)}个")
            return cross_sectional_factors

        except Exception as e:
            self.logger.error(f"生成截面因子失败: {str(e)}")
            return {}

    def generate_technical_factors(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """生成技术因子"""
        try:
            technical_factors = {}

            # RSI
            if 'close' in data.columns:
                def calculate_rsi(x, period=14):
                    delta = x.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    return 100 - (100 / (1 + rs))

                technical_factors['rsi_14'] = data.groupby('symbol')['close'].transform(calculate_rsi)

                # MACD
                def calculate_macd(x, fast=12, slow=26, signal=9):
                    ema_fast = x.ewm(span=fast).mean()
                    ema_slow = x.ewm(span=slow).mean()
                    macd_line = ema_fast - ema_slow
                    signal_line = macd_line.ewm(span=signal).mean()
                    return macd_line - signal_line

                technical_factors['macd'] = data.groupby('symbol')['close'].transform(calculate_macd)

                # 布林带
                def calculate_bollinger_bands(x, period=20, std_dev=2):
                    ma = x.rolling(period).mean()
                    std = x.rolling(period).std()
                    upper_band = ma + (std * std_dev)
                    lower_band = ma - (std * std_dev)
                    return (x - lower_band) / (upper_band - lower_band)

                technical_factors['bollinger_position'] = data.groupby('symbol')['close'].transform(calculate_bollinger_bands)

                # 动量指标
                technical_factors['momentum_5d'] = data.groupby('symbol')['close'].transform(lambda x: x.pct_change(5))
                technical_factors['momentum_20d'] = data.groupby('symbol')['close'].transform(lambda x: x.pct_change(20))

            # 成交量指标
            if 'volume' in data.columns and 'close' in data.columns:
                # 量价相关性
                def price_volume_corr(x, window=20):
                    return x['close'].rolling(window).corr(x['volume'])

                technical_factors['price_volume_corr'] = data.groupby('symbol').apply(
                    lambda x: price_volume_corr(x)
                ).reset_index(level=0, drop=True)

                # OBV (On Balance Volume)
                def calculate_obv(x):
                    price_change = x['close'].diff()
                    obv = (price_change > 0) * x['volume'] - (price_change < 0) * x['volume']
                    return obv.cumsum()

                technical_factors['obv'] = data.groupby('symbol').apply(calculate_obv).reset_index(level=0, drop=True)

            self.logger.info(f"生成技术因子: {len(technical_factors)}个")
            return technical_factors

        except Exception as e:
            self.logger.error(f"生成技术因子失败: {str(e)}")
            return {}

    def _validate_factor(self, factor: pd.Series) -> bool:
        """验证因子质量"""
        try:
            # 检查是否全为NaN
            if factor.isna().all():
                return False

            # 检查是否全为常数
            if factor.std() == 0:
                return False

            # 检查有效值比例
            valid_ratio = factor.notna().sum() / len(factor)
            if valid_ratio < 0.5:  # 有效值比例低于50%
                return False

            # 检查极值比例
            if isinstance(factor, pd.Series):
                extreme_values = (np.abs(factor) > 10 * factor.std()).sum()
                extreme_ratio = extreme_values / len(factor)
                if extreme_ratio > 0.1:  # 极值比例超过10%
                    return False

            return True

        except Exception as e:
            self.logger.debug(f"因子验证失败: {str(e)}")
            return False

    def generate_all_factors(self,
                           data: pd.DataFrame,
                           include_simple: bool = True,
                           include_combination: bool = True,
                           include_cross_sectional: bool = True,
                           include_technical: bool = True,
                           max_complexity: int = 3,
                           max_combinations: int = 200) -> Dict[str, pd.Series]:
        """生成所有类型因子"""
        try:
            self.logger.info("开始生成因子")

            all_factors = {}

            # 简单因子
            if include_simple:
                simple_factors = self.generate_simple_factors(data)
                all_factors.update(simple_factors)

            # 组合因子
            if include_combination:
                base_factor_names = list(all_factors.keys())[:10]  # 使用前10个因子作为基础
                if len(base_factor_names) >= 2:
                    combination_factors = self.generate_combination_factors(
                        data, base_factor_names, max_complexity, max_combinations
                    )
                    all_factors.update(combination_factors)

            # 截面因子
            if include_cross_sectional:
                cross_sectional_factors = self.generate_cross_sectional_factors(data)
                all_factors.update(cross_sectional_factors)

            # 技术因子
            if include_technical:
                technical_factors = self.generate_technical_factors(data)
                all_factors.update(technical_factors)

            self.logger.info(f"总计生成因子: {len(all_factors)}个")
            return all_factors

        except Exception as e:
            self.logger.error(f"生成因子失败: {str(e)}")
            return {}

    def evaluate_factors(self,
                        factors: Dict[str, pd.Series],
                        returns: pd.Series,
                        method: str = 'ic') -> List[Tuple[str, float]]:
        """评估因子有效性"""
        try:
            evaluation_results = []

            for factor_name, factor_values in factors.items():
                try:
                    # 对齐数据
                    aligned_data = pd.DataFrame({
                        'factor': factor_values,
                        'returns': returns
                    }).dropna()

                    if len(aligned_data) < 100:  # 样本太少
                        continue

                    if method == 'ic':
                        # 计算IC值
                        ic = aligned_data['factor'].corr(aligned_data['returns'], method='spearman')
                        if not np.isnan(ic):
                            evaluation_results.append((factor_name, abs(ic)))
                    elif method == 'sharpe':
                        # 计算分层收益的夏普比率
                        # 这里简化处理，实际应该计算分层收益
                        factor_rank = aligned_data['factor'].rank(pct=True)
                        long_returns = aligned_data[factor_rank > 0.8]['returns'].mean()
                        short_returns = aligned_data[factor_rank < 0.2]['returns'].mean()
                        long_short_return = long_returns - short_returns
                        sharpe = long_short_return / aligned_data['returns'].std() if aligned_data['returns'].std() > 0 else 0
                        if not np.isnan(sharpe):
                            evaluation_results.append((factor_name, abs(sharpe)))

                except Exception as e:
                    self.logger.debug(f"评估因子{factor_name}失败: {str(e)}")
                    continue

            # 按有效性排序
            evaluation_results.sort(key=lambda x: x[1], reverse=True)
            self.logger.info(f"评估因子完成，有效因子: {len(evaluation_results)}个")

            return evaluation_results

        except Exception as e:
            self.logger.error(f"评估因子失败: {str(e)}")
            return []