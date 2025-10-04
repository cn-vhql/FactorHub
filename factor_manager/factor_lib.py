"""
预置因子库
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod

from utils.logger import logger
from utils.config import PRESET_FACTORS

class BaseFactor(ABC):
    """因子基类"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        pass

class TrendFactor(BaseFactor):
    """趋势类因子"""

    def __init__(self, name: str, description: str, ma_period: int):
        super().__init__(name, description)
        self.ma_period = ma_period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算移动平均因子"""
        return data['close'].rolling(self.ma_period).mean()

class MomentumFactor(BaseFactor):
    """动量类因子"""

    def __init__(self, name: str, description: str, period: int = 12):
        super().__init__(name, description)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算动量因子"""
        return data['close'].pct_change(self.period)

class RSI(BaseFactor):
    """相对强弱指标"""

    def __init__(self, name: str = "RSI", description: str = "相对强弱指标", period: int = 14):
        super().__init__(name, description)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class VolatilityFactor(BaseFactor):
    """波动率因子"""

    def __init__(self, name: str, description: str, period: int = 20):
        super().__init__(name, description)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算波动率因子"""
        return data['close'].rolling(self.period).std()

class VolumeFactor(BaseFactor):
    """成交量因子"""

    def __init__(self, name: str, description: str):
        super().__init__(name, description)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算成交量因子"""
        # 成交量标准化
        volume_ma = data['volume'].rolling(20).mean()
        return (data['volume'] - volume_ma) / volume_ma

class ValueFactor(BaseFactor):
    """估值因子"""

    def __init__(self, name: str, description: str):
        super().__init__(name, description)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算估值因子"""
        # 这里需要外部提供PE、PB等基本面数据
        # 暂时使用价格作为替代
        return data['close']

class MACD(BaseFactor):
    """MACD指标"""

    def __init__(self, name: str = "MACD", description: str = "MACD指标"):
        super().__init__(name, description)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算MACD"""
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        macd_line = ema12 - ema26
        return macd_line

class BollingerBands(BaseFactor):
    """布林带因子"""

    def __init__(self, name: str = "BB", description: str = "布林带"):
        super().__init__(name, description)
        self.period = 20
        self.std_dev = 2

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算布林带因子"""
        ma = data['close'].rolling(self.period).mean()
        std = data['close'].rolling(self.period).std()
        upper_band = ma + (std * self.std_dev)
        lower_band = ma - (std * self.std_dev)
        # 使用价格在布林带中的位置作为因子
        return (data['close'] - lower_band) / (upper_band - lower_band)

class FactorLibrary:
    """因子库管理器"""

    def __init__(self):
        self.logger = logger
        self.factors = {}
        self._initialize_preset_factors()

    def _initialize_preset_factors(self):
        """初始化预置因子"""
        # 趋势类因子
        self.factors['MA5'] = TrendFactor("MA5", "5日移动平均线", 5)
        self.factors['MA10'] = TrendFactor("MA10", "10日移动平均线", 10)
        self.factors['MA20'] = TrendFactor("MA20", "20日移动平均线", 20)
        self.factors['MA60'] = TrendFactor("MA60", "60日移动平均线", 60)

        # 动量类因子
        self.factors['MOM12'] = MomentumFactor("MOM12", "12日动量", 12)
        self.factors['MOM20'] = MomentumFactor("MOM20", "20日动量", 20)

        # RSI
        self.factors['RSI'] = RSI()

        # 波动率类因子
        self.factors['STD20'] = VolatilityFactor("STD20", "20日波动率", 20)
        self.factors['VOL'] = VolumeFactor("VOL", "成交量波动率")

        # MACD
        self.factors['MACD'] = MACD()

        # 布林带
        self.factors['BB'] = BollingerBands()

        self.logger.info(f"初始化预置因子库: {len(self.factors)}个因子")

    def get_factor(self, factor_name: str) -> Optional[BaseFactor]:
        """获取因子"""
        return self.factors.get(factor_name)

    def list_factors(self, category: str = None) -> Dict[str, Dict]:
        """列出因子"""
        if category is None:
            return {name: {"name": factor.name, "description": factor.description}
                   for name, factor in self.factors.items()}
        else:
            # 按类别筛选
            category_factors = {}
            for name, factor in self.factors.items():
                if self._get_factor_category(name) == category:
                    category_factors[name] = {"name": factor.name, "description": factor.description}
            return category_factors

    def _get_factor_category(self, factor_name: str) -> str:
        """获取因子类别"""
        if factor_name.startswith('MA'):
            return "trend"
        elif factor_name.startswith('MOM'):
            return "momentum"
        elif factor_name == 'RSI':
            return "momentum"
        elif factor_name in ['STD20', 'VOL']:
            return "volatility"
        elif factor_name == 'MACD':
            return "trend"
        elif factor_name == 'BB':
            return "volatility"
        else:
            return "other"

    def add_factor(self, factor: BaseFactor):
        """添加因子"""
        self.factors[factor.name] = factor
        self.logger.info(f"添加因子: {factor.name}")

    def remove_factor(self, factor_name: str):
        """移除因子"""
        if factor_name in self.factors:
            del self.factors[factor_name]
            self.logger.info(f"移除因子: {factor_name}")

    def calculate_factor(self,
                        factor_name: str,
                        data: pd.DataFrame,
                        symbol_column: str = 'symbol',
                        date_column: str = 'date') -> pd.DataFrame:
        """计算因子值"""
        if factor_name not in self.factors:
            self.logger.error(f"因子不存在: {factor_name}")
            return pd.DataFrame()

        factor = self.factors[factor_name]
        result_data = []

        # 按股票分组计算
        for symbol in data[symbol_column].unique():
            symbol_data = data[data[symbol_column] == symbol].copy()
            symbol_data = symbol_data.sort_values(date_column)

            try:
                factor_values = factor.calculate(symbol_data)
                if isinstance(factor_values, pd.Series):
                    symbol_data[factor_name] = factor_values.values
                else:
                    symbol_data[factor_name] = factor_values

                result_data.append(symbol_data)

            except Exception as e:
                self.logger.error(f"计算股票{symbol}的因子{factor_name}失败: {str(e)}")
                continue

        if result_data:
            result_df = pd.concat(result_data, ignore_index=True)
            self.logger.info(f"因子{factor_name}计算完成")
            return result_df
        else:
            return pd.DataFrame()

    def get_factor_categories(self) -> List[str]:
        """获取因子类别列表"""
        categories = set()
        for name in self.factors.keys():
            categories.add(self._get_factor_category(name))
        return list(categories)

    def batch_calculate_factors(self,
                               factor_names: List[str],
                               data: pd.DataFrame,
                               symbol_column: str = 'symbol',
                               date_column: str = 'date') -> pd.DataFrame:
        """批量计算多个因子"""
        result_df = data.copy()

        for factor_name in factor_names:
            if factor_name in self.factors:
                factor_df = self.calculate_factor(factor_name, data, symbol_column, date_column)
                if not factor_df.empty and factor_name in factor_df.columns:
                    result_df[factor_name] = factor_df[factor_name]

        return result_df

    def get_factor_statistics(self, factor_name: str, data: pd.DataFrame) -> Dict:
        """获取因子统计信息"""
        if factor_name not in self.factors:
            return {"error": "因子不存在"}

        factor_df = self.calculate_factor(factor_name, data)
        if factor_df.empty or factor_name not in factor_df.columns:
            return {"error": "因子计算失败"}

        factor_values = factor_df[factor_name].dropna()
        if len(factor_values) == 0:
            return {"error": "无有效因子值"}

        stats = {
            "count": len(factor_values),
            "mean": factor_values.mean(),
            "std": factor_values.std(),
            "min": factor_values.min(),
            "max": factor_values.max(),
            "median": factor_values.median(),
            "q25": factor_values.quantile(0.25),
            "q75": factor_values.quantile(0.75),
            "missing_count": len(factor_df) - len(factor_values),
            "missing_rate": (len(factor_df) - len(factor_values)) / len(factor_df)
        }

        return stats