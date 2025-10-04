"""
数据预处理模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from utils.logger import logger
from utils.helpers import (
    remove_outliers_3sigma, forward_fill, linear_interpolate,
    calculate_returns, format_date
)

class DataPreprocessor:
    """数据预处理器"""

    def __init__(self):
        self.logger = logger

    def adjust_price(self,
                    df: pd.DataFrame,
                    adjust_type: str = "qfq") -> pd.DataFrame:
        """复权处理"""
        try:
            if adjust_type == "hfq":  # 后复权
                # AKShare默认返回前复权数据，后复权需要特殊处理
                df = df.copy()
                # 这里简化处理，实际应该使用复权因子
                logger.warning("后复权功能需要额外复权因子数据")
                return df
            elif adjust_type == "qfq":  # 前复权
                # AKShare默认返回前复权数据
                return df
            else:
                raise ValueError(f"不支持的复权类型: {adjust_type}")

        except Exception as e:
            self.logger.error(f"复权处理失败: {str(e)}")
            return df

    def fill_missing_values(self,
                          df: pd.DataFrame,
                          method: str = "ffill") -> pd.DataFrame:
        """填充缺失值"""
        try:
            df = df.copy()

            # 按股票分组处理
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('date').set_index('date')

                # 数值列处理
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
                for col in numeric_columns:
                    if col in symbol_data.columns:
                        if method == "ffill":
                            symbol_data[col] = symbol_data[col].fillna(method='ffill')
                        elif method == "linear":
                            symbol_data[col] = symbol_data[col].interpolate(method='linear')
                        elif method == "zero":
                            symbol_data[col] = symbol_data[col].fillna(0)

                # 更新回原DataFrame
                df.loc[df['symbol'] == symbol, numeric_columns] = symbol_data[numeric_columns].values

            return df

        except Exception as e:
            self.logger.error(f"缺失值填充失败: {str(e)}")
            return df

    def remove_outliers(self,
                       df: pd.DataFrame,
                       method: str = "3sigma",
                       columns: List[str] = None) -> pd.DataFrame:
        """剔除异常值"""
        try:
            df = df.copy()

            if columns is None:
                columns = ['open', 'high', 'low', 'close', 'volume', 'amount']

            for col in columns:
                if col in df.columns:
                    if method == "3sigma":
                        # 按股票分组处理
                        for symbol in df['symbol'].unique():
                            mask = df['symbol'] == symbol
                            df.loc[mask, col] = remove_outliers_3sigma(df.loc[mask, col])
                    elif method == "quantile":
                        # 分位数方法
                        lower = df[col].quantile(0.01)
                        upper = df[col].quantile(0.99)
                        df[col] = df[col].clip(lower=lower, upper=upper)

            return df

        except Exception as e:
            self.logger.error(f"异常值处理失败: {str(e)}")
            return df

    def calculate_technical_indicators(self,
                                    df: pd.DataFrame,
                                    indicators: List[str] = None) -> pd.DataFrame:
        """计算技术指标"""
        try:
            if indicators is None:
                indicators = ['ma5', 'ma10', 'ma20', 'ma60', 'rsi', 'macd']

            result_df = df.copy()

            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('date').reset_index(drop=True)

                # 移动平均
                if 'ma5' in indicators:
                    symbol_data['ma5'] = symbol_data['close'].rolling(5).mean()
                if 'ma10' in indicators:
                    symbol_data['ma10'] = symbol_data['close'].rolling(10).mean()
                if 'ma20' in indicators:
                    symbol_data['ma20'] = symbol_data['close'].rolling(20).mean()
                if 'ma60' in indicators:
                    symbol_data['ma60'] = symbol_data['close'].rolling(60).mean()

                # RSI
                if 'rsi' in indicators:
                    symbol_data['rsi'] = self._calculate_rsi(symbol_data['close'])

                # MACD
                if 'macd' in indicators:
                    macd_df = self._calculate_macd(symbol_data['close'])
                    symbol_data = pd.concat([symbol_data, macd_df], axis=1)

                # 更新回原DataFrame
                for col in symbol_data.columns:
                    if col not in ['symbol', 'date'] + df.columns.tolist():
                        result_df.loc[result_df['symbol'] == symbol, col] = symbol_data[col].values

            return result_df

        except Exception as e:
            self.logger.error(f"技术指标计算失败: {str(e)}")
            return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series,
                       fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        })

    def normalize_data(self,
                      df: pd.DataFrame,
                      method: str = "minmax") -> pd.DataFrame:
        """数据标准化"""
        try:
            df = df.copy()
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']

            for col in numeric_columns:
                if col in df.columns:
                    if method == "minmax":
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if max_val != min_val:
                            df[col] = (df[col] - min_val) / (max_val - min_val)
                    elif method == "zscore":
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        if std_val != 0:
                            df[col] = (df[col] - mean_val) / std_val

            return df

        except Exception as e:
            self.logger.error(f"数据标准化失败: {str(e)}")
            return df

    def add_returns(self,
                   df: pd.DataFrame,
                   periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """添加收益率列"""
        try:
            df = df.copy()

            for period in periods:
                col_name = f'return_{period}d'
                if period == 1:
                    df[col_name] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change())
                else:
                    df[col_name] = df.groupby('symbol')['close'].transform(
                        lambda x: x.pct_change(period)
                    )

            return df

        except Exception as e:
            self.logger.error(f"收益率计算失败: {str(e)}")
            return df

    def filter_data(self,
                   df: pd.DataFrame,
                   start_date: str = None,
                   end_date: str = None,
                   symbols: List[str] = None) -> pd.DataFrame:
        """过滤数据"""
        try:
            result_df = df.copy()

            # 日期过滤
            if start_date:
                result_df = result_df[result_df['date'] >= start_date]
            if end_date:
                result_df = result_df[result_df['date'] <= end_date]

            # 股票过滤
            if symbols:
                result_df = result_df[result_df['symbol'].isin(symbols)]

            return result_df

        except Exception as e:
            self.logger.error(f"数据过滤失败: {str(e)}")
            return df

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """获取数据摘要信息"""
        try:
            if df.empty:
                return {"error": "数据为空"}

            summary = {
                "total_records": len(df),
                "unique_symbols": df['symbol'].nunique(),
                "date_range": {
                    "start": df['date'].min().strftime('%Y-%m-%d'),
                    "end": df['date'].max().strftime('%Y-%m-%d')
                },
                "columns": list(df.columns),
                "missing_values": df.isnull().sum().to_dict(),
                "symbols_count": df['symbol'].value_counts().head(10).to_dict()
            }

            return summary

        except Exception as e:
            self.logger.error(f"数据摘要生成失败: {str(e)}")
            return {"error": str(e)}

    def preprocess_pipeline(self,
                          df: pd.DataFrame,
                          adjust_type: str = "qfq",
                          fill_method: str = "ffill",
                          outlier_method: str = "3sigma",
                          add_returns: bool = True,
                          add_indicators: bool = False) -> pd.DataFrame:
        """预处理管道"""
        try:
            self.logger.info("开始数据预处理")

            # 复权处理
            if adjust_type != "none":
                df = self.adjust_price(df, adjust_type)

            # 缺失值填充
            df = self.fill_missing_values(df, fill_method)

            # 异常值处理
            if outlier_method != "none":
                df = self.remove_outliers(df, outlier_method)

            # 添加收益率
            if add_returns:
                df = self.add_returns(df)

            # 添加技术指标
            if add_indicators:
                df = self.calculate_technical_indicators(df)

            self.logger.info(f"数据预处理完成，处理结果: {len(df)}条记录")
            return df

        except Exception as e:
            self.logger.error(f"数据预处理失败: {str(e)}")
            return df