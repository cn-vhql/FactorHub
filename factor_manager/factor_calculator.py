"""
因子计算器
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import time

from utils.logger import logger
from utils.helpers import save_to_cache, load_from_cache
from utils.config import FACTORS_DIR
from .factor_lib import FactorLibrary
from .custom_factor import CustomFactorManager

class FactorCalculator:
    """因子计算器"""

    def __init__(self):
        self.logger = logger
        self.factor_lib = FactorLibrary()
        self.custom_manager = CustomFactorManager()
        self.cache_enabled = True
        self.parallel_enabled = True

    def calculate_single_factor(self,
                              factor_name: str,
                              data: pd.DataFrame,
                              symbol_column: str = 'symbol',
                              date_column: str = 'date',
                              use_cache: bool = True) -> pd.DataFrame:
        """计算单个因子"""
        try:
            # 检查缓存
            if use_cache and self.cache_enabled:
                cache_key = f"factor_{factor_name}_{hash(str(data.shape))}"
                cached_result = load_from_cache(cache_key)
                if cached_result is not None:
                    self.logger.info(f"从缓存加载因子: {factor_name}")
                    return cached_result

            # 计算预置因子
            if factor_name in self.factor_lib.factors:
                result = self.factor_lib.calculate_factor(
                    factor_name, data, symbol_column, date_column
                )
            # 计算自定义因子
            elif factor_name in [f.name for f in self.custom_manager.list_factors()]:
                result = self.custom_manager.execute_factor(
                    factor_name, data, symbol_column, date_column
                )
            else:
                self.logger.error(f"因子不存在: {factor_name}")
                return pd.DataFrame()

            # 缓存结果
            if use_cache and self.cache_enabled and not result.empty:
                save_to_cache(result, cache_key)

            return result

        except Exception as e:
            self.logger.error(f"计算因子{factor_name}失败: {str(e)}")
            return pd.DataFrame()

    def calculate_multiple_factors(self,
                                 factor_names: List[str],
                                 data: pd.DataFrame,
                                 symbol_column: str = 'symbol',
                                 date_column: str = 'date',
                                 use_cache: bool = True,
                                 parallel: bool = True) -> pd.DataFrame:
        """计算多个因子"""
        start_time = time.time()

        try:
            if parallel and self.parallel_enabled and len(factor_names) > 1:
                # 并行计算
                result = self._calculate_factors_parallel(
                    factor_names, data, symbol_column, date_column, use_cache
                )
            else:
                # 串行计算
                result = data.copy()
                for factor_name in factor_names:
                    factor_df = self.calculate_single_factor(
                        factor_name, data, symbol_column, date_column, use_cache
                    )
                    if not factor_df.empty and factor_name in factor_df.columns:
                        result[factor_name] = factor_df[factor_name]

            elapsed_time = time.time() - start_time
            self.logger.info(f"批量计算{len(factor_names)}个因子完成，耗时: {elapsed_time:.2f}秒")

            return result

        except Exception as e:
            self.logger.error(f"批量计算因子失败: {str(e)}")
            return data.copy()

    def _calculate_factors_parallel(self,
                                  factor_names: List[str],
                                  data: pd.DataFrame,
                                  symbol_column: str,
                                  date_column: str,
                                  use_cache: bool) -> pd.DataFrame:
        """并行计算因子"""
        result_df = data.copy()

        # 准备任务
        tasks = [
            (factor_name, data, symbol_column, date_column, use_cache)
            for factor_name in factor_names
        ]

        # 使用进程池并行计算
        max_workers = min(cpu_count(), len(factor_names))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(self._calculate_factor_task, *task)
                futures.append(future)

            # 收集结果
            for i, future in enumerate(futures):
                try:
                    factor_df = future.result()
                    if not factor_df.empty:
                        factor_name = factor_names[i]
                        if factor_name in factor_df.columns:
                            result_df[factor_name] = factor_df[factor_name]
                except Exception as e:
                    self.logger.error(f"并行计算因子失败: {str(e)}")

        return result_df

    @staticmethod
    def _calculate_factor_task(factor_name: str,
                             data: pd.DataFrame,
                             symbol_column: str,
                             date_column: str,
                             use_cache: bool) -> pd.DataFrame:
        """单个因子计算任务（用于并行处理）"""
        # 创建计算器实例
        calculator = FactorCalculator()
        return calculator.calculate_single_factor(
            factor_name, data, symbol_column, date_column, use_cache
        )

    def calculate_all_factors(self,
                            data: pd.DataFrame,
                            include_custom: bool = True,
                            symbol_column: str = 'symbol',
                            date_column: str = 'date') -> pd.DataFrame:
        """计算所有可用因子"""
        # 获取所有预置因子
        preset_factors = list(self.factor_lib.factors.keys())

        # 获取启用的自定义因子
        custom_factors = []
        if include_custom:
            custom_factors = [
                f.name for f in self.custom_manager.list_factors()
                if f.enabled
            ]

        all_factors = preset_factors + custom_factors
        self.logger.info(f"计算所有因子: {len(all_factors)}个（预置: {len(preset_factors)}, 自定义: {len(custom_factors)}）")

        return self.calculate_multiple_factors(
            all_factors, data, symbol_column, date_column
        )

    def calculate_factor_by_symbol(self,
                                  factor_name: str,
                                  data: pd.DataFrame,
                                  symbol: str,
                                  date_column: str = 'date') -> pd.Series:
        """计算单只股票的因子值"""
        symbol_data = data[data['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values(date_column)

        factor_df = self.calculate_single_factor(factor_name, symbol_data)
        if not factor_df.empty and factor_name in factor_df.columns:
            return factor_df[factor_name]
        else:
            return pd.Series()

    def get_factor_correlation(self,
                              data: pd.DataFrame,
                              factor_names: List[str] = None) -> pd.DataFrame:
        """计算因子间相关性"""
        try:
            if factor_names is None:
                # 获取所有可用因子
                factor_columns = [col for col in data.columns
                                if col not in ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            else:
                factor_columns = factor_names

            # 检查数据
            available_factors = [col for col in factor_columns if col in data.columns]
            if not available_factors:
                return pd.DataFrame()

            # 计算相关性矩阵
            factor_data = data[available_factors].dropna()
            correlation_matrix = factor_data.corr()

            self.logger.info(f"计算{len(available_factors)}个因子的相关性矩阵")
            return correlation_matrix

        except Exception as e:
            self.logger.error(f"计算因子相关性失败: {str(e)}")
            return pd.DataFrame()

    def rank_factor_values(self,
                          data: pd.DataFrame,
                          factor_name: str,
                          ascending: bool = True) -> pd.DataFrame:
        """因子值排序"""
        try:
            if factor_name not in data.columns:
                self.logger.error(f"因子不存在: {factor_name}")
                return data

            result = data.copy()
            result[f'{factor_name}_rank'] = result.groupby('date')[factor_name].rank(
                ascending=ascending, method='min'
            )

            return result

        except Exception as e:
            self.logger.error(f"因子排序失败: {str(e)}")
            return data

    def neutralize_factor(self,
                         data: pd.DataFrame,
                         factor_name: str,
                         neutralize_factors: List[str] = None) -> pd.DataFrame:
        """因子中性化"""
        try:
            if factor_name not in data.columns:
                self.logger.error(f"因子不存在: {factor_name}")
                return data

            result = data.copy()
            neutralized_values = []

            for date in data['date'].unique():
                date_data = data[data['date'] == date].copy()

                if len(date_data) < 3:  # 样本太少
                    neutralized_values.extend(date_data[factor_name].values)
                    continue

                # 准备中性化因子
                if neutralize_factors is None:
                    X_features = []
                else:
                    X_features = [f for f in neutralize_factors if f in date_data.columns]

                if X_features:
                    X = date_data[X_features].values
                    y = date_data[factor_name].values

                    # 线性回归中性化
                    try:
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        residual = y - y_pred
                        neutralized_values.extend(residual)
                    except:
                        neutralized_values.extend(date_data[factor_name].values)
                else:
                    neutralized_values.extend(date_data[factor_name].values)

            result[f'{factor_name}_neutralized'] = neutralized_values
            return result

        except Exception as e:
            self.logger.error(f"因子中性化失败: {str(e)}")
            return data

    def calculate_factor_summary(self,
                                data: pd.DataFrame,
                                factor_name: str) -> Dict:
        """计算因子摘要统计"""
        try:
            if factor_name not in data.columns:
                return {"error": "因子不存在"}

            factor_values = data[factor_name].dropna()
            if len(factor_values) == 0:
                return {"error": "无有效因子值"}

            summary = {
                "factor_name": factor_name,
                "total_count": len(data),
                "valid_count": len(factor_values),
                "missing_count": len(data) - len(factor_values),
                "missing_rate": (len(data) - len(factor_values)) / len(data),
                "mean": factor_values.mean(),
                "std": factor_values.std(),
                "min": factor_values.min(),
                "max": factor_values.max(),
                "median": factor_values.median(),
                "q25": factor_values.quantile(0.25),
                "q75": factor_values.quantile(0.75),
                "skewness": factor_values.skew(),
                "kurtosis": factor_values.kurtosis(),
                "date_range": {
                    "start": data['date'].min().strftime('%Y-%m-%d'),
                    "end": data['date'].max().strftime('%Y-%m-%d')
                },
                "symbol_count": data['symbol'].nunique()
            }

            return summary

        except Exception as e:
            self.logger.error(f"计算因子摘要失败: {str(e)}")
            return {"error": str(e)}

    def get_factor_summary(self, data: pd.DataFrame, factor_name: str) -> Dict:
        """获取因子摘要统计 - 兼容性方法"""
        return self.calculate_factor_summary(data, factor_name)