"""
因子分析器
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from utils.logger import logger
from utils.helpers import calculate_ic, calculate_ir, calculate_ic_win_rate, format_number
from utils.config import DEFAULT_CONFIG

class FactorAnalyzer:
    """因子分析器"""

    def __init__(self):
        self.logger = logger
        self.default_config = DEFAULT_CONFIG['analysis']

    def calculate_ic_analysis(self,
                            factor_data: pd.DataFrame,
                            factor_name: str,
                            return_col: str = 'return_1d',
                            method: str = 'spearman',
                            window: int = None) -> Dict:
        """计算IC分析"""
        try:
            if factor_name not in factor_data.columns:
                return {"error": f"因子{factor_name}不存在"}

            if return_col not in factor_data.columns:
                return {"error": f"收益率列{return_col}不存在"}

            # 按日期分组计算IC
            ic_results = []
            dates = factor_data['date'].unique()

            for date in dates:
                date_data = factor_data[factor_data['date'] == date]
                if len(date_data) < 5:  # 样本太少
                    continue

                factor_values = date_data[factor_name]
                return_values = date_data[return_col]

                # 移除缺失值
                valid_mask = ~(pd.isna(factor_values) | pd.isna(return_values))
                if valid_mask.sum() < 3:
                    continue

                factor_clean = factor_values[valid_mask]
                return_clean = return_values[valid_mask]

                # 计算IC
                if method == 'spearman':
                    ic, p_value = stats.spearmanr(factor_clean, return_clean)
                elif method == 'pearson':
                    ic, p_value = stats.pearsonr(factor_clean, return_clean)
                else:
                    raise ValueError("method must be 'spearman' or 'pearson'")

                if not np.isnan(ic):
                    ic_results.append({
                        'date': date,
                        'ic': ic,
                        'p_value': p_value,
                        'count': valid_mask.sum()
                    })

            if not ic_results:
                return {"error": "没有有效的IC计算结果"}

            ic_df = pd.DataFrame(ic_results)
            ic_series = ic_df['ic']

            # 计算IC统计指标
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ir = calculate_ir(ic_series)
            ic_win_rate = calculate_ic_win_rate(ic_series)
            ic_abs_mean = np.abs(ic_series).mean()

            # 滚动IC分析
            rolling_stats = {}
            if window and len(ic_series) >= window:
                rolling_mean = ic_series.rolling(window).mean()
                rolling_std = ic_series.rolling(window).std()
                rolling_ir = rolling_mean / rolling_std

                rolling_stats = {
                    'rolling_mean_last': rolling_mean.iloc[-1],
                    'rolling_std_last': rolling_std.iloc[-1],
                    'rolling_ir_last': rolling_ir.iloc[-1]
                }

            # t检验
            t_stat, t_p_value = stats.ttest_1samp(ic_series, 0)

            results = {
                'factor_name': factor_name,
                'ic_series': ic_series,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ir': ir,
                'ic_win_rate': ic_win_rate,
                'ic_abs_mean': ic_abs_mean,
                'ic_max': ic_series.max(),
                'ic_min': ic_series.min(),
                'total_periods': len(ic_series),
                't_statistic': t_stat,
                't_p_value': t_p_value,
                'significant_periods': (ic_df['p_value'] < 0.05).sum(),
                'method': method,
                'rolling_stats': rolling_stats
            }

            self.logger.info(f"因子{factor_name} IC分析完成")
            return results

        except Exception as e:
            self.logger.error(f"IC分析失败: {str(e)}")
            return {"error": str(e)}

    def calculate_layered_returns(self,
                                 factor_data: pd.DataFrame,
                                 factor_name: str,
                                 return_col: str = 'return_1d',
                                 layers: int = 5,
                                 quantiles: List[float] = None) -> Dict:
        """计算分层收益"""
        try:
            if factor_name not in factor_data.columns:
                return {"error": f"因子{factor_name}不存在"}

            if return_col not in factor_data.columns:
                return {"error": f"收益率列{return_col}不存在"}

            if quantiles is None:
                quantiles = [i/layers for i in range(1, layers)]

            layer_results = []
            dates = factor_data['date'].unique()

            for date in dates:
                date_data = factor_data[factor_data['date'] == date].copy()

                if len(date_data) < layers * 2:  # 样本太少
                    continue

                factor_values = date_data[factor_name]
                return_values = date_data[return_col]

                # 移除缺失值
                valid_mask = ~(pd.isna(factor_values) | pd.isna(return_values))
                if valid_mask.sum() < layers * 2:
                    continue

                factor_clean = factor_values[valid_mask]
                return_clean = return_values[valid_mask]

                # 计算分位数
                factor_quantiles = factor_clean.quantile(quantiles)

                # 分层
                layer_returns = {}
                layer_counts = {}

                for i, q in enumerate(quantiles):
                    if i == 0:
                        mask = factor_clean <= q
                    else:
                        mask = (factor_clean > factor_quantiles.iloc[i-1]) & (factor_clean <= q)

                    layer_returns[f'layer_{i+1}'] = return_clean[mask].mean()
                    layer_counts[f'layer_{i+1}'] = mask.sum()

                # 最后一层
                mask = factor_clean > factor_quantiles.iloc[-1]
                layer_returns[f'layer_{layers}'] = return_clean[mask].mean()
                layer_counts[f'layer_{layers}'] = mask.sum()

                # 计算多空收益
                long_short = layer_returns[f'layer_{layers}'] - layer_returns['layer_1']

                result = {
                    'date': date,
                    **layer_returns,
                    'long_short': long_short,
                    **{f'count_{k.replace("layer_", "")}': v for k, v in layer_counts.items()}
                }

                layer_results.append(result)

            if not layer_results:
                return {"error": "没有有效的分层收益计算结果"}

            layer_df = pd.DataFrame(layer_results)

            # 计算各层累计收益和统计指标
            summary = {}
            for i in range(1, layers + 1):
                layer_col = f'layer_{i}'
                if layer_col in layer_df.columns:
                    returns = layer_df[layer_col]
                    summary[f'layer_{i}'] = {
                        'cumulative_return': (1 + returns).prod() - 1,
                        'mean_return': returns.mean(),
                        'std_return': returns.std(),
                        'sharpe': returns.mean() / returns.std() * np.sqrt(252),
                        'win_rate': (returns > 0).sum() / len(returns),
                        'periods': len(returns)
                    }

            # 多空组合统计
            if 'long_short' in layer_df.columns:
                ls_returns = layer_df['long_short']
                summary['long_short'] = {
                    'cumulative_return': (1 + ls_returns).prod() - 1,
                    'mean_return': ls_returns.mean(),
                    'std_return': ls_returns.std(),
                    'sharpe': ls_returns.mean() / ls_returns.std() * np.sqrt(252),
                    'win_rate': (ls_returns > 0).sum() / len(ls_returns),
                    'periods': len(ls_returns)
                }

            results = {
                'factor_name': factor_name,
                'layers': layers,
                'layer_data': layer_df,
                'summary': summary,
                'date_range': {
                    'start': layer_df['date'].min(),
                    'end': layer_df['date'].max()
                }
            }

            self.logger.info(f"因子{factor_name}分层收益分析完成")
            return results

        except Exception as e:
            self.logger.error(f"分层收益分析失败: {str(e)}")
            return {"error": str(e)}

    def calculate_turnover_analysis(self,
                                   factor_data: pd.DataFrame,
                                   factor_name: str,
                                   top_percent: float = 0.2) -> Dict:
        """计算换手率分析"""
        try:
            if factor_name not in factor_data.columns:
                return {"error": f"因子{factor_name}不存在"}

            dates = sorted(factor_data['date'].unique())
            turnover_rates = []

            for i in range(1, len(dates)):
                prev_date = dates[i-1]
                curr_date = dates[i]

                prev_data = factor_data[factor_data['date'] == prev_date]
                curr_data = factor_data[factor_data['date'] == curr_date]

                if len(prev_data) == 0 or len(curr_data) == 0:
                    continue

                # 选择头部股票
                top_n = max(1, int(len(curr_data) * top_percent))

                prev_top = prev_data.nlargest(top_n, factor_name)['symbol'].tolist()
                curr_top = curr_data.nlargest(top_n, factor_name)['symbol'].tolist()

                # 计算换手率
                intersection = set(prev_top) & set(curr_top)
                turnover = 1 - len(intersection) / top_n

                turnover_rates.append({
                    'date': curr_date,
                    'turnover': turnover,
                    'new_positions': len(set(curr_top) - set(prev_top)),
                    'exited_positions': len(set(prev_top) - set(curr_top)),
                    'maintained_positions': len(intersection)
                })

            if not turnover_rates:
                return {"error": "没有有效的换手率计算结果"}

            turnover_df = pd.DataFrame(turnover_rates)
            turnover_series = turnover_df['turnover']

            results = {
                'factor_name': factor_name,
                'turnover_data': turnover_df,
                'avg_turnover': turnover_series.mean(),
                'std_turnover': turnover_series.std(),
                'max_turnover': turnover_series.max(),
                'min_turnover': turnover_series.min(),
                'periods': len(turnover_series),
                'date_range': {
                    'start': turnover_df['date'].min(),
                    'end': turnover_df['date'].max()
                }
            }

            self.logger.info(f"因子{factor_name}换手率分析完成")
            return results

        except Exception as e:
            self.logger.error(f"换手率分析失败: {str(e)}")
            return {"error": str(e)}

    def calculate_stability_analysis(self,
                                    factor_data: pd.DataFrame,
                                    factor_name: str,
                                    window: int = 60) -> Dict:
        """计算因子稳定性分析"""
        try:
            if factor_name not in factor_data.columns:
                return {"error": f"因子{factor_name}不存在"}

            # 按日期计算因子值统计
            daily_stats = []
            for date in factor_data['date'].unique():
                date_data = factor_data[factor_data['date'] == date]
                factor_values = date_data[factor_name].dropna()

                if len(factor_values) > 0:
                    daily_stats.append({
                        'date': date,
                        'mean': factor_values.mean(),
                        'std': factor_values.std(),
                        'min': factor_values.min(),
                        'max': factor_values.max(),
                        'count': len(factor_values)
                    })

            if not daily_stats:
                return {"error": "没有有效的因子值"}

            stats_df = pd.DataFrame(daily_stats)

            # 计算滚动统计
            rolling_stats = {}
            if len(stats_df) >= window:
                rolling_mean_std = stats_df['std'].rolling(window).mean()
                rolling_mean = stats_df['mean'].rolling(window).mean()
                rolling_cv = rolling_mean_std / rolling_mean.abs()

                rolling_stats = {
                    'rolling_std_mean': rolling_mean_std.iloc[-1],
                    'rolling_mean': rolling_mean.iloc[-1],
                    'rolling_cv': rolling_cv.iloc[-1]
                }

            # 因子值分布稳定性
            stability_metrics = {
                'mean_cv': stats_df['mean'].std() / abs(stats_df['mean'].mean()),
                'std_cv': stats_df['std'].std() / stats_df['std'].mean(),
                'range_cv': (stats_df['max'] - stats_df['min']).std() / (stats_df['max'] - stats_df['min']).mean()
            }

            results = {
                'factor_name': factor_name,
                'daily_stats': stats_df,
                'rolling_stats': rolling_stats,
                'stability_metrics': stability_metrics,
                'date_range': {
                    'start': stats_df['date'].min(),
                    'end': stats_df['date'].max()
                },
                'total_days': len(stats_df)
            }

            self.logger.info(f"因子{factor_name}稳定性分析完成")
            return results

        except Exception as e:
            self.logger.error(f"稳定性分析失败: {str(e)}")
            return {"error": str(e)}

    def generate_factor_report(self,
                             factor_data: pd.DataFrame,
                             factor_name: str,
                             config: Dict = None) -> Dict:
        """生成因子综合分析报告"""
        try:
            if config is None:
                config = self.default_config

            self.logger.info(f"开始生成因子{factor_name}的综合分析报告")

            # IC分析
            ic_analysis = self.calculate_ic_analysis(
                factor_data, factor_name,
                method='spearman',
                window=config.get('rolling_window', 60)
            )

            # 分层收益分析
            layer_analysis = self.calculate_layered_returns(
                factor_data, factor_name,
                layers=config.get('default_periods', 5)
            )

            # 换手率分析
            turnover_analysis = self.calculate_turnover_analysis(
                factor_data, factor_name,
                top_percent=0.2
            )

            # 稳定性分析
            stability_analysis = self.calculate_stability_analysis(
                factor_data, factor_name,
                window=config.get('rolling_window', 60)
            )

            # 综合评分
            score = self._calculate_factor_score(ic_analysis, layer_analysis)

            results = {
                'factor_name': factor_name,
                'ic_analysis': ic_analysis,
                'layer_analysis': layer_analysis,
                'turnover_analysis': turnover_analysis,
                'stability_analysis': stability_analysis,
                'overall_score': score,
                'report_date': pd.Timestamp.now().isoformat()
            }

            self.logger.info(f"因子{factor_name}综合分析报告生成完成")
            return results

        except Exception as e:
            self.logger.error(f"生成因子报告失败: {str(e)}")
            return {"error": str(e)}

    def _calculate_factor_score(self, ic_analysis: Dict, layer_analysis: Dict) -> Dict:
        """计算因子综合评分"""
        try:
            score = 0
            components = {}

            # IC评分 (40%)
            if 'ic_mean' in ic_analysis and not np.isnan(ic_analysis['ic_mean']):
                ic_score = min(abs(ic_analysis['ic_mean']) * 100, 40)
                score += ic_score
                components['ic_score'] = ic_score

            # IR评分 (30%)
            if 'ir' in ic_analysis and not np.isnan(ic_analysis['ir']):
                ir_score = min(abs(ic_analysis['ir']) * 20, 30)
                score += ir_score
                components['ir_score'] = ir_score

            # 分层收益评分 (20%)
            if 'summary' in layer_analysis and 'long_short' in layer_analysis['summary']:
                ls_sharpe = layer_analysis['summary']['long_short'].get('sharpe', 0)
                layer_score = min(ls_sharpe * 10, 20)
                score += layer_score
                components['layer_score'] = layer_score

            # 胜率评分 (10%)
            if 'ic_win_rate' in ic_analysis and not np.isnan(ic_analysis['ic_win_rate']):
                win_score = ic_analysis['ic_win_rate'] * 10
                score += win_score
                components['win_score'] = win_score

            return {
                'total_score': min(score, 100),
                'components': components,
                'rating': self._get_rating(score)
            }

        except Exception as e:
            self.logger.error(f"计算因子评分失败: {str(e)}")
            return {'total_score': 0, 'rating': 'Unknown'}

    def _get_rating(self, score: float) -> str:
        """根据评分获取评级"""
        if score >= 80:
            return 'A+'
        elif score >= 70:
            return 'A'
        elif score >= 60:
            return 'B+'
        elif score >= 50:
            return 'B'
        elif score >= 40:
            return 'C+'
        elif score >= 30:
            return 'C'
        else:
            return 'D'

    def compare_factors(self,
                       factor_data: pd.DataFrame,
                       factor_names: List[str]) -> Dict:
        """比较多个因子"""
        try:
            comparison_results = {}

            for factor_name in factor_names:
                if factor_name in factor_data.columns:
                    report = self.generate_factor_report(factor_data, factor_name)
                    comparison_results[factor_name] = report

            # 排序
            if comparison_results:
                sorted_factors = sorted(
                    comparison_results.items(),
                    key=lambda x: x[1]['overall_score']['total_score'],
                    reverse=True
                )

                ranking = {
                    'rankings': sorted_factors,
                    'best_factor': sorted_factors[0][0] if sorted_factors else None,
                    'worst_factor': sorted_factors[-1][0] if sorted_factors else None
                }

                return {
                    'comparison_results': comparison_results,
                    'ranking': ranking,
                    'comparison_date': pd.Timestamp.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"因子比较失败: {str(e)}")
            return {"error": str(e)}