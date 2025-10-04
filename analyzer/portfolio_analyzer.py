"""
组合分析器
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from utils.logger import logger
from utils.helpers import calculate_returns, format_number
from utils.config import DEFAULT_CONFIG

class PortfolioAnalyzer:
    """组合分析器"""

    def __init__(self):
        self.logger = logger
        self.default_config = DEFAULT_CONFIG['backtest']

    def calculate_portfolio_metrics(self,
                                  portfolio_data: pd.DataFrame,
                                  benchmark_data: pd.DataFrame = None,
                                  risk_free_rate: float = 0.03) -> Dict:
        """计算组合绩效指标"""
        try:
            if 'value' not in portfolio_data.columns:
                return {"error": "组合净值数据不存在"}

            portfolio_values = portfolio_data['value']
            portfolio_returns = portfolio_values.pct_change().dropna()

            # 基本收益指标
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
            daily_return_mean = portfolio_returns.mean()
            daily_return_std = portfolio_returns.std()

            # 风险指标
            sharpe_ratio = (annual_return - risk_free_rate) / (daily_return_std * np.sqrt(252))
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            volatility = daily_return_std * np.sqrt(252)

            # 胜率指标
            positive_days = (portfolio_returns > 0).sum()
            total_days = len(portfolio_returns)
            win_rate = positive_days / total_days

            # 其他指标
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurtosis()
            var_95 = portfolio_returns.quantile(0.05)
            var_99 = portfolio_returns.quantile(0.01)

            results = {
                'return_metrics': {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'daily_return_mean': daily_return_mean,
                    'cumulative_return': total_return
                },
                'risk_metrics': {
                    'volatility': volatility,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'var_95': var_95,
                    'var_99': var_99
                },
                'other_metrics': {
                    'win_rate': win_rate,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'total_days': total_days,
                    'positive_days': positive_days,
                    'negative_days': total_days - positive_days
                }
            }

            # 如果有基准数据，计算相对指标
            if benchmark_data is not None and 'value' in benchmark_data.columns:
                benchmark_values = benchmark_data['value']
                benchmark_returns = benchmark_values.pct_change().dropna()

                # 对齐日期
                aligned_returns = pd.DataFrame({
                    'portfolio': portfolio_returns,
                    'benchmark': benchmark_returns
                }).dropna()

                if len(aligned_returns) > 0:
                    # 超额收益
                    excess_returns = aligned_returns['portfolio'] - aligned_returns['benchmark']
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    information_ratio = excess_returns.mean() * np.sqrt(252) / excess_returns.std()
                    beta = np.cov(aligned_returns['portfolio'], aligned_returns['benchmark'])[0, 1] / np.var(aligned_returns['benchmark'])
                    alpha = (aligned_returns['portfolio'].mean() * 252) - risk_free_rate - beta * (aligned_returns['benchmark'].mean() * 252 - risk_free_rate)

                    results['relative_metrics'] = {
                        'alpha': alpha,
                        'beta': beta,
                        'tracking_error': tracking_error,
                        'information_ratio': information_ratio,
                        'correlation': aligned_returns['portfolio'].corr(aligned_returns['benchmark'])
                    }

            self.logger.info("组合绩效指标计算完成")
            return results

        except Exception as e:
            self.logger.error(f"组合绩效计算失败: {str(e)}")
            return {"error": str(e)}

    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + values.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def calculate_sector_exposure(self,
                                 portfolio_data: pd.DataFrame,
                                 sector_mapping: Dict[str, str]) -> Dict:
        """计算行业暴露度"""
        try:
            if 'symbol' not in portfolio_data.columns or 'weight' not in portfolio_data.columns:
                return {"error": "缺少必要的列数据"}

            # 添加行业信息
            portfolio_data['sector'] = portfolio_data['symbol'].map(sector_mapping)

            # 计算行业权重
            sector_weights = portfolio_data.groupby('sector')['weight'].sum()
            sector_weights = sector_weights.sort_values(ascending=False)

            # 计算行业暴露度统计
            total_weight = sector_weights.sum()
            sector_exposure = (sector_weights / total_weight).to_dict()

            # 集中度指标
            herfindahl_index = sum((weight / 100) ** 2 for weight in sector_weights.values)

            results = {
                'sector_exposure': sector_exposure,
                'sector_weights': sector_weights.to_dict(),
                'herfindahl_index': herfindahl_index,
                'top_sector': sector_weights.index[0] if len(sector_weights) > 0 else None,
                'sector_count': len(sector_weights),
                'max_exposure': sector_weights.max() if len(sector_weights) > 0 else 0
            }

            self.logger.info("行业暴露度分析完成")
            return results

        except Exception as e:
            self.logger.error(f"行业暴露度分析失败: {str(e)}")
            return {"error": str(e)}

    def calculate_factor_exposure(self,
                                portfolio_data: pd.DataFrame,
                                factor_data: pd.DataFrame,
                                factor_names: List[str]) -> Dict:
        """计算因子暴露度"""
        try:
            if 'symbol' not in portfolio_data.columns or 'weight' not in portfolio_data.columns:
                return {"error": "缺少必要的列数据"}

            # 合并数据
            merged_data = portfolio_data.merge(factor_data, on='symbol', how='inner')

            if merged_data.empty:
                return {"error": "无法合并组合和因子数据"}

            factor_exposures = {}

            for factor_name in factor_names:
                if factor_name in merged_data.columns:
                    # 加权平均因子暴露度
                    weighted_exposure = (merged_data[factor_name] * merged_data['weight']).sum() / merged_data['weight'].sum()
                    factor_exposures[factor_name] = weighted_exposure

            # 因子暴露度统计
            if factor_exposures:
                exposure_values = list(factor_exposures.values())
                results = {
                    'factor_exposures': factor_exposures,
                    'statistics': {
                        'max_exposure': max(exposure_values),
                        'min_exposure': min(exposure_values),
                        'mean_exposure': np.mean(exposure_values),
                        'std_exposure': np.std(exposure_values)
                    },
                    'highest_exposure_factor': max(factor_exposures, key=factor_exposures.get),
                    'lowest_exposure_factor': min(factor_exposures, key=factor_exposures.get)
                }
            else:
                results = {"error": "没有有效的因子暴露度数据"}

            self.logger.info("因子暴露度分析完成")
            return results

        except Exception as e:
            self.logger.error(f"因子暴露度分析失败: {str(e)}")
            return {"error": str(e)}

    def calculate_attribution_analysis(self,
                                     portfolio_returns: pd.Series,
                                     sector_returns: pd.DataFrame,
                                     sector_weights: pd.Series) -> Dict:
        """计算归因分析"""
        try:
            # 行业归因分析
            attribution_results = {}

            for sector in sector_weights.index:
                if sector in sector_returns.columns:
                    sector_weight = sector_weights[sector]
                    sector_return = sector_returns[sector]

                    # 贡献度 = 权重 * 收益率
                    contribution = sector_weight * sector_return.mean()
                    attribution_results[sector] = {
                        'weight': sector_weight,
                        'return': sector_return.mean(),
                        'contribution': contribution,
                        'contribution_pct': contribution / portfolio_returns.mean() if portfolio_returns.mean() != 0 else 0
                    }

            # 排序
            sorted_attribution = sorted(
                attribution_results.items(),
                key=lambda x: abs(x[1]['contribution']),
                reverse=True
            )

            results = {
                'sector_attribution': attribution_results,
                'ranking': sorted_attribution,
                'total_attributed_return': sum(item[1]['contribution'] for item in attribution_results.items()),
                'attribution_gap': portfolio_returns.mean() - sum(item[1]['contribution'] for item in attribution_results.items())
            }

            self.logger.info("归因分析完成")
            return results

        except Exception as e:
            self.logger.error(f"归因分析失败: {str(e)}")
            return {"error": str(e)}

    def calculate_concentration_risk(self,
                                   portfolio_data: pd.DataFrame,
                                   top_n: int = 10) -> Dict:
        """计算集中度风险"""
        try:
            if 'symbol' not in portfolio_data.columns or 'weight' not in portfolio_data.columns:
                return {"error": "缺少必要的列数据"}

            # 按权重排序
            sorted_weights = portfolio_data.sort_values('weight', ascending=False)

            # 前N大持仓
            top_holdings = sorted_weights.head(top_n)
            top_weight_sum = top_holdings['weight'].sum()

            # 集中度指标
            total_weight = portfolio_data['weight'].sum()
            weight_distribution = portfolio_data['weight'] / total_weight

            # 基尼系数
            gini_coefficient = self._calculate_gini_coefficient(weight_distribution.values)

            # HHI指数
            hhi_index = sum((weight / 100) ** 2 for weight in portfolio_data['weight'].values)

            results = {
                'top_n_holdings': top_holdings[['symbol', 'weight']].to_dict('records'),
                'top_n_weight': top_weight_sum,
                'top_n_weight_pct': top_weight_sum / total_weight * 100,
                'gini_coefficient': gini_coefficient,
                'hhi_index': hhi_index,
                'max_single_weight': portfolio_data['weight'].max(),
                'min_single_weight': portfolio_data['weight'].min(),
                'total_holdings': len(portfolio_data)
            }

            self.logger.info("集中度风险分析完成")
            return results

        except Exception as e:
            self.logger.error(f"集中度风险分析失败: {str(e)}")
            return {"error": str(e)}

    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """计算基尼系数"""
        if len(values) == 0:
            return 0

        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
        return gini

    def generate_risk_report(self,
                           portfolio_data: pd.DataFrame,
                           portfolio_returns: pd.Series,
                           benchmark_returns: pd.Series = None) -> Dict:
        """生成风险报告"""
        try:
            self.logger.info("开始生成组合风险报告")

            # 基本绩效指标
            portfolio_metrics = self.calculate_portfolio_metrics(
                portfolio_data, benchmark_returns
            )

            # 集中度风险
            concentration_risk = self.calculate_concentration_risk(portfolio_data)

            # VaR计算
            var_analysis = self._calculate_var_analysis(portfolio_returns)

            # 压力测试
            stress_test = self._calculate_stress_test(portfolio_returns)

            # 波动率分析
            volatility_analysis = self._calculate_volatility_analysis(portfolio_returns)

            results = {
                'portfolio_metrics': portfolio_metrics,
                'concentration_risk': concentration_risk,
                'var_analysis': var_analysis,
                'stress_test': stress_test,
                'volatility_analysis': volatility_analysis,
                'report_date': pd.Timestamp.now().isoformat()
            }

            self.logger.info("组合风险报告生成完成")
            return results

        except Exception as e:
            self.logger.error(f"生成风险报告失败: {str(e)}")
            return {"error": str(e)}

    def _calculate_var_analysis(self, returns: pd.Series) -> Dict:
        """计算VaR分析"""
        try:
            returns_clean = returns.dropna()

            var_95 = returns_clean.quantile(0.05)
            var_99 = returns_clean.quantile(0.01)
            cvar_95 = returns_clean[returns_clean <= var_95].mean()
            cvar_99 = returns_clean[returns_clean <= var_99].mean()

            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'var_95_bps': var_95 * 10000,
                'var_99_bps': var_99 * 10000
            }

        except Exception as e:
            self.logger.error(f"VaR分析失败: {str(e)}")
            return {"error": str(e)}

    def _calculate_stress_test(self, returns: pd.Series) -> Dict:
        """计算压力测试"""
        try:
            returns_clean = returns.dropna()

            # 历史最差情况
            worst_day = returns_clean.min()
            worst_week = returns_clean.rolling(5).sum().min()
            worst_month = returns_clean.rolling(20).sum().min()

            # 连续下跌天数
            negative_returns = (returns_clean < 0)
            consecutive_losses = self._calculate_consecutive_losses(negative_returns)

            return {
                'worst_day': worst_day,
                'worst_week': worst_week,
                'worst_month': worst_month,
                'max_consecutive_losses': consecutive_losses,
                'loss_days': negative_returns.sum(),
                'loss_days_ratio': negative_returns.mean()
            }

        except Exception as e:
            self.logger.error(f"压力测试失败: {str(e)}")
            return {"error": str(e)}

    def _calculate_consecutive_losses(self, losses: pd.Series) -> int:
        """计算最大连续损失天数"""
        max_consecutive = 0
        current_consecutive = 0

        for loss in losses:
            if loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_volatility_analysis(self, returns: pd.Series) -> Dict:
        """计算波动率分析"""
        try:
            returns_clean = returns.dropna()

            # 不同时间窗口的波动率
            vol_10d = returns_clean.rolling(10).std() * np.sqrt(252)
            vol_20d = returns_clean.rolling(20).std() * np.sqrt(252)
            vol_60d = returns_clean.rolling(60).std() * np.sqrt(252)

            return {
                'volatility_10d_mean': vol_10d.mean(),
                'volatility_20d_mean': vol_20d.mean(),
                'volatility_60d_mean': vol_60d.mean(),
                'volatility_10d_std': vol_10d.std(),
                'volatility_20d_std': vol_20d.std(),
                'volatility_60d_std': vol_60d.std(),
                'volatility_trend': 'increasing' if vol_20d.iloc[-5:].mean() > vol_20d.iloc[-10:-5].mean() else 'decreasing'
            }

        except Exception as e:
            self.logger.error(f"波动率分析失败: {str(e)}")
            return {"error": str(e)}