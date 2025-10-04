"""
回测引擎
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.logger import logger
from utils.config import DEFAULT_CONFIG
from utils.helpers import save_to_cache, load_from_cache
from .strategy import BaseStrategy
from analyzer.portfolio_analyzer import PortfolioAnalyzer

class Backtester:
    """回测引擎"""

    def __init__(self,
                 initial_capital: float = 1000000,
                 commission: float = 0.0003,
                 slippage: float = 0.001,
                 benchmark: str = "000300"):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.benchmark = benchmark
        self.logger = logger
        self.portfolio_analyzer = PortfolioAnalyzer()

        # 回测状态
        self.reset()

    def reset(self):
        """重置回测状态"""
        self.current_capital = self.initial_capital
        self.positions = {}  # 持仓 {symbol: quantity}
        self.portfolio_value = []
        self.cash_history = []
        self.positions_history = []
        self.trades_history = []
        self.benchmark_data = []

    def load_price_data(self, data: pd.DataFrame, symbol: str, date: pd.Timestamp) -> Dict:
        """获取股票价格数据"""
        symbol_data = data[data['symbol'] == symbol]
        price_data = symbol_data[symbol_data['date'] <= date].sort_values('date')

        if len(price_data) == 0:
            return {}

        latest = price_data.iloc[-1]
        return {
            'open': latest['open'],
            'close': latest['close'],
            'high': latest['high'],
            'low': latest['low'],
            'volume': latest['volume']
        }

    def execute_trade(self,
                     symbol: str,
                     target_weight: float,
                     current_price: float,
                     date: pd.Timestamp) -> Dict:
        """执行交易"""
        try:
            # 计算目标价值和当前价值
            portfolio_value = self.calculate_portfolio_value(date)
            target_value = target_weight * portfolio_value

            current_quantity = self.positions.get(symbol, 0)
            current_value = current_quantity * current_price

            # 计算需要交易的数量
            value_diff = target_value - current_value
            if abs(value_diff) < self.initial_capital * 0.001:  # 忽略小额差异
                return {"symbol": symbol, "action": "hold", "quantity": 0, "cost": 0}

            # 考虑滑点和手续费
            if value_diff > 0:  # 买入
                adjusted_price = current_price * (1 + self.slippage)
                transaction_cost = self.commission
            else:  # 卖出
                adjusted_price = current_price * (1 - self.slippage)
                transaction_cost = self.commission

            # 计算交易数量
            if value_diff > 0:
                trade_quantity = value_diff / (adjusted_price * (1 + transaction_cost))
                action = "buy"
            else:
                trade_quantity = abs(value_diff) / (adjusted_price * (1 - transaction_cost))
                action = "sell"

            # 四舍五入到100股的整数倍（A股规则）
            trade_quantity = int(trade_quantity / 100) * 100

            if trade_quantity == 0:
                return {"symbol": symbol, "action": "hold", "quantity": 0, "cost": 0}

            # 计算交易成本
            trade_cost = trade_quantity * adjusted_price * transaction_cost

            # 更新现金和持仓
            if action == "buy":
                total_cost = trade_quantity * adjusted_price + trade_cost
                if total_cost <= self.current_capital:
                    self.current_capital -= total_cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + trade_quantity
                else:
                    return {"symbol": symbol, "action": "hold", "quantity": 0, "cost": 0}
            else:  # sell
                max_sellable = min(trade_quantity, self.positions.get(symbol, 0))
                if max_sellable > 0:
                    total_proceeds = max_sellable * adjusted_price - trade_cost
                    self.current_capital += total_proceeds
                    self.positions[symbol] -= max_sellable
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                    trade_quantity = max_sellable
                else:
                    return {"symbol": symbol, "action": "hold", "quantity": 0, "cost": 0}

            # 记录交易
            trade_record = {
                "date": date,
                "symbol": symbol,
                "action": action,
                "quantity": trade_quantity,
                "price": adjusted_price,
                "cost": trade_cost,
                "target_weight": target_weight
            }
            self.trades_history.append(trade_record)

            return {
                "symbol": symbol,
                "action": action,
                "quantity": trade_quantity,
                "cost": trade_cost,
                "price": adjusted_price
            }

        except Exception as e:
            self.logger.error(f"执行交易失败: {str(e)}")
            return {"symbol": symbol, "action": "error", "quantity": 0, "cost": 0}

    def calculate_portfolio_value(self, date: pd.Timestamp, data: pd.DataFrame = None) -> float:
        """计算组合总价值"""
        try:
            if data is not None:
                # 使用提供的数据计算持仓价值
                positions_value = 0
                for symbol, quantity in self.positions.items():
                    price_data = self.load_price_data(data, symbol, date)
                    if price_data:
                        positions_value += quantity * price_data['close']
            else:
                # 使用历史数据
                positions_value = sum(
                    position['value'] for position in self.positions_history
                    if position['date'] == date
                )

            return self.current_capital + positions_value

        except Exception as e:
            self.logger.error(f"计算组合价值失败: {str(e)}")
            return self.current_capital

    def run_backtest(self,
                    strategy: BaseStrategy,
                    data: pd.DataFrame,
                    start_date: str,
                    end_date: str,
                    rebalance_frequency: str = "monthly") -> Dict:
        """运行回测"""
        try:
            self.logger.info(f"开始回测策略: {strategy.name}")
            self.reset()

            # 预处理数据
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
            data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

            # 生成回测日期
            dates = sorted(data['date'].unique())
            if len(dates) == 0:
                return {"error": "没有有效的回测数据"}

            # 确定调仓日期
            rebalance_dates = self._get_rebalance_dates(dates, rebalance_frequency)

            self.logger.info(f"回测期间: {start_date} 至 {end_date}, 共{len(dates)}个交易日")

            # 初始化基准数据
            benchmark_values = []
            current_benchmark_value = 1.0

            # 回测主循环
            for i, date in enumerate(dates):
                date_data = data[data['date'] == date]

                # 计算当前组合价值
                portfolio_val = self.calculate_portfolio_value(date, data)

                # 记录组合历史
                self.portfolio_value.append({
                    "date": date,
                    "value": portfolio_val,
                    "capital": self.current_capital
                })

                # 记录现金历史
                self.cash_history.append({
                    "date": date,
                    "cash": self.current_capital,
                    "cash_ratio": self.current_capital / portfolio_val if portfolio_val > 0 else 0
                })

                # 记录持仓历史
                positions_detail = []
                total_positions_value = 0
                for symbol, quantity in self.positions.items():
                    price_data = self.load_price_data(data, symbol, date)
                    if price_data:
                        position_value = quantity * price_data['close']
                        positions_detail.append({
                            "symbol": symbol,
                            "quantity": quantity,
                            "price": price_data['close'],
                            "value": position_value,
                            "weight": position_value / portfolio_val if portfolio_val > 0 else 0
                        })
                        total_positions_value += position_value

                self.positions_history.append({
                    "date": date,
                    "positions": positions_detail,
                    "total_positions_value": total_positions_value,
                    "positions_count": len(self.positions)
                })

                # 基准数据更新（简化处理，假设基准指数日收益率）
                if i > 0:
                    daily_return = np.random.normal(0.0005, 0.02)  # 简化的基准收益率
                    current_benchmark_value *= (1 + daily_return)
                benchmark_values.append({
                    "date": date,
                    "value": current_benchmark_value
                })

                # 调仓
                if date in rebalance_dates:
                    self.logger.debug(f"调仓日期: {date}")
                    signals = strategy.generate_signals(data, date)

                    if signals:
                        # 执行交易
                        for symbol, target_weight in signals.items():
                            price_data = self.load_price_data(data, symbol, date)
                            if price_data:
                                self.execute_trade(symbol, target_weight, price_data['close'], date)

            # 计算回测结果
            results = self._calculate_backtest_results(strategy, data)
            results['benchmark_data'] = pd.DataFrame(benchmark_values)

            self.logger.info(f"回测完成，总收益率: {results['total_return']:.2%}")
            return results

        except Exception as e:
            self.logger.error(f"回测失败: {str(e)}")
            return {"error": str(e)}

    def _get_rebalance_dates(self, dates: List[pd.Timestamp], frequency: str) -> List[pd.Timestamp]:
        """获取调仓日期"""
        if frequency == "daily":
            return dates
        elif frequency == "weekly":
            # 选择每周第一个交易日
            rebalance_dates = []
            current_week = None
            for date in dates:
                week = date.isocalendar()[1:2]
                if week != current_week:
                    rebalance_dates.append(date)
                    current_week = week
            return rebalance_dates
        elif frequency == "monthly":
            # 选择每月第一个交易日
            rebalance_dates = []
            current_month = None
            for date in dates:
                month = (date.year, date.month)
                if month != current_month:
                    rebalance_dates.append(date)
                    current_month = month
            return rebalance_dates
        else:
            # 默认按月调仓
            return self._get_rebalance_dates(dates, "monthly")

    def _calculate_backtest_results(self, strategy: BaseStrategy, data: pd.DataFrame) -> Dict:
        """计算回测结果"""
        try:
            if not self.portfolio_value:
                return {"error": "没有组合价值数据"}

            portfolio_df = pd.DataFrame(self.portfolio_value)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df = portfolio_df.sort_values('date')

            # 计算收益率
            portfolio_df['return'] = portfolio_df['value'].pct_change()
            returns = portfolio_df['return'].dropna()

            # 基本绩效指标
            total_return = (portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0

            # 最大回撤
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # 胜率
            win_rate = (returns > 0).sum() / len(returns)

            # 交易统计
            total_trades = len(self.trades_history)
            buy_trades = len([t for t in self.trades_history if t['action'] == 'buy'])
            sell_trades = len([t for t in self.trades_history if t['action'] == 'sell'])
            total_commission = sum(t['cost'] for t in self.trades_history)

            # 持仓统计
            avg_positions = len(self.positions) if self.positions else 0
            max_positions = max([p['positions_count'] for p in self.positions_history]) if self.positions_history else 0

            # 换手率
            turnover_rates = self._calculate_turnover_rates()
            avg_turnover = np.mean(turnover_rates) if turnover_rates else 0

            results = {
                'strategy_name': strategy.name,
                'backtest_period': {
                    'start': portfolio_df['date'].min().strftime('%Y-%m-%d'),
                    'end': portfolio_df['date'].max().strftime('%Y-%m-%d'),
                    'trading_days': len(portfolio_df)
                },
                'performance_metrics': {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate
                },
                'trading_statistics': {
                    'total_trades': total_trades,
                    'buy_trades': buy_trades,
                    'sell_trades': sell_trades,
                    'total_commission': total_commission,
                    'commission_rate': total_commission / self.initial_capital
                },
                'portfolio_statistics': {
                    'initial_capital': self.initial_capital,
                    'final_value': portfolio_df['value'].iloc[-1],
                    'avg_positions': avg_positions,
                    'max_positions': max_positions,
                    'avg_turnover': avg_turnover
                },
                'detailed_data': {
                    'portfolio_values': portfolio_df,
                    'positions_history': pd.DataFrame(self.positions_history),
                    'trades_history': pd.DataFrame(self.trades_history),
                    'cash_history': pd.DataFrame(self.cash_history)
                }
            }

            return results

        except Exception as e:
            self.logger.error(f"计算回测结果失败: {str(e)}")
            return {"error": str(e)}

    def _calculate_turnover_rates(self) -> List[float]:
        """计算换手率"""
        try:
            if len(self.positions_history) < 2:
                return []

            turnover_rates = []
            for i in range(1, len(self.positions_history)):
                prev_positions = set(p['symbol'] for p in self.positions_history[i-1]['positions'])
                curr_positions = set(p['symbol'] for p in self.positions_history[i]['positions'])

                if prev_positions or curr_positions:
                    # 计算持仓变化
                    changes = len(prev_positions.symmetric_difference(curr_positions))
                    total = len(prev_positions.union(curr_positions))
                    turnover = changes / total if total > 0 else 0
                    turnover_rates.append(turnover)

            return turnover_rates

        except Exception as e:
            self.logger.error(f"计算换手率失败: {str(e)}")
            return []

    def compare_strategies(self,
                          strategies: List[BaseStrategy],
                          data: pd.DataFrame,
                          start_date: str,
                          end_date: str) -> Dict:
        """比较多个策略"""
        try:
            comparison_results = {}

            for strategy in strategies:
                self.logger.info(f"回测策略: {strategy.name}")
                result = self.run_backtest(strategy, data, start_date, end_date)
                comparison_results[strategy.name] = result

            # 排序和统计
            if comparison_results:
                # 按总收益率排序
                sorted_strategies = sorted(
                    comparison_results.items(),
                    key=lambda x: x[1].get('performance_metrics', {}).get('total_return', 0),
                    reverse=True
                )

                best_strategy = sorted_strategies[0] if sorted_strategies else None

                # 计算统计指标
                metrics_comparison = {}
                metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']

                for metric in metrics:
                    values = []
                    for name, result in comparison_results.items():
                        if 'performance_metrics' in result and metric in result['performance_metrics']:
                            values.append((name, result['performance_metrics'][metric]))

                    if values:
                        metrics_comparison[metric] = {
                            'values': values,
                            'best': max(values, key=lambda x: x[1]) if metric != 'max_drawdown' else min(values, key=lambda x: x[1]),
                            'average': np.mean([v[1] for v in values]),
                            'std': np.std([v[1] for v in values])
                        }

                results = {
                    'individual_results': comparison_results,
                    'ranking': sorted_strategies,
                    'best_strategy': best_strategy[0] if best_strategy else None,
                    'metrics_comparison': metrics_comparison,
                    'comparison_date': pd.Timestamp.now().isoformat()
                }

                return results

        except Exception as e:
            self.logger.error(f"策略比较失败: {str(e)}")
            return {"error": str(e)}

    def export_results(self, results: Dict, file_path: str, format: str = "excel") -> bool:
        """导出回测结果"""
        try:
            if format == "excel":
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # 摘要页
                    summary_data = []
                    for category, metrics in results.items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    summary_data.append({
                                        'Category': category,
                                        'Metric': metric_name,
                                        'Value': value
                                    })
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                    # 详细数据
                    if 'detailed_data' in results:
                        detailed = results['detailed_data']
                        if 'portfolio_values' in detailed:
                            detailed['portfolio_values'].to_excel(writer, sheet_name='Portfolio_Values', index=False)
                        if 'trades_history' in detailed:
                            detailed['trades_history'].to_excel(writer, sheet_name='Trades', index=False)
                        if 'positions_history' in detailed:
                            # 处理嵌套的positions数据
                            positions_expanded = []
                            for pos_record in detailed['positions_history'].to_dict('records'):
                                for position in pos_record['positions']:
                                    position['date'] = pos_record['date']
                                    positions_expanded.append(position)
                            pd.DataFrame(positions_expanded).to_excel(writer, sheet_name='Positions', index=False)

            self.logger.info(f"回测结果已导出到: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"导出回测结果失败: {str(e)}")
            return False