"""
基于 vectorbt 的回测服务
更准确、更高效的回测引擎
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from backend.services.portfolio_analysis_service import portfolio_analysis_service

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False


class VectorBTBacktestService:
    """基于 vectorbt 的回测服务"""

    def __init__(self, initial_capital: float = 1000000, commission_rate: float = 0.0003, slippage: float = 0.0):
        """
        初始化回测服务

        Args:
            initial_capital: 初始资金，默认100万
            commission_rate: 手续费率，默认万三
            slippage: 滑点率，默认0（不考虑滑点）
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("vectorbt未安装，请运行: pip install vectorbt")

        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage

    def prepare_multi_factor_data(
        self,
        df: pd.DataFrame,
        factor_names: List[str],
        weights: Optional[List[float]] = None,
        method: str = "equal_weight",
    ) -> pd.DataFrame:
        """根据多个因子构建组合得分列。"""
        prepared_df = df.copy()

        if not isinstance(prepared_df.index, pd.DatetimeIndex):
            if "date" in prepared_df.columns:
                prepared_df = prepared_df.set_index("date")
            prepared_df.index = pd.to_datetime(prepared_df.index)

        normalized_factors = []
        available_factor_names = []
        for factor_name in factor_names:
            if factor_name not in prepared_df.columns:
                continue

            normalized_name = f"{factor_name}_normalized"
            factor_mean = prepared_df[factor_name].mean()
            factor_std = prepared_df[factor_name].std()
            if factor_std and not np.isnan(factor_std):
                prepared_df[normalized_name] = (
                    prepared_df[factor_name] - factor_mean
                ) / factor_std
            else:
                prepared_df[normalized_name] = 0.0
            normalized_factors.append(normalized_name)
            available_factor_names.append(factor_name)

        if not normalized_factors:
            raise ValueError("没有可用于构建组合得分的有效因子列")

        if weights is None:
            if method == "equal_weight":
                weights = [1.0 / len(normalized_factors)] * len(normalized_factors)
            else:
                prepared_df["returns"] = prepared_df["close"].pct_change().shift(-1)
                factor_returns = {}
                for factor_name, normalized_name in zip(available_factor_names, normalized_factors):
                    aligned = pd.DataFrame(
                        {
                            "factor": prepared_df[normalized_name],
                            "returns": prepared_df["returns"],
                        }
                    ).replace([np.inf, -np.inf], np.nan).dropna()

                    if aligned.empty:
                        continue

                    factor_returns[factor_name] = aligned["factor"] * aligned["returns"]

                factor_returns_df = pd.DataFrame(factor_returns).dropna(how="all")
                if factor_returns_df.empty:
                    raise ValueError("无法基于真实因子收益序列构建组合权重")

                optimized = portfolio_analysis_service.optimize_weights(
                    factor_returns=factor_returns_df,
                    method=method,
                )
                if "error" in optimized:
                    raise ValueError(optimized["error"])

                weights = [
                    optimized["weights"].get(factor_name, 0.0)
                    for factor_name in available_factor_names
                ]
        elif len(weights) != len(normalized_factors):
            raise ValueError("自定义权重数量与有效因子数量不匹配")

        prepared_df["composite_score"] = sum(
            prepared_df[nf] * w for nf, w in zip(normalized_factors, weights)
        )

        return prepared_df

    def single_factor_backtest(
        self,
        df: pd.DataFrame,
        factor_name: str,
        percentile: int = 50,
        direction: str = "long",
        n_quantiles: int = 5,
        shares_per_trade: int = 100,  # 添加每次交易手数参数，默认1手（100股）
    ) -> Dict:
        """
        单因子分层回测（使用vectorbt）

        Args:
            df: 包含价格和因子数据的DataFrame，必须有 close 列和因子列
            factor_name: 因子名称
            percentile: 分位数阈值（0-100），用于做多/做空判断
            direction: 交易方向，"long"做多或"short"做空
            n_quantiles: 分层数量，默认5层

        Returns:
            Dict: 包含各层收益、整体收益、净值曲线等数据的字典
        """
        # 确保索引是 DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)

        # 1. 计算收益率
        df["returns"] = df["close"].pct_change()
        df["forward_returns"] = df["close"].pct_change().shift(-1)

        # 2. 计算因子分位数（滚动窗口252天）
        factor_rank = df[factor_name].rolling(252, min_periods=1).rank(pct=True)

        # 3. 生成分层信号
        percentile_threshold = percentile / 100.0
        if direction == "long":
            raw_entries = factor_rank >= percentile_threshold
            raw_exits = factor_rank < percentile_threshold
        else:
            raw_entries = factor_rank <= percentile_threshold
            raw_exits = factor_rank > percentile_threshold

        # 因子在 t 日收盘后才可观测，交易信号需顺延到下一交易日执行
        entries = raw_entries.shift(1, fill_value=False)
        exits = raw_exits.shift(1, fill_value=False)

        # 4. 计算各层收益
        quantile_returns = {}
        for q in range(n_quantiles):
            q_min = q / n_quantiles
            q_max = (q + 1) / n_quantiles
            layer_mask = (factor_rank >= q_min) & (factor_rank < q_max)
            layer_returns = df.loc[layer_mask, "forward_returns"]
            quantile_returns[f"Q{q + 1}"] = layer_returns

        # 5. 创建回测结果（使用vectorbt的信号回测）
        # 创建Portfolio对象，设置手续费和滑点
        # 使用固定手数交易（每次交易shares_per_trade股）
        pf = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries,
            exits=exits,  # 添加退出信号
            init_cash=self.initial_capital,
            freq="D",  # 日频
            cash_sharing=False,  # 不共享现金
            fees=self.commission_rate,  # 手续费率
            slippage=self.slippage,  # 滑点率
            size=shares_per_trade,  # 每次交易的数量（股）
        )

        # 获取净值、收益和性能指标
        equity = pf.value()
        returns = pf.returns()
        returns_clean = returns.dropna()

        # 使用 VectorBT 的 stats() 方法获取所有指标
        stats = pf.stats()

        # 从 stats Series 中提取指标
        # VectorBT 返回的百分比值需要除以100转换为小数
        total_return = stats.get('Total Return [%]', 0) / 100.0

        # 年化收益率：如果VectorBT返回0或NaN，则手动计算
        annual_return = stats.get('Annual Return [%]', 0) / 100.0
        if annual_return == 0 or np.isnan(annual_return):
            # 手动计算年化收益率 = (1 + 总收益率)^(252/交易天数) - 1
            n_days = len(returns_clean)
            if n_days > 0:
                annual_return = (1 + total_return) ** (252 / n_days) - 1
            else:
                annual_return = 0.0

        volatility = self._calculate_volatility(returns_clean, stats)

        sharpe_ratio = stats.get('Sharpe Ratio', 0)
        sortino_ratio = stats.get('Sortino Ratio', 0)
        max_drawdown = stats.get('Max Drawdown [%]', 0) / 100.0
        calmar_ratio = stats.get('Calmar Ratio', 0)
        win_rate = stats.get('Win Rate [%]', 0) / 100.0

        # VaR 和 CVaR 需要自己计算
        var_95, cvar_95 = self._calculate_var_cvar(returns_clean)

        # 计算交易次数
        trades_count = stats.get('Total Trades', 0)

        # 提取交易记录（使用VectorBT的records_readable方法）
        trades_df = None
        try:
            # 使用records_readable获取可读的交易记录
            # records_readable返回的是pandas DataFrame
            if hasattr(pf, 'trades') and hasattr(pf.trades, 'records_readable'):
                trades_readable = pf.trades.records_readable

                if trades_readable is not None and len(trades_readable) > 0:
                    # records_readable是DataFrame，直接重命名列
                    trades_df = trades_readable.copy()

                    # 创建列名映射（英文 -> 中文）
                    column_mapping = {
                        'Trade Id': '交易ID',
                        'Column': '股票代码',
                        'Size': '数量',
                        'Entry Timestamp': '入场时间',
                        'Avg Entry Price': '入场价格',
                        'Entry Fees': '入场手续费',
                        'Exit Timestamp': '出场时间',
                        'Avg Exit Price': '出场价格',
                        'Exit Fees': '出场手续费',
                        'PnL': '收益',
                        'Return': '收益率',
                        'Direction': '方向',
                        'Status': '状态',
                        'Parent Id': '父ID'
                    }

                    # 重命名列
                    trades_df.rename(columns=column_mapping, inplace=True)

                    # 转换方向和状态为中文
                    # VectorBT records_readable: 'Long'/'Short' (字符串)
                    if '方向' in trades_df.columns:
                        trades_df['方向'] = trades_df['方向'].map({
                            'Long': '做多',
                            'Short': '做空'
                        }).fillna('未知')

                    if '状态' in trades_df.columns:
                        trades_df['状态'] = trades_df['状态'].map({
                            'Open': '持仓中',
                            'Closed': '已平仓'
                        }).fillna('未知')

                    # 将入场时间设为索引（如果存在）
                    if '入场时间' in trades_df.columns:
                        # 如果入场时间是时间戳，转换为datetime
                        try:
                            trades_df['入场时间'] = pd.to_datetime(trades_df['入场时间'], errors='coerce')
                        except Exception as e:
                            import logging
                            logging.getLogger(__name__).debug(f"入场时间转换失败: {e}")
                        trades_df.set_index('入场时间', inplace=True)

                    # 转换出场时间为可读格式
                    # VectorBT返回的Exit Timestamp已经是datetime64[ns]类型
                    if '出场时间' in trades_df.columns:
                        try:
                            # 创建一个新Series来存储格式化后的字符串
                            exit_time_series = trades_df['出场时间'].copy()
                            # 格式化非NaT值为字符串
                            mask = exit_time_series.notna()
                            if mask.any():
                                # 使用apply逐个转换，确保返回字符串
                                formatted = exit_time_series[mask].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else x)
                                # 将整个列替换为object类型
                                trades_df['出场时间'] = exit_time_series.astype(object)
                                # 更新非NaT值
                                trades_df.loc[mask, '出场时间'] = formatted.values
                        except Exception as e:
                            import logging
                            logging.warning(f"Failed to convert exit time: {e}")

                    # 对于单股票回测，如果股票代码列全是0，则删除该列
                    if '股票代码' in trades_df.columns:
                        unique_codes = trades_df['股票代码'].unique()
                        if len(unique_codes) == 1 and (unique_codes[0] == 0 or unique_codes[0] == '0'):
                            trades_df = trades_df.drop(columns=['股票代码'])

                    # 删除不需要的列
                    columns_to_drop = ['Exit Trade Id', 'Position Id']
                    for col in columns_to_drop:
                        if col in trades_df.columns:
                            trades_df = trades_df.drop(columns=[col])

                    # 计算交易价值（入场价格 * 数量）
                    if '入场价格' in trades_df.columns and '数量' in trades_df.columns:
                        trades_df['价值'] = trades_df['入场价格'] * trades_df['数量'].abs()
        except Exception as e:
            # 记录错误但继续执行
            import logging
            logging.warning(f"Failed to extract trades from VectorBT: {str(e)}")
            import traceback
            logging.warning(traceback.format_exc())
            trades_df = None

        return {
            "quantile_returns": quantile_returns,
            "portfolio_returns": returns,
            "equity_curve": equity,
            "trades_count": int(trades_count),
            "trades": trades_df,
            # 手动计算的指标
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar_ratio),
            "win_rate": float(win_rate),
            "volatility": float(volatility),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
        }

    def multi_factor_backtest(
        self,
        df: pd.DataFrame,
        factor_names: List[str],
        weights: Optional[List[float]] = None,
        method: str = "equal_weight",
        percentile: int = 50,
        direction: str = "long",
        n_quantiles: int = 5,
        shares_per_trade: int = 100,
    ) -> Dict:
        """
        多因子组合回测（使用vectorbt）

        Args:
            df: 包含价格和因子数据的DataFrame
            factor_names: 因子名称列表
            weights: 因子权重列表（可选）
            method: 权重分配方法，"equal_weight"等权重, "ic_weight" IC加权, "risk_parity"风险平价
            percentile: 分位数阈值（0-100）
            direction: 交易方向，"long"做多或"short"做空
            n_quantiles: 分层数量

        Returns:
            Dict: 回测结果
        """
        # 确保索引是 DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)

        df = self.prepare_multi_factor_data(
            df=df,
            factor_names=factor_names,
            weights=weights,
            method=method,
        )

        # 3. 使用组合得分进行回测
        return self.single_factor_backtest(
            df=df,
            factor_name="composite_score",
            percentile=percentile,
            direction=direction,
            n_quantiles=n_quantiles,
            shares_per_trade=shares_per_trade,
        )

    def cross_sectional_backtest(
        self,
        df: pd.DataFrame,
        factor_name: str,
        top_percentile: float = 0.2,
        direction: str = "long",
    ) -> Dict:
        """
        股票池横截面回测（使用vectorbt）

        Args:
            df: 包含多只股票数据的DataFrame
            factor_name: 因子名称
            top_percentile: 选择股票的百分比（0.2表示前20%）
            direction: "long"做多或"short"做空

        Returns:
            Dict: 回测结果
        """
        # 确保索引正确
        if "date" not in df.columns:
            df = df.reset_index()

        # 透视数据：将股票代码转为列
        price_df = df.pivot(index="date", columns="stock_code", values="close")
        factor_df = df.pivot(index="date", columns="stock_code", values=factor_name)

        # 确保索引是 DatetimeIndex
        price_df.index = pd.to_datetime(price_df.index)

        # 1. 计算收益率
        # 2. 每日选择股票（横截面排名）
        selected_stocks = {}
        for date in factor_df.index:
            # 计算该日期所有股票的因子排名
            factor_values = factor_df.loc[date].dropna()
            ranks = factor_values.rank(pct=True)

            # 选择股票
            if direction == "long":
                # 做多：选择排名前 (1-top_percentile) 的股票
                selected = ranks[ranks >= (1 - top_percentile)].index.tolist()
            else:
                # 做空：选择排名后 top_percentile 的股票
                selected = ranks[ranks <= top_percentile].index.tolist()

            selected_stocks[date] = selected

        # 3. 创建目标持仓矩阵
        target_positions = pd.DataFrame(
            False,
            index=price_df.index,
            columns=price_df.columns,
        )

        for date, selected in selected_stocks.items():
            if selected:
                target_positions.loc[date, selected] = True

        # 横截面因子在 t 日收盘后观测，t+1 日起生效
        holdings = target_positions.shift(1, fill_value=False).astype(bool)
        previous_holdings = holdings.shift(1, fill_value=False).astype(bool)
        entries = holdings & ~previous_holdings
        exits = ~holdings & previous_holdings

        # 4. 使用vectorbt进行回测
        pf = vbt.Portfolio.from_signals(
            close=price_df,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            freq="D",
            cash_sharing=True,
            fees=self.commission_rate,  # 手续费率
            slippage=self.slippage,  # 滑点率
        )

        # 5. 获取结果
        equity = pf.value()
        returns = pf.returns()
        returns_clean = returns.dropna()

        # 使用 VectorBT 的 stats() 方法获取所有指标
        stats = pf.stats(agg_func=np.mean)

        # 从 stats Series 中提取指标
        # VectorBT 返回的百分比值需要除以100转换为小数
        total_return = stats.get('Total Return [%]', 0) / 100.0

        # 年化收益率：如果VectorBT返回0或NaN，则手动计算
        annual_return = stats.get('Annual Return [%]', 0) / 100.0
        if annual_return == 0 or np.isnan(annual_return):
            # 手动计算年化收益率 = (1 + 总收益率)^(252/交易天数) - 1
            n_days = len(returns_clean)
            if n_days > 0:
                annual_return = (1 + total_return) ** (252 / n_days) - 1
            else:
                annual_return = 0.0

        volatility = self._calculate_volatility(returns_clean, stats)

        sharpe_ratio = stats.get('Sharpe Ratio', 0)
        sortino_ratio = stats.get('Sortino Ratio', 0)
        max_drawdown = stats.get('Max Drawdown [%]', 0) / 100.0
        calmar_ratio = stats.get('Calmar Ratio', 0)
        win_rate = stats.get('Win Rate [%]', 0) / 100.0

        # VaR 和 CVaR 需要自己计算
        var_95, cvar_95 = self._calculate_var_cvar(returns_clean)

        # 计算交易次数（每日调仓次数）
        trades_count = stats.get('Total Trades', len(selected_stocks))

        # 提取交易记录（使用VectorBT的records_readable方法）
        trades_df = None
        try:
            # 使用records_readable获取可读的交易记录
            # records_readable返回的是pandas DataFrame
            if hasattr(pf, 'trades') and hasattr(pf.trades, 'records_readable'):
                trades_readable = pf.trades.records_readable

                if trades_readable is not None and len(trades_readable) > 0:
                    # records_readable是DataFrame，直接重命名列
                    trades_df = trades_readable.copy()

                    # 创建列名映射（英文 -> 中文）
                    column_mapping = {
                        'Trade Id': '交易ID',
                        'Column': '股票代码',
                        'Size': '数量',
                        'Entry Timestamp': '入场时间',
                        'Avg Entry Price': '入场价格',
                        'Entry Fees': '入场手续费',
                        'Exit Timestamp': '出场时间',
                        'Avg Exit Price': '出场价格',
                        'Exit Fees': '出场手续费',
                        'PnL': '收益',
                        'Return': '收益率',
                        'Direction': '方向',
                        'Status': '状态',
                        'Parent Id': '父ID'
                    }

                    # 重命名列
                    trades_df.rename(columns=column_mapping, inplace=True)

                    # 转换方向和状态为中文
                    # VectorBT records_readable: 'Long'/'Short' (字符串)
                    if '方向' in trades_df.columns:
                        trades_df['方向'] = trades_df['方向'].map({
                            'Long': '做多',
                            'Short': '做空'
                        }).fillna('未知')

                    if '状态' in trades_df.columns:
                        trades_df['状态'] = trades_df['状态'].map({
                            'Open': '持仓中',
                            'Closed': '已平仓'
                        }).fillna('未知')

                    # 将入场时间设为索引（如果存在）
                    if '入场时间' in trades_df.columns:
                        # 如果入场时间是时间戳，转换为datetime
                        try:
                            trades_df['入场时间'] = pd.to_datetime(trades_df['入场时间'], errors='coerce')
                        except Exception as e:
                            import logging
                            logging.getLogger(__name__).debug(f"入场时间转换失败: {e}")
                        trades_df.set_index('入场时间', inplace=True)

                    # 转换出场时间为可读格式
                    # VectorBT返回的Exit Timestamp已经是datetime64[ns]类型
                    if '出场时间' in trades_df.columns:
                        try:
                            # 创建一个新Series来存储格式化后的字符串
                            exit_time_series = trades_df['出场时间'].copy()
                            # 格式化非NaT值为字符串
                            mask = exit_time_series.notna()
                            if mask.any():
                                # 使用apply逐个转换，确保返回字符串
                                formatted = exit_time_series[mask].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else x)
                                # 将整个列替换为object类型
                                trades_df['出场时间'] = exit_time_series.astype(object)
                                # 更新非NaT值
                                trades_df.loc[mask, '出场时间'] = formatted.values
                        except Exception as e:
                            import logging
                            logging.warning(f"Failed to convert exit time: {e}")

                    # 对于单股票回测，如果股票代码列全是0，则删除该列
                    if '股票代码' in trades_df.columns:
                        unique_codes = trades_df['股票代码'].unique()
                        if len(unique_codes) == 1 and (unique_codes[0] == 0 or unique_codes[0] == '0'):
                            trades_df = trades_df.drop(columns=['股票代码'])

                    # 删除不需要的列
                    columns_to_drop = ['Exit Trade Id', 'Position Id']
                    for col in columns_to_drop:
                        if col in trades_df.columns:
                            trades_df = trades_df.drop(columns=[col])

                    # 计算交易价值（入场价格 * 数量）
                    if '入场价格' in trades_df.columns and '数量' in trades_df.columns:
                        trades_df['价值'] = trades_df['入场价格'] * trades_df['数量'].abs()
        except Exception as e:
            # 记录错误但继续执行
            import logging
            logging.warning(f"Failed to extract trades from VectorBT: {str(e)}")
            import traceback
            logging.warning(traceback.format_exc())
            trades_df = None

        return {
            "portfolio_returns": returns,
            "equity_curve": equity,
            "trades_count": trades_count,
            "trades": trades_df,
            "daily_selected_count": trades_count,
            "selection_by_date": {
                pd.Timestamp(date).strftime("%Y-%m-%d"): selected
                for date, selected in selected_stocks.items()
            },
            # 手动计算的指标
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar_ratio),
            "win_rate": float(win_rate),
            "volatility": float(volatility),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
        }

    def calculate_metrics(
        self, returns: pd.Series, equity_curve: pd.Series = None, annual_trading_days: int = 252, risk_free_rate: float = 0.03
    ) -> Dict:
        """
        计算性能指标

        Args:
            returns: 收益率序列
            equity_curve: 净值曲线（可选），如果提供则用它计算回撤，更准确
            annual_trading_days: 年化交易日数，默认252
            risk_free_rate: 无风险利率，默认3%

        Returns:
            Dict: 包含各种性能指标的字典
        """
        # 删除NaN值
        returns_clean = returns.dropna()

        if len(returns_clean) == 0:
            return self._empty_metrics()

        # 基础指标
        total_return = (1 + returns_clean).prod() - 1  # 复利计算累计收益
        n_days = len(returns_clean)

        # 年化收益率（使用复利公式，与 backtest_service.py 保持一致）
        if n_days > 0:
            annual_return = (1 + total_return) ** (annual_trading_days / n_days) - 1
        else:
            annual_return = 0.0

        # 波动率（年化）
        volatility = returns_clean.std() * np.sqrt(annual_trading_days)

        # 夏普比率
        daily_rf = risk_free_rate / annual_trading_days
        excess_returns = returns_clean - daily_rf
        if volatility > 0:
            sharpe_ratio = excess_returns.mean() * annual_trading_days / volatility
        else:
            sharpe_ratio = 0.0

        # 最大回撤（优先使用净值曲线计算，更准确）
        if equity_curve is not None and len(equity_curve) > 0:
            # 使用传入的净值曲线计算
            equity_array = equity_curve.values if hasattr(equity_curve, 'values') else list(equity_curve.values())
            peak = pd.Series(equity_array).cummax()
            drawdown = (peak - pd.Series(equity_array)) / peak
            max_drawdown = drawdown.max()
        else:
            # 回退方案：从收益率序列计算净值曲线（使用复利）
            equity = self.initial_capital * (1 + returns_clean).cumprod()
            peak = equity.cummax()
            drawdown = (peak - equity) / peak
            max_drawdown = drawdown.max()

        # 卡玛比率（设置最小阈值避免除零）
        if max_drawdown > 0.0001:  # 回撤至少大于0.01%
            calmar_ratio = annual_return / max_drawdown
        else:
            # 如果回撤极小，根据收益率设置卡玛比率
            if annual_return > 0:
                calmar_ratio = 999.99  # 接近无穷大
            else:
                calmar_ratio = -999.99  # 负收益且无回撤，设置为很大的负数

        # 胜率
        win_rate = (returns_clean > 0).mean()

        # 索提诺比率（只考虑下行风险）
        downside_returns = returns_clean[returns_clean < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(annual_trading_days)
            if downside_std > 0:
                sortino_ratio = (returns_clean.mean() * annual_trading_days - risk_free_rate) / downside_std
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = 0.0

        # VaR (95%置信度)
        var_95 = returns_clean.quantile(0.05)

        # CVaR (条件VaR，平均损失)
        cvar_95 = returns_clean[returns_clean <= var_95].mean()

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "sortino_ratio": sortino_ratio,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }

    def _calculate_volatility(self, returns_clean: pd.Series | pd.DataFrame, stats: pd.Series, annual_trading_days: int = 252) -> float:
        """Calculate volatility from stats or compute manually"""
        if 'Volatility (Ann.) [%]' in stats:
            return stats.get('Volatility (Ann.) [%]', 0) / 100.0

        # For multi-asset case, calculate portfolio returns volatility
        if isinstance(returns_clean, pd.DataFrame):
            portfolio_returns = returns_clean.mean(axis=1)
            return portfolio_returns.std() * np.sqrt(annual_trading_days) if len(portfolio_returns) > 0 else 0.0

        return returns_clean.std() * np.sqrt(annual_trading_days) if len(returns_clean) > 0 else 0.0

    def _calculate_var_cvar(self, returns_clean: pd.Series | pd.DataFrame) -> tuple[float, float]:
        """Calculate VaR and CVaR from returns"""
        if len(returns_clean) == 0:
            return 0.0, 0.0

        if isinstance(returns_clean, pd.DataFrame):
            portfolio_returns = returns_clean.mean(axis=1)
            var_95 = portfolio_returns.quantile(0.05)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        else:
            var_95 = returns_clean.quantile(0.05)
            cvar_95 = returns_clean[returns_clean <= var_95].mean()

        return var_95, cvar_95

    def _empty_metrics(self) -> Dict:
        """返回空的性能指标字典"""
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "sortino_ratio": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
        }


def check_vectorbt_available() -> bool:
    """检查vectorbt是否可用"""
    return VECTORBT_AVAILABLE
