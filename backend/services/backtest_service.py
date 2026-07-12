"""
回测服务核心引擎

集成了新的策略系统，支持预置策略和策略对比
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 只忽略特定类型的警告，而不是所有警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

# 导入策略系统
from backend.strategies.base_strategy import BaseStrategy
from backend.services.strategy_registry import strategy_registry
from backend.services.strategy_comparison_service import strategy_comparison_service
from backend.services.position_analysis_service import position_analysis_service
from backend.services.portfolio_analysis_service import portfolio_analysis_service
from backend.services.export_service import export_service


class BacktestService:
    """回测服务核心引擎"""

    def __init__(self, initial_capital: float = 1000000, commission_rate: float = 0.0003):
        """
        初始化回测服务

        Args:
            initial_capital: 初始资金，默认100万
            commission_rate: 手续费率，默认万三
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate

    # ==================== 单因子回测 ====================

    def single_factor_backtest(
        self,
        df: pd.DataFrame,
        factor_name: str,
        percentile: int = 50,
        direction: str = "long",
        n_quantiles: int = 5,
    ) -> Dict:
        """
        单因子分层回测

        策略逻辑：
        1. 按因子值分为5层（五分位）
        2. 每日重新分层
        3. 做多：持有因子值高的层，做空：持有因子值低的层
        4. 计算各层收益和整体收益

        Args:
            df: 包含价格和因子数据的DataFrame，必须有 close 列和因子列
            factor_name: 因子名称
            percentile: 分位数阈值（0-100），用于做多/做空判断
            direction: 交易方向，"long"做多或"short"做空
            n_quantiles: 分层数量，默认5层

        Returns:
            Dict: 包含各层收益、整体收益、净值曲线等数据的字典
        """
        df = df.copy()

        # 确保数据按日期排序
        if "date" in df.columns:
            df = df.sort_values("date")
        elif df.index.name == "date":
            df = df.sort_index()

        # 1. 计算因子分位数（滚动窗口252天）
        df["factor_rank"] = df[factor_name].rolling(
            window=252, min_periods=1
        ).rank(pct=True)

        # 2. 分层（0-1之间分为n_quantiles层，使用qcut确保等频）
        df["quantile"] = pd.qcut(
            df["factor_rank"], q=n_quantiles, labels=False, duplicates="drop"
        )

        # 3. 计算未来收益率
        df["next_return"] = df["close"].pct_change(1).shift(-1)

        # 4. 计算各层收益
        quantile_returns = {}
        for q in range(n_quantiles):
            mask = df["quantile"] == q
            layer_returns = df.loc[mask, "next_return"]
            quantile_returns[f"Q{q + 1}"] = layer_returns

        # 5. 生成交易信号（使用百分位阈值）
        percentile_threshold = percentile / 100.0
        if direction == "long":
            # 做多：因子值高于阈值的时期
            signal_mask = df["factor_rank"] >= percentile_threshold
        else:  # short
            # 做空：因子值低于阈值的时期
            signal_mask = df["factor_rank"] <= percentile_threshold

        # 6. 计算组合收益（满足条件时持仓，不满足时空仓）
        portfolio_returns = df["next_return"].copy()
        portfolio_returns[~signal_mask] = 0  # 不满足条件时空仓

        # 过滤异常值（单日收益率超过±50%的视为异常）
        portfolio_returns = portfolio_returns.clip(lower=-0.5, upper=0.5)

        # 7. 计算净值曲线
        equity = (1 + portfolio_returns.fillna(0)).cumprod() * self.initial_capital

        # 8. 计算交易次数（信号变化次数）
        signal_changes = signal_mask.astype(int).diff().abs().sum()
        trades_count = int(signal_changes)

        return {
            "quantile_returns": quantile_returns,
            "portfolio_returns": portfolio_returns,
            "equity_curve": equity,
            "trades_count": trades_count,
            "signal_mask": signal_mask,
            "factor_rank": df["factor_rank"],
        }

    def cross_sectional_backtest(
        self,
        df: pd.DataFrame,
        factor_name: str,
        top_percentile: float = 0.2,
        direction: str = "long",
    ) -> Dict:
        """
        股票池横截面回测

        策略逻辑：
        1. 每个交易日计算所有股票的因子值
        2. 选择因子值排名前N%的股票（做多）或后N%的股票（做空）
        3. 等权重配置选中股票
        4. 每日调仓

        Args:
            df: 包含多只股票数据的DataFrame，必须有 stock_code, close, date 列和因子列
            factor_name: 因子名称
            top_percentile: 选择股票的百分比（0.2表示选择前20%的股票）
            direction: "long"做多或"short"做空

        Returns:
            Dict: 回测结果
        """
        # 确保有日期索引
        if "date" not in df.columns:
            df = df.reset_index()

        # 1. 计算未来收益率
        df["next_return"] = df.groupby("stock_code")["close"].pct_change(1).shift(-1)

        # 2. 每个交易日计算因子排名
        daily_returns = []

        for date, group in df.groupby("date"):
            # 计算因子值的横截面排名
            factor_values = group[factor_name].dropna()

            if len(factor_values) == 0:
                continue

            # 计算分位数
            ranks = factor_values.rank(pct=True)

            # 选择股票
            if direction == "long":
                # 做多：选择排名前top_percentile的股票
                selected_stocks = ranks[ranks >= (1 - top_percentile)].index
            else:
                # 做空：选择排名后top_percentile的股票
                selected_stocks = ranks[ranks <= top_percentile].index

            # 获取选中股票的收益率
            selected_returns = group.loc[selected_stocks, "next_return"]

            # 等权平均收益
            if len(selected_returns) > 0:
                daily_return = selected_returns.mean()
            else:
                daily_return = 0.0

            daily_returns.append({"date": date, "return": daily_return})

        # 3. 构建收益序列
        returns_df = pd.DataFrame(daily_returns).set_index("date").sort_index()
        portfolio_returns = returns_df["return"]

        # 4. 计算净值曲线
        equity = (1 + portfolio_returns.fillna(0)).cumprod() * self.initial_capital

        # 5. 计算交易次数（每日调仓，假设选中的股票数量变化才算一次交易）
        if len(daily_returns) > 0:
            trades_count = len(daily_returns)  # 每日调仓
        else:
            trades_count = 0

        return {
            "portfolio_returns": portfolio_returns,
            "equity_curve": equity,
            "trades_count": trades_count,
            "daily_selected_count": len(daily_returns),
        }

    # ==================== 多因子回测 ====================

    def multi_factor_backtest(
        self,
        df: pd.DataFrame,
        factor_names: List[str],
        weights: Optional[List[float]] = None,
        method: str = "equal_weight",
        percentile: int = 50,
        direction: str = "long",
    ) -> Dict:
        """
        多因子组合回测

        策略逻辑：
        1. 计算组合得分（因子加权求和）
        2. 按组合得分分层
        3. 做多/做空

        Args:
            df: 包含价格和因子数据的DataFrame
            factor_names: 因子名称列表
            weights: 因子权重列表，如果为None则根据method计算
            method: 权重分配方法
                - "equal_weight": 等权重
                - "ic_weight": IC加权
                - "risk_parity": 风险平价
                - "max_sharpe": 最大夏普比率
                - "min_variance": 最小方差
            percentile: 分位数阈值
            direction: 交易方向

        Returns:
            Dict: 回测结果字典
        """
        df = df.copy()

        # 确保数据按日期排序
        if "date" in df.columns:
            df = df.sort_values("date")

        # 1. 标准化因子值（Z-score）
        for factor_name in factor_names:
            if factor_name in df.columns:
                df[f"{factor_name}_std"] = (
                    df[factor_name] - df[factor_name].rolling(252, min_periods=1).mean()
                ) / df[factor_name].rolling(252, min_periods=1).std()
                # 填充NaN
                df[f"{factor_name}_std"] = df[f"{factor_name}_std"].fillna(0)

        std_factor_names = [f"{fn}_std" for fn in factor_names]

        df["next_return"] = df["close"].pct_change(1).shift(-1)

        # 2. 计算因子权重
        if weights is None:
            if method == "equal_weight":
                weights = [1.0 / len(factor_names)] * len(factor_names)
            else:
                factor_return_frame = {}
                for factor_name, std_factor_name in zip(factor_names, std_factor_names):
                    aligned = pd.DataFrame(
                        {
                            "factor": df[std_factor_name],
                            "return": df["next_return"],
                        }
                    ).replace([np.inf, -np.inf], np.nan).dropna()

                    if aligned.empty:
                        continue

                    factor_return_frame[factor_name] = aligned["factor"] * aligned["return"]

                factor_returns_df = pd.DataFrame(factor_return_frame).dropna(how="all")
                if factor_returns_df.empty:
                    raise ValueError("无法基于真实因子收益序列计算权重")

                optimized = portfolio_analysis_service.optimize_weights(
                    factor_returns=factor_returns_df,
                    method=method,
                )
                if "error" in optimized:
                    raise ValueError(optimized["error"])

                weights = [optimized["weights"].get(name, 0.0) for name in factor_names]

        # 3. 计算组合得分
        df["composite_score"] = sum(
            df[fn] * w for fn, w in zip(std_factor_names, weights)
        )

        # 4. 计算组合得分分位数
        df["score_rank"] = (
            df["composite_score"].rolling(252, min_periods=1).rank(pct=True)
        )

        # 5. 生成交易信号
        percentile_threshold = percentile / 100.0
        if direction == "long":
            signal_mask = df["score_rank"] >= percentile_threshold
        else:
            signal_mask = df["score_rank"] <= percentile_threshold

        # 6. 计算收益
        portfolio_returns = df["next_return"].copy()
        portfolio_returns[~signal_mask] = 0

        # 7. 计算净值曲线
        equity = (1 + portfolio_returns.fillna(0)).cumprod() * self.initial_capital

        # 8. 计算交易次数
        signal_changes = signal_mask.astype(int).diff().abs().sum()
        trades_count = int(signal_changes)

        return {
            "portfolio_returns": portfolio_returns,
            "equity_curve": equity,
            "trades_count": trades_count,
            "composite_score": df["composite_score"],
            "signal_mask": signal_mask,
            "factor_weights": dict(zip(factor_names, weights)),
        }

    # ==================== 性能指标计算 ====================

    def calculate_metrics(
        self, returns: pd.Series, annual_trading_days: int = 252, risk_free_rate: float = 0.03
    ) -> Dict:
        """
        计算性能指标

        Args:
            returns: 收益率序列
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
        total_return = (1 + returns_clean).prod() - 1
        n_days = len(returns_clean)

        # 年化收益率
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

        # 最大回撤
        equity = (1 + returns_clean).cumprod()
        peak = equity.cummax()
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()

        # 卡玛比率
        if max_drawdown > 0:
            calmar_ratio = annual_return / max_drawdown
        else:
            calmar_ratio = 0.0

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

    # ==================== 回撤计算 ====================

    def calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """
        计算回撤序列

        Args:
            equity_curve: 净值曲线

        Returns:
            pd.Series: 回撤序列
        """
        peak = equity_curve.cummax()
        drawdown = (peak - equity_curve) / peak
        return drawdown

    # ==================== 信号生成 ====================

    def generate_signals(
        self,
        df: pd.DataFrame,
        factor_name: str,
        method: str = "percentile",
        threshold: float = 0.5,
        direction: str = "long",
    ) -> pd.Series:
        """
        生成交易信号

        Args:
            df: 包含因子数据的DataFrame
            factor_name: 因子名称
            method: 信号生成方法
                - "percentile": 按分位数（0-1）
                - "threshold": 按绝对阈值
            threshold: 阈值
            direction: "long" 或 "short"

        Returns:
            pd.Series: 信号序列（1为买入信号，0为无信号）
        """
        if method == "percentile":
            # 按分位数
            rank = df[factor_name].rolling(252, min_periods=1).rank(pct=True)
            if direction == "long":
                signals = (rank >= threshold).astype(int)
            else:
                signals = (rank <= threshold).astype(int)
        else:  # threshold
            # 按绝对阈值
            if direction == "long":
                signals = (df[factor_name] >= threshold).astype(int)
            else:
                signals = (df[factor_name] <= threshold).astype(int)

        return signals

    # ==================== 基准对比 ====================

    def calculate_benchmark_metrics(
        self, returns: pd.Series, benchmark_returns: pd.Series, annual_trading_days: int = 252
    ) -> Dict:
        """
        计算基准对比指标

        Args:
            returns: 策略收益序列
            benchmark_returns: 基准收益序列
            annual_trading_days: 年化交易日数

        Returns:
            Dict: 基准对比指标
        """
        # 对齐索引
        aligned_data = pd.DataFrame(
            {"strategy": returns, "benchmark": benchmark_returns}
        ).dropna()

        if len(aligned_data) == 0:
            return {"excess_return": 0.0, "tracking_error": 0.0, "information_ratio": 0.0}

        strategy_returns = aligned_data["strategy"]
        benchmark_returns = aligned_data["benchmark"]

        # 超额收益
        excess_returns = strategy_returns - benchmark_returns
        excess_return = excess_returns.mean() * annual_trading_days

        # 跟踪误差
        tracking_error = excess_returns.std() * np.sqrt(annual_trading_days)

        # 信息比率
        if tracking_error > 0:
            information_ratio = excess_return / tracking_error
        else:
            information_ratio = 0.0

        # 相关系数
        correlation = strategy_returns.corr(benchmark_returns)

        # Beta
        covariance = strategy_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        return {
            "excess_return": excess_return,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "correlation": correlation,
            "beta": beta,
        }

    # ==================== 月度收益计算 ====================

    def calculate_monthly_returns(self, returns: pd.Series) -> pd.DataFrame:
        """
        计算月度收益

        Args:
            returns: 日收益率序列，索引为日期

        Returns:
            pd.DataFrame: 月度收益率DataFrame
        """
        if len(returns) == 0:
            return pd.DataFrame()

        # 确保索引是datetime类型
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

        # 按月分组计算收益
        monthly_returns = (
            (1 + returns).resample("M").prod() - 1
        )  # 月度收益率

        # 转换为年份-月份的格式
        monthly_df = monthly_returns.to_frame(name="return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month

        # 创建透视表
        pivot_table = monthly_df.pivot(index="year", columns="month", values="return")

        return pivot_table

    # ==================== 策略系统支持 ====================

    def run_strategy(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        strategy_params: Optional[Dict] = None,
    ) -> Dict:
        """
        使用指定策略运行回测

        Args:
            df: 回测数据
            strategy_name: 策略名称
            strategy_params: 策略参数

        Returns:
            回测结果
        """
        if strategy_params is None:
            strategy_params = {}

        # 获取策略实例
        strategy = strategy_registry.get_strategy(strategy_name, **strategy_params)

        # 执行回测
        backtest_result = strategy.backtest(df)

        # 计算性能指标
        metrics = strategy.calculate_metrics(backtest_result["portfolio_returns"])

        return {
            "strategy_name": strategy_name,
            "backtest": backtest_result,
            "metrics": metrics,
        }

    def run_strategy_comparison(
        self,
        df: pd.DataFrame,
        strategy_names: List[str],
        strategy_params: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        """
        对比多个策略

        Args:
            df: 回测数据
            strategy_names: 策略名称列表
            strategy_params: 策略参数字典

        Returns:
            对比结果
        """
        return strategy_comparison_service.compare_strategies(
            df=df,
            strategy_names=strategy_names,
            strategy_params=strategy_params,
        )

    def analyze_positions(
        self,
        positions: pd.Series,
        initial_capital: float = 1000000,
    ) -> Dict:
        """
        分析持仓统计信息

        Args:
            positions: 持仓序列
            initial_capital: 初始资金

        Returns:
            持仓统计信息
        """
        return position_analysis_service.analyze_positions(
            positions=positions,
            initial_capital=initial_capital
        )

    def export_to_excel(
        self,
        backtest_result: Dict,
        output_path: str,
        strategy_name: str = "策略",
    ):
        """
        导出回测结果到Excel

        Args:
            backtest_result: 回测结果
            output_path: 输出路径
            strategy_name: 策略名称
        """
        metrics = backtest_result.get("metrics")
        export_service.export_backtest_to_excel(
            backtest_result=backtest_result,
            output_path=output_path,
            metrics=metrics,
            strategy_name=strategy_name
        )

    def export_comparison_to_excel(
        self,
        comparison_result: Dict,
        output_path: str,
    ):
        """
        导出策略对比结果到Excel

        Args:
            comparison_result: 对比结果
            output_path: 输出路径
        """
        export_service.export_comparison_to_excel(
            comparison_result=comparison_result,
            output_path=output_path
        )
