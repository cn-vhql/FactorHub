"""
时序筛选验证服务模块
"""
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, ttest_rel

logger = logging.getLogger(__name__)


def _inactive_backtest_metrics(
    trading_days: int = 0,
    active_days: int = 0,
    rebalance_count: int = 0,
    active_rebalance_count: int = 0,
) -> dict:
    return {
        "annual_return": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "win_rate": float("nan"),
        "volatility": float("nan"),
        "total_return": float("nan"),
        "active_days": float(active_days),
        "trading_days": float(trading_days),
        "active_ratio": float(active_days / trading_days) if trading_days > 0 else 0.0,
        "rebalance_count": float(rebalance_count),
        "active_rebalance_count": float(active_rebalance_count),
    }


class TemporalFilterValidationService:
    """时序筛选验证主服务"""

    def __init__(self):
        from backend.services.temporal_scoring_service import temporal_scoring_service
        from backend.services.data_service import data_service
        from backend.services.statistics_service import StatisticsService
        self.scoring_service = temporal_scoring_service
        self.data_service = data_service
        self.stats_service = StatisticsService()
        self.output_base_dir = Path("data/reports/temporal_filter_validation")
        self.score_cache_dir = Path("data/cache/temporal_scores")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.score_cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 股票池加载
    # ------------------------------------------------------------------

    def load_fixed_pool(self, csv_path: str) -> list:
        """
        加载固定股票池 CSV（含 'code' 列）。
        返回去重后的股票代码列表（6 位零填充，去除 .0 后缀）。
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {csv_path}")
        df = pd.read_csv(path)
        if "code" not in df.columns:
            raise ValueError(f"CSV 缺少 'code' 列: {csv_path}")

        def _normalize_code(c) -> str:
            s = str(c).strip()
            # 去除 pandas float 读入产生的 .0 后缀
            if s.endswith(".0"):
                s = s[:-2]
            # 纯数字补零到 6 位
            if s.isdigit():
                s = s.zfill(6)
            return s

        return (
            df["code"]
            .dropna()
            .map(_normalize_code)
            .drop_duplicates()
            .tolist()
        )

    def load_snapshot_pool(self, csv_path: str) -> pd.DataFrame:
        """
        加载历史快照股票池 CSV（含 'date', 'code' 列）。
        返回 DataFrame，date 列为 DatetimeIndex。
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {csv_path}")
        df = pd.read_csv(path)
        for col in ("date", "code"):
            if col not in df.columns:
                raise ValueError(f"CSV 缺少 '{col}' 列: {csv_path}")
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    # 面板构建
    # ------------------------------------------------------------------

    def build_score_panel(
        self,
        codes: list,
        start_date: str,
        end_date: str,
        max_workers: int = 1,
        profile_id: str | None = None,
    ) -> pd.DataFrame:
        """
        构建历史评分面板。

        若传入 profile_id，则调用 TemporalPoolService.score_pool_history 按指定 profile 打分，
        返回列：date, code, score, rank。
        否则保持原有行为（默认 8 因子），返回列：date, code, score, rank,
        trend_break, high_volatility, liquidity_risk。

        rank 为每个 date 截面内的 score 降序排名（1 = 最高分）。
        max_workers 透传到底层调用。
        """
        if profile_id is not None:
            from backend.services.temporal_pool_service import TemporalPoolService
            pool_svc = TemporalPoolService()
            panel = pool_svc.score_pool_history(
                codes, profile_id, start_date, end_date, max_workers=max_workers
            )
            return panel

        panel = self.scoring_service.score_many_stocks_history(
            codes, start_date, end_date, max_workers=max_workers
        )
        if panel.empty:
            return panel
        # 先过滤掉 score 为 NaN 的行，再计算排名
        panel = panel.dropna(subset=["score"]).copy()
        panel["rank"] = (
            panel.groupby("date")["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        cols = ["date", "code", "score", "rank", "trend_break", "high_volatility", "liquidity_risk"]
        return panel[cols].reset_index(drop=True)

    def run_temporal_pool_backtest(
        self,
        codes: list,
        profile_id: str,
        start_date: str,
        end_date: str,
        top_n: int = 10,
        score_threshold: Optional[float] = None,
        rebalance: str = "W-FRI",
        hold_days: Optional[int] = None,
        max_workers: int = 1,
    ) -> dict:
        """
        基于 TemporalPoolService 历史评分面板，模拟 TopN 持仓策略的历史回测。

        Args:
            codes: 股票代码列表
            profile_id: profile 唯一标识
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            top_n: 每期持仓数量
            score_threshold: 可选，最低分阈值，低于此分数的股票不进入持仓
            rebalance: 调仓频率，如 "W-FRI"、"2W-FRI"、"M"；hold_days 有值时优先用 hold_days
            hold_days: 可选，持仓天数（交易日），有值时用 "{hold_days}B" offset 生成调仓日期
            max_workers: 并发线程数，透传到 score_pool_history

        Returns:
            回测指标字典，包含：
            - annual_return: 年化收益率
            - sharpe: Sharpe 比率
            - max_drawdown: 最大回撤（负数）
            - win_rate: 日胜率
            - volatility: 年化波动率
            - total_return: 区间总收益率
        """
        from backend.services.temporal_pool_service import TemporalPoolService

        pool_svc = TemporalPoolService()

        # 构建历史评分面板
        score_panel = pool_svc.score_pool_history(
            codes, profile_id, start_date, end_date, max_workers=max_workers
        )

        if score_panel.empty:
            return _inactive_backtest_metrics()

        # 获取价格数据
        all_codes = score_panel["code"].unique().tolist()
        price_data = self.data_service.get_multiple_stocks_data(all_codes, start_date, end_date)

        # 生成调仓日期：hold_days 有值时用交易日 offset，否则用 rebalance 字符串
        all_trading_dates = sorted(score_panel["date"].unique())
        if hold_days and hold_days > 0:
            rebalance_offset = f"{hold_days}B"
        else:
            rebalance_offset = rebalance
        rebalance_range = pd.date_range(
            start=all_trading_dates[0],
            end=all_trading_dates[-1],
            freq=rebalance_offset,
        )

        def nearest_trading_date(target):
            candidates = [d for d in all_trading_dates if d <= target]
            return candidates[-1] if candidates else None

        # 构建每期持仓
        holdings: dict = {}
        for rb_date in rebalance_range:
            td = nearest_trading_date(rb_date)
            if td is None:
                continue
            day_panel = score_panel[score_panel["date"] == td]
            if day_panel.empty:
                continue

            # 应用 score_threshold 过滤
            if score_threshold is not None:
                day_panel = day_panel[day_panel["score"] >= score_threshold]

            # 取 TopN
            top_stocks = day_panel.nlargest(top_n, "score")["code"].tolist()
            holdings[td] = top_stocks

        if not holdings:
            return _inactive_backtest_metrics()

        # 执行等权回测
        nav_series, active_flags = self._equal_weight_backtest_with_diagnostics(
            holdings, price_data
        )
        trading_days = int(len(nav_series))
        active_days = int(active_flags.sum()) if not active_flags.empty else 0
        rebalance_count = len(holdings)
        active_rebalance_count = sum(1 for pos in holdings.values() if pos)

        if nav_series.empty or len(nav_series) < 2:
            return _inactive_backtest_metrics(
                trading_days=trading_days,
                active_days=active_days,
                rebalance_count=rebalance_count,
                active_rebalance_count=active_rebalance_count,
            )

        if active_days == 0:
            return _inactive_backtest_metrics(
                trading_days=trading_days,
                active_days=active_days,
                rebalance_count=rebalance_count,
                active_rebalance_count=active_rebalance_count,
            )

        # 计算回测指标
        daily_returns = nav_series.pct_change().dropna()

        if len(daily_returns) == 0:
            return _inactive_backtest_metrics(
                trading_days=trading_days,
                active_days=active_days,
                rebalance_count=rebalance_count,
                active_rebalance_count=active_rebalance_count,
            )

        mean_r = float(daily_returns.mean())
        std_r = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else float("nan")
        annual_return = mean_r * 252
        sharpe = mean_r / std_r * np.sqrt(252) if (std_r and not np.isnan(std_r) and std_r != 0) else float("nan")
        volatility = std_r * np.sqrt(252) if (std_r and not np.isnan(std_r)) else float("nan")
        active_return_mask = active_flags.reindex(daily_returns.index, fill_value=False).astype(bool)
        active_returns = daily_returns[active_return_mask]
        win_rate = float((active_returns > 0).mean()) if len(active_returns) > 0 else float("nan")

        # 最大回撤
        rolling_max = nav_series.cummax()
        drawdown = (nav_series - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        # 区间总收益
        total_return = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1)

        return {
            "annual_return": annual_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "volatility": volatility,
            "total_return": total_return,
            "active_days": float(active_days),
            "trading_days": float(trading_days),
            "active_ratio": float(active_days / trading_days) if trading_days > 0 else 0.0,
            "rebalance_count": float(rebalance_count),
            "active_rebalance_count": float(active_rebalance_count),
        }

    def build_forward_returns(
        self,
        codes: list,
        start_date: str,
        end_date: str,
        horizons: tuple = (5, 10, 20),
    ) -> pd.DataFrame | dict:
        """
        构建未来收益面板。
        future_return_h = close[t+h] / open[t+1] - 1
        若 open 不可用则退化为 close[t+1]

        当数据覆盖率低于 EVAL_MIN_CODE_COVERAGE 时，返回 invalid_run 字典。
        """
        from backend.core.settings import settings as _settings

        # 需要多取 max(horizons)+1 天的数据以计算未来收益
        max_h = max(horizons)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=max_h + 10)
        fetch_end = end_dt.strftime("%Y-%m-%d")

        # 优先使用带覆盖率报告的接口；若 data_service 不支持或返回非真实 report 则退化到旧接口
        # 用 try/except 而非 hasattr 检测，避免 MagicMock 把任何属性都报告为存在
        use_report = False
        report = None
        code_coverage = 1.0
        try:
            _report = self.data_service.get_multiple_stocks_data_with_report(codes, start_date, fetch_end)
            # 验证返回值确实是带 code_coverage 的真实 report 对象
            # MagicMock 的 .data 不是 dict，用此区分真实 report 与 mock
            if isinstance(_report.data, dict):
                code_coverage = float(_report.code_coverage)
                report = _report
                use_report = True
        except (AttributeError, TypeError, ValueError):
            use_report = False

        if use_report:
            # 覆盖率门禁
            reason_codes = []
            if code_coverage < _settings.EVAL_MIN_CODE_COVERAGE:
                reason_codes.append("CODE_COVERAGE_TOO_LOW")

            # 有效日期数门禁
            all_dates: set = set()
            for df in report.data.values():
                if df is not None and not df.empty:
                    all_dates.update(df.index.tolist())
            valid_date_count = len(all_dates)

            if valid_date_count < _settings.EVAL_MIN_IC_SAMPLES:
                reason_codes.append("INSUFFICIENT_IC_SAMPLES")

            if reason_codes:
                logger.warning(
                    f"build_forward_returns: 覆盖率门禁触发 {reason_codes}，"
                    f"code_coverage={code_coverage:.2%}, valid_dates={valid_date_count}，返回 invalid_run"
                )
                return {
                    "result_status": "invalid_run",
                    "reason_codes": reason_codes,
                    "code_coverage": code_coverage,
                }
            price_data = report.data
        else:
            # 旧接口路径（测试 mock / 旧调用方）
            price_data = self.data_service.get_multiple_stocks_data(codes, start_date, fetch_end)

        frames = []
        for code in codes:
            try:
                df = price_data.get(code)
                if df is None or df.empty:
                    continue

                close_series = df["close"] if "close" in df.columns else None
                open_series = df["open"] if "open" in df.columns else None

                if close_series is None:
                    continue

                # entry_price = open[t+1], fallback to close[t+1]
                if open_series is not None and not open_series.shift(-1).isna().all():
                    entry_price = open_series.shift(-1)
                else:
                    entry_price = close_series.shift(-1)

                row_data = {"code": code}
                for h in horizons:
                    future_close = close_series.shift(-h)
                    ret = future_close / entry_price - 1
                    row_data[f"future_return_{h}"] = ret

                result_df = pd.DataFrame(row_data, index=df.index)
                result_df.index.name = "date"
                result_df = result_df.reset_index()
                frames.append(result_df)
            except Exception as exc:
                logger.warning(f"股票 {code} 未来收益计算失败: {exc}")
                continue

        if not frames:
            cols = ["date", "code"] + [f"future_return_{h}" for h in horizons]
            return pd.DataFrame(columns=cols)

        combined = pd.concat(frames, ignore_index=True)
        # 过滤到 [start_date, end_date]
        combined = combined[
            (combined["date"] >= pd.Timestamp(start_date))
            & (combined["date"] <= pd.Timestamp(end_date))
        ].reset_index(drop=True)
        return combined

    def build_research_panel(
        self,
        score_panel: pd.DataFrame,
        forward_returns: pd.DataFrame | dict,
    ) -> pd.DataFrame:
        """
        合并评分面板与未来收益面板（按 date, code 内连接）。
        返回列：date, code, score, rank, future_return_5, future_return_10, future_return_20

        若 forward_returns 是 invalid_run 字典（覆盖率门禁触发），返回空 DataFrame。
        """
        # 覆盖率门禁触发时 build_forward_returns 返回 dict，直接短路
        if isinstance(forward_returns, dict):
            logger.warning(
                f"build_research_panel: forward_returns 是 invalid_run 字典 "
                f"({forward_returns.get('reason_codes', [])}），返回空面板"
            )
            return pd.DataFrame(columns=["date", "code", "score", "rank"])

        merged = score_panel.merge(forward_returns, on=["date", "code"], how="inner")
        keep_cols = ["date", "code", "score", "rank"]
        for col in forward_returns.columns:
            if col.startswith("future_return_"):
                keep_cols.append(col)
        available = [c for c in keep_cols if c in merged.columns]
        return merged[available].reset_index(drop=True)

    # ------------------------------------------------------------------
    # IC 验证
    # ------------------------------------------------------------------

    def calculate_cross_sectional_ic(
        self,
        panel: pd.DataFrame,
        horizons: tuple = (5, 10, 20),
    ) -> dict:
        """
        计算横截面 IC（Spearman 相关）。
        对每个 date，在截面内计算 spearmanr(score, future_return_h)。
        返回每个 horizon 的 ic_mean, ic_std, ir, positive_ratio, t_stat, p_value。
        """
        result = {}
        for h in horizons:
            col = f"future_return_{h}"
            if col not in panel.columns:
                continue

            ic_values = []
            for date, group in panel.groupby("date"):
                sub = group[["score", col]].dropna()
                if len(sub) < 2:
                    continue
                ic, _ = spearmanr(sub["score"], sub[col])
                if not np.isnan(ic):
                    ic_values.append(ic)

            ic_series = np.array(ic_values)
            key = f"ic_{h}"

            if len(ic_series) == 0:
                result[key] = {
                    "ic_mean": float("nan"),
                    "ic_std": float("nan"),
                    "ir": float("nan"),
                    "positive_ratio": float("nan"),
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "insufficient_samples": True,
                }
                continue

            ic_mean = float(np.mean(ic_series))
            ic_std = float(np.std(ic_series, ddof=1)) if len(ic_series) > 1 else float("nan")
            ir = ic_mean / ic_std if (ic_std and not np.isnan(ic_std) and ic_std != 0) else float("nan")
            positive_ratio = float(np.mean(ic_series > 0))

            entry: dict = {
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ir": ir,
                "positive_ratio": positive_ratio,
            }

            if len(ic_series) >= 2:
                t_stat, p_value = ttest_1samp(ic_series, 0)
                entry["t_stat"] = float(t_stat)
                entry["p_value"] = float(p_value)
            else:
                entry["t_stat"] = float("nan")
                entry["p_value"] = float("nan")

            if len(ic_series) < 30:
                entry["insufficient_samples"] = True

            result[key] = entry

        return result

    # ------------------------------------------------------------------
    # 分层分析
    # ------------------------------------------------------------------

    def analyze_quantile_spread(
        self,
        panel: pd.DataFrame,
        n_quantiles: int = 5,
        horizons: tuple = (5, 10, 20),
    ) -> dict:
        """
        分层收益验证。
        对每个 date，按 score 将股票分成 n_quantiles 组（Q1 = 最高分组）。
        统计各组未来收益的均值、年化收益、Sharpe、胜率、样本数。
        计算 spread（Q1 - Q_last）及其统计显著性。
        """
        result = {}
        for h in horizons:
            col = f"future_return_{h}"
            if col not in panel.columns:
                continue

            # 收集每个分位组的收益序列
            quantile_returns: dict = {f"Q{i+1}": [] for i in range(n_quantiles)}

            for date, group in panel.groupby("date"):
                sub = group[["score", col]].dropna()
                if len(sub) < n_quantiles:
                    continue
                try:
                    labels = pd.qcut(sub["score"], n_quantiles, labels=False, duplicates="drop")
                except Exception:
                    continue
                # labels: 0 = lowest score, n_quantiles-1 = highest score
                # Q1 = highest score = label n_quantiles-1
                sub = sub.copy()
                sub["_label"] = labels
                for q_idx in range(n_quantiles):
                    # invert: Q1 = highest score group
                    score_label = n_quantiles - 1 - q_idx
                    q_name = f"Q{q_idx + 1}"
                    mask = sub["_label"] == score_label
                    quantile_returns[q_name].extend(sub.loc[mask, col].tolist())

            horizon_result = {}
            for q_name, returns in quantile_returns.items():
                arr = np.array([r for r in returns if not np.isnan(r)])
                if len(arr) == 0:
                    horizon_result[q_name] = {
                        "mean_return": float("nan"),
                        "annual_return": float("nan"),
                        "sharpe": float("nan"),
                        "win_rate": float("nan"),
                        "sample_count": 0,
                    }
                    continue
                mean_ret = float(np.mean(arr))
                std_ret = float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")
                annual_ret = mean_ret * 252
                sharpe = mean_ret / std_ret * np.sqrt(252) if (std_ret and not np.isnan(std_ret) and std_ret != 0) else float("nan")
                win_rate = float(np.mean(arr > 0))
                horizon_result[q_name] = {
                    "mean_return": mean_ret,
                    "annual_return": annual_ret,
                    "sharpe": sharpe,
                    "win_rate": win_rate,
                    "sample_count": len(arr),
                }

            # Spread: Q1 - Q_last
            q1_returns = np.array([r for r in quantile_returns["Q1"] if not np.isnan(r)])
            qlast_name = f"Q{n_quantiles}"
            qlast_returns = np.array([r for r in quantile_returns[qlast_name] if not np.isnan(r)])

            spread_entry: dict = {}
            if len(q1_returns) > 0 and len(qlast_returns) > 0:
                spread_mean = float(np.mean(q1_returns) - np.mean(qlast_returns))
                spread_entry["mean_spread"] = spread_mean
                # t-test on spread (independent samples)
                min_len = min(len(q1_returns), len(qlast_returns))
                if min_len >= 2:
                    from scipy.stats import ttest_ind
                    t_stat, p_value = ttest_ind(q1_returns, qlast_returns)
                    spread_entry["t_stat"] = float(t_stat)
                    spread_entry["p_value"] = float(p_value)
                    spread_entry["significant"] = bool(p_value < 0.05)
                else:
                    spread_entry["t_stat"] = float("nan")
                    spread_entry["p_value"] = float("nan")
                    spread_entry["significant"] = False
            else:
                spread_entry["mean_spread"] = float("nan")
                spread_entry["t_stat"] = float("nan")
                spread_entry["p_value"] = float("nan")
                spread_entry["significant"] = False

            horizon_result["spread"] = spread_entry
            result[f"horizon_{h}"] = horizon_result

        return result

    # ------------------------------------------------------------------
    # 组合回测
    # ------------------------------------------------------------------

    def _equal_weight_backtest(
        self,
        holdings_by_date: dict,
        price_data: dict,
    ) -> pd.Series:
        nav_series, _ = self._equal_weight_backtest_with_diagnostics(
            holdings_by_date, price_data
        )
        return nav_series

    def _equal_weight_backtest_with_diagnostics(
        self,
        holdings_by_date: dict,
        price_data: dict,
    ) -> tuple[pd.Series, pd.Series]:
        """
        等权再平衡回测器。
        holdings_by_date: {rebalance_date -> [code1, code2, ...]}
        price_data: {code -> DataFrame with close prices (DatetimeIndex)}
        返回 (每日净值序列, 是否持仓的布尔序列)。
        """
        if not holdings_by_date or not price_data:
            return pd.Series(dtype=float), pd.Series(dtype=bool)

        all_dates = sorted(set().union(*[df.index for df in price_data.values() if not df.empty]))
        if not all_dates:
            return pd.Series(dtype=float), pd.Series(dtype=bool)

        rebalance_dates = sorted(holdings_by_date.keys())

        nav = 1.0
        nav_series = {}
        active_series = {}
        current_holdings_set: set = set()
        current_weights: dict = {}
        prev_prices: dict = {}

        for date in all_dates:
            applicable = [rd for rd in rebalance_dates if rd <= date]
            if applicable:
                latest_rebalance = max(applicable)
                new_holdings = holdings_by_date[latest_rebalance]
                new_set = set(new_holdings)
                # 用集合比较，避免顺序敏感触发不必要的再平衡
                if new_set != current_holdings_set:
                    current_holdings_set = new_set
                    valid = [c for c in new_holdings if c in price_data and not price_data[c].empty]
                    if valid:
                        w = 1.0 / len(valid)
                        current_weights = {c: w for c in valid}
                    else:
                        current_weights = {}
                    prev_prices = {}
                    for c in current_weights:
                        df = price_data[c]
                        avail = df.index[df.index <= date]
                        if len(avail) > 0:
                            prev_prices[c] = float(df.loc[avail[-1], "close"])

            if current_weights and prev_prices:
                port_ret = 0.0
                for c, w in current_weights.items():
                    if c not in price_data or price_data[c].empty:
                        continue
                    df = price_data[c]
                    if date not in df.index:
                        continue
                    cur_price = float(df.loc[date, "close"])
                    pp = prev_prices.get(c)
                    if pp and pp > 0:
                        port_ret += w * (cur_price / pp - 1)
                    prev_prices[c] = cur_price
                nav = nav * (1 + port_ret)

            nav_series[date] = nav
            active_series[date] = bool(current_weights)

        return pd.Series(nav_series), pd.Series(active_series, dtype=bool)

    def run_portfolio_backtest(
        self,
        panel: pd.DataFrame,
        top_n: int = 10,
        rebalance: str = "W-FRI",
        score_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        组合回测，输出 4 组策略的净值曲线。
        策略 A: universe100_eq  - 全部等权
        策略 B: random10_avg    - 随机抽 top_n（seed=42，20次均值）
        策略 C: top10_score     - 评分最高前 top_n 等权
        策略 D: top10_score_threshold - score >= threshold 后取前 top_n，不足留现金
        """
        if panel.empty:
            return pd.DataFrame(columns=["date", "universe100_eq", "random10_avg", "top10_score", "top10_score_threshold"])

        start_date = panel["date"].min().strftime("%Y-%m-%d")
        end_date = panel["date"].max().strftime("%Y-%m-%d")

        # Fetch price data for all codes
        all_codes = panel["code"].unique().tolist()
        price_data = self.data_service.get_multiple_stocks_data(all_codes, start_date, end_date)

        # Generate rebalance dates
        all_trading_dates = sorted(panel["date"].unique())
        rebalance_range = pd.date_range(
            start=all_trading_dates[0],
            end=all_trading_dates[-1],
            freq=rebalance,
        )

        # For each rebalance date, find the closest available trading date
        trading_date_set = set(all_trading_dates)

        def nearest_trading_date(target):
            candidates = [d for d in all_trading_dates if d <= target]
            return candidates[-1] if candidates else None

        holdings_A: dict = {}
        holdings_B_runs: list = []  # list of 20 dicts
        holdings_C: dict = {}
        holdings_D: dict = {}

        # Pre-generate random samples for strategy B
        rng = np.random.default_rng(42)

        for rb_date in rebalance_range:
            td = nearest_trading_date(rb_date)
            if td is None:
                continue
            day_panel = panel[panel["date"] == td]
            if day_panel.empty:
                continue

            codes_today = day_panel["code"].tolist()

            # Strategy A: all codes equal weight
            holdings_A[td] = codes_today

            # Strategy C: top top_n by score
            top_c = day_panel.nlargest(top_n, "score")["code"].tolist()
            holdings_C[td] = top_c

            # Strategy D: filter by threshold, then top top_n
            if score_threshold is not None:
                filtered = day_panel[day_panel["score"] >= score_threshold]
            else:
                filtered = day_panel
            top_d = filtered.nlargest(top_n, "score")["code"].tolist()
            holdings_D[td] = top_d  # may be fewer than top_n

        # Strategy B: 20 random runs
        for run_i in range(20):
            run_holdings: dict = {}
            for rb_date in rebalance_range:
                td = nearest_trading_date(rb_date)
                if td is None:
                    continue
                day_panel = panel[panel["date"] == td]
                if day_panel.empty:
                    continue
                codes_today = day_panel["code"].tolist()
                n_sample = min(top_n, len(codes_today))
                sampled = rng.choice(codes_today, size=n_sample, replace=False).tolist()
                run_holdings[td] = sampled
            holdings_B_runs.append(run_holdings)

        # Run backtests
        nav_A = self._equal_weight_backtest(holdings_A, price_data)
        nav_C = self._equal_weight_backtest(holdings_C, price_data)

        # Strategy D with cash for missing positions
        nav_D = self._equal_weight_backtest_with_cash(holdings_D, price_data, top_n)

        # Strategy B: average of 20 runs
        nav_B_list = []
        for run_holdings in holdings_B_runs:
            nav_run = self._equal_weight_backtest(run_holdings, price_data)
            nav_B_list.append(nav_run)

        if nav_B_list:
            nav_B_df = pd.concat(nav_B_list, axis=1).ffill()
            nav_B = nav_B_df.mean(axis=1)
        else:
            nav_B = pd.Series(dtype=float)

        # Align all series to common dates
        all_idx = nav_A.index.union(nav_C.index).union(nav_B.index).union(nav_D.index)
        result_df = pd.DataFrame(index=all_idx)
        result_df["universe100_eq"] = nav_A.reindex(all_idx).ffill()
        result_df["random10_avg"] = nav_B.reindex(all_idx).ffill()
        result_df["top10_score"] = nav_C.reindex(all_idx).ffill()
        result_df["top10_score_threshold"] = nav_D.reindex(all_idx).ffill()
        result_df.index.name = "date"
        result_df = result_df.reset_index()
        return result_df

    def _equal_weight_backtest_with_cash(
        self,
        holdings_by_date: dict,
        price_data: dict,
        top_n: int,
    ) -> pd.Series:
        """
        Strategy D variant: each stock gets weight 1/top_n, remaining is cash (return 0).
        """
        if not holdings_by_date or not price_data:
            return pd.Series(dtype=float)

        all_dates = sorted(set().union(*[df.index for df in price_data.values() if not df.empty]))
        if not all_dates:
            return pd.Series(dtype=float)

        rebalance_dates = sorted(holdings_by_date.keys())

        nav = 1.0
        nav_series = {}
        current_holdings_set: set = set()
        current_weights: dict = {}
        prev_prices: dict = {}

        for date in all_dates:
            applicable = [rd for rd in rebalance_dates if rd <= date]
            if applicable:
                latest_rebalance = max(applicable)
                new_holdings = holdings_by_date[latest_rebalance]
                new_set = set(new_holdings)
                # 用集合比较，避免顺序敏感触发不必要的再平衡
                if new_set != current_holdings_set:
                    current_holdings_set = new_set
                    valid = [c for c in new_holdings if c in price_data and not price_data[c].empty]
                    # Each stock gets 1/top_n weight; remaining is cash
                    w = 1.0 / top_n if top_n > 0 else 0.0
                    current_weights = {c: w for c in valid}
                    prev_prices = {}
                    for c in current_weights:
                        df = price_data[c]
                        avail = df.index[df.index <= date]
                        if len(avail) > 0:
                            prev_prices[c] = float(df.loc[avail[-1], "close"])

            if current_weights and prev_prices:
                port_ret = 0.0
                for c, w in current_weights.items():
                    if c not in price_data or price_data[c].empty:
                        continue
                    df = price_data[c]
                    if date not in df.index:
                        continue
                    cur_price = float(df.loc[date, "close"])
                    pp = prev_prices.get(c)
                    if pp and pp > 0:
                        port_ret += w * (cur_price / pp - 1)
                    prev_prices[c] = cur_price
                nav = nav * (1 + port_ret)

            nav_series[date] = nav

        return pd.Series(nav_series)

    # ------------------------------------------------------------------
    # 策略对比与统计检验
    # ------------------------------------------------------------------

    def compare_strategies(self, curves_or_returns: pd.DataFrame) -> dict:
        """
        统计显著性检验与策略对比。
        输入：DataFrame，date 列（或 index）+ 策略 NAV 曲线列。
        返回：metrics, statistical_tests, conclusion。
        """
        df = curves_or_returns.copy()
        if "date" in df.columns:
            df = df.set_index("date")

        # Compute daily returns from NAV curves
        daily_returns = df.pct_change().dropna()

        metrics = {}
        for col in daily_returns.columns:
            ret = daily_returns[col].dropna()
            if len(ret) == 0:
                metrics[col] = {
                    "annual_return": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown": float("nan"),
                    "volatility": float("nan"),
                    "win_rate": float("nan"),
                }
                continue

            mean_r = float(ret.mean())
            std_r = float(ret.std(ddof=1)) if len(ret) > 1 else float("nan")
            annual_return = mean_r * 252
            sharpe = mean_r / std_r * np.sqrt(252) if (std_r and not np.isnan(std_r) and std_r != 0) else float("nan")
            volatility = std_r * np.sqrt(252) if (std_r and not np.isnan(std_r)) else float("nan")
            win_rate = float((ret > 0).mean())

            # Max drawdown from NAV curve
            nav_col = df[col].dropna()
            if len(nav_col) > 0:
                rolling_max = nav_col.cummax()
                drawdown = (nav_col - rolling_max) / rolling_max
                max_drawdown = float(drawdown.min())
            else:
                max_drawdown = float("nan")

            entry: dict = {
                "annual_return": annual_return,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "win_rate": win_rate,
            }
            if len(ret) < 30:
                entry["insufficient_samples"] = True
            metrics[col] = entry

        # Statistical tests
        statistical_tests = {}

        def _paired_test(col_a: str, col_b: str, test_name: str):
            if col_a not in daily_returns.columns or col_b not in daily_returns.columns:
                return
            r_a = daily_returns[col_a].dropna()
            r_b = daily_returns[col_b].dropna()
            common_idx = r_a.index.intersection(r_b.index)
            if len(common_idx) < 2:
                statistical_tests[test_name] = {
                    "paired_t_stat": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                    "ir": float("nan"),
                    "insufficient_samples": True,
                }
                return
            ra = r_a.loc[common_idx]
            rb = r_b.loc[common_idx]
            excess = ra - rb
            t_stat, p_value = ttest_rel(ra, rb)
            excess_mean = float(excess.mean())
            excess_std = float(excess.std(ddof=1)) if len(excess) > 1 else float("nan")
            ir = excess_mean / excess_std if (excess_std and not np.isnan(excess_std) and excess_std != 0) else float("nan")
            entry: dict = {
                "paired_t_stat": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "ir": ir,
            }
            if len(common_idx) < 30:
                entry["insufficient_samples"] = True
            statistical_tests[test_name] = entry

        _paired_test("top10_score", "universe100_eq", "top10_vs_universe100")
        _paired_test("top10_score", "random10_avg", "top10_vs_random10")

        # Build conclusion text
        conclusion_parts = []
        for test_name, test_result in statistical_tests.items():
            sig = test_result.get("significant", False)
            p = test_result.get("p_value", float("nan"))
            ir = test_result.get("ir", float("nan"))
            if not np.isnan(p):
                sig_text = "显著优于" if sig else "未显著优于"
                conclusion_parts.append(
                    f"{test_name}: top10_score {sig_text}基准（p={p:.4f}, IR={ir:.3f}）"
                )
        conclusion = "；".join(conclusion_parts) if conclusion_parts else "样本不足，无法得出结论"

        return {
            "metrics": metrics,
            "statistical_tests": statistical_tests,
            "conclusion": conclusion,
        }

    # ------------------------------------------------------------------
    # 结果输出
    # ------------------------------------------------------------------

    def save_results(
        self,
        run_id: str,
        score_panel: pd.DataFrame,
        forward_returns: pd.DataFrame,
        ic_results: dict,
        quantile_results: dict,
        portfolio_curves: pd.DataFrame,
        strategy_comparison: dict,
        factor_expression_source: str,
    ) -> Path:
        """
        将所有验证结果写入输出目录。
        返回输出目录 Path。
        """
        out_dir = self.output_base_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # score_panel.parquet
        score_panel.to_parquet(out_dir / "score_panel.parquet", index=False)

        # forward_returns.parquet
        forward_returns.to_parquet(out_dir / "forward_returns.parquet", index=False)

        # ic_summary.csv
        ic_rows = []
        for key, val in ic_results.items():
            horizon = key.replace("ic_", "")
            ic_rows.append({
                "horizon": horizon,
                "ic_mean": val.get("ic_mean"),
                "ic_std": val.get("ic_std"),
                "ir": val.get("ir"),
                "positive_ratio": val.get("positive_ratio"),
                "t_stat": val.get("t_stat"),
                "p_value": val.get("p_value"),
            })
        pd.DataFrame(ic_rows).to_csv(out_dir / "ic_summary.csv", index=False)

        # quantile_summary.csv
        q_rows = []
        for horizon_key, horizon_val in quantile_results.items():
            horizon = horizon_key.replace("horizon_", "")
            for q_name, q_val in horizon_val.items():
                if q_name == "spread":
                    continue
                q_rows.append({
                    "horizon": horizon,
                    "quantile": q_name,
                    "mean_return": q_val.get("mean_return"),
                    "annual_return": q_val.get("annual_return"),
                    "sharpe": q_val.get("sharpe"),
                    "win_rate": q_val.get("win_rate"),
                    "sample_count": q_val.get("sample_count"),
                })
        pd.DataFrame(q_rows).to_csv(out_dir / "quantile_summary.csv", index=False)

        # portfolio_curves.csv
        portfolio_curves.to_csv(out_dir / "portfolio_curves.csv", index=False)

        # strategy_comparison.json
        def _make_serializable(obj):
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_serializable(v) for v in obj]
            if isinstance(obj, float) and np.isnan(obj):
                return None
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return obj

        with open(out_dir / "strategy_comparison.json", "w", encoding="utf-8") as f:
            json.dump(_make_serializable(strategy_comparison), f, ensure_ascii=False, indent=2)

        # scoring_errors.csv
        errors = getattr(self.scoring_service, "_last_history_errors", [])
        pd.DataFrame(errors if errors else [], columns=["code", "error"]).to_csv(
            out_dir / "scoring_errors.csv", index=False
        )

        # validation_report.md
        self._write_validation_report(
            out_dir, run_id, ic_results, quantile_results,
            strategy_comparison, factor_expression_source
        )

        # run_manifest.json
        from backend.services.champion_strategy_service import _write_run_manifest
        _write_run_manifest(
            out_dir,
            module="temporal_filter_validation",
            data_coverage=1.0,
            quality_gate=[],
            result_status="completed",
        )

        return out_dir

    def _write_validation_report(
        self,
        out_dir: Path,
        run_id: str,
        ic_results: dict,
        quantile_results: dict,
        strategy_comparison: dict,
        factor_expression_source: str,
    ) -> None:
        """生成可读的验证报告 Markdown 文件。"""
        lines = [
            "# 时序筛选验证报告",
            "",
            f"**Run ID:** {run_id}",
            f"**因子表达式来源 (factor_expression_source):** `{factor_expression_source}`",
            "",
            "> 注意：本报告使用统计验证语言，不作确定性结论。所有结果仅供参考，不构成投资建议。",
            "",
            "---",
            "",
            "## 一、横截面 IC 分析",
            "",
        ]

        for key, val in ic_results.items():
            horizon = key.replace("ic_", "")
            lines.append(f"### Horizon {horizon} 日")
            if val.get("insufficient_samples"):
                lines.append("- ⚠️ 样本量不足（< 30），结果仅供参考")
            lines.append(f"- IC 均值: {val.get('ic_mean', 'N/A'):.4f}" if isinstance(val.get('ic_mean'), float) and not np.isnan(val.get('ic_mean', float('nan'))) else "- IC 均值: N/A")
            lines.append(f"- IC 标准差: {val.get('ic_std', 'N/A'):.4f}" if isinstance(val.get('ic_std'), float) and not np.isnan(val.get('ic_std', float('nan'))) else "- IC 标准差: N/A")
            lines.append(f"- IR: {val.get('ir', 'N/A'):.4f}" if isinstance(val.get('ir'), float) and not np.isnan(val.get('ir', float('nan'))) else "- IR: N/A")
            lines.append(f"- 正 IC 比例: {val.get('positive_ratio', 'N/A'):.2%}" if isinstance(val.get('positive_ratio'), float) and not np.isnan(val.get('positive_ratio', float('nan'))) else "- 正 IC 比例: N/A")
            p = val.get('p_value')
            if isinstance(p, float) and not np.isnan(p):
                sig = "显著（p < 0.05）" if p < 0.05 else "不显著（p >= 0.05）"
                lines.append(f"- p 值: {p:.4f}（{sig}）")
            lines.append("")

        lines += [
            "---",
            "",
            "## 二、分层收益分析",
            "",
        ]

        for horizon_key, horizon_val in quantile_results.items():
            horizon = horizon_key.replace("horizon_", "")
            lines.append(f"### Horizon {horizon} 日")
            for q_name, q_val in horizon_val.items():
                if q_name == "spread":
                    continue
                mr = q_val.get('mean_return')
                ar = q_val.get('annual_return')
                lines.append(f"- {q_name}: 均值收益={mr:.4f}" if isinstance(mr, float) and not np.isnan(mr) else f"- {q_name}: 均值收益=N/A")
            spread = horizon_val.get("spread", {})
            ms = spread.get("mean_spread")
            if isinstance(ms, float) and not np.isnan(ms):
                lines.append(f"- Spread (Q1-Q_last): {ms:.4f}")
            lines.append("")

        lines += [
            "---",
            "",
            "## 三、组合回测与策略对比",
            "",
        ]

        metrics = strategy_comparison.get("metrics", {})
        for strat, m in metrics.items():
            ar = m.get('annual_return')
            sh = m.get('sharpe')
            md = m.get('max_drawdown')
            lines.append(f"### {strat}")
            lines.append(f"- 年化收益: {ar:.2%}" if isinstance(ar, float) and not np.isnan(ar) else "- 年化收益: N/A")
            lines.append(f"- Sharpe: {sh:.3f}" if isinstance(sh, float) and not np.isnan(sh) else "- Sharpe: N/A")
            lines.append(f"- 最大回撤: {md:.2%}" if isinstance(md, float) and not np.isnan(md) else "- 最大回撤: N/A")
            lines.append("")

        lines += [
            "---",
            "",
            "## 四、统计显著性检验",
            "",
        ]

        for test_name, test_val in strategy_comparison.get("statistical_tests", {}).items():
            p = test_val.get("p_value")
            ir = test_val.get("ir")
            sig = test_val.get("significant", False)
            lines.append(f"### {test_name}")
            if test_val.get("insufficient_samples"):
                lines.append("- ⚠️ 样本量不足（< 30），结果仅供参考")
            lines.append(f"- p 值: {p:.4f}" if isinstance(p, float) and not np.isnan(p) else "- p 值: N/A")
            lines.append(f"- IR: {ir:.4f}" if isinstance(ir, float) and not np.isnan(ir) else "- IR: N/A")
            lines.append(f"- 显著性: {'显著（p < 0.05）' if sig else '不显著（p >= 0.05）'}")
            lines.append("")

        conclusion = strategy_comparison.get("conclusion", "")
        lines += [
            "---",
            "",
            "## 五、综合结论",
            "",
            conclusion,
            "",
            "> 以上结论基于统计检验，存在不确定性，不应作为唯一决策依据。",
        ]

        (out_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


# 全局单例
temporal_filter_validation_service = TemporalFilterValidationService()
