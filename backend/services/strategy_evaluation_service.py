"""
StrategyEvaluationService
对候选策略执行 walk-forward 回测并计算稳定性总分。
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_EVAL_RESULTS_DIR = Path("data/reports/strategy_evaluation")


def _parse_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


def _fmt(d: date) -> str:
    return d.strftime("%Y-%m-%d")


class StrategyEvaluationService:
    """对候选策略执行 walk-forward 回测并计算稳定性总分的服务。"""

    def __init__(self) -> None:
        from backend.services.temporal_filter_validation_service import (
            TemporalFilterValidationService,
        )

        self._tfvs = TemporalFilterValidationService()

    # ------------------------------------------------------------------
    # 窗口生成
    # ------------------------------------------------------------------

    def _generate_walk_forward_windows(
        self,
        start_date: str,
        end_date: str,
        train_months: int = 12,
        valid_months: int = 3,
        test_months: int = 3,
        step_months: int = 3,
    ) -> list[dict]:
        """生成无重叠滚动窗口列表。

        每个窗口包含 train_start, train_end, valid_start, valid_end,
        test_start, test_end 六个字段（均为 "YYYY-MM-DD" 字符串）。
        窗口间无重叠，整体覆盖完整时间范围。

        Args:
            start_date: 整体起始日期
            end_date: 整体结束日期
            train_months: 训练期月数
            valid_months: 验证期月数
            test_months: 测试期月数
            step_months: 滚动步长月数

        Returns:
            窗口字典列表，按时间顺序排列。
        """
        start = _parse_date(start_date)
        end = _parse_date(end_date)

        window_total_months = train_months + valid_months + test_months
        windows: list[dict] = []
        window_id = 0

        cursor = start
        while True:
            train_start = cursor
            train_end = train_start + relativedelta(months=train_months) - relativedelta(days=1)
            valid_start = train_end + relativedelta(days=1)
            valid_end = valid_start + relativedelta(months=valid_months) - relativedelta(days=1)
            test_start = valid_end + relativedelta(days=1)
            test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)

            # 测试期结束不能超过整体结束日期
            if test_end > end:
                break

            windows.append(
                {
                    "window_id": window_id,
                    "train_start": _fmt(train_start),
                    "train_end": _fmt(train_end),
                    "valid_start": _fmt(valid_start),
                    "valid_end": _fmt(valid_end),
                    "test_start": _fmt(test_start),
                    "test_end": _fmt(test_end),
                }
            )
            window_id += 1
            cursor = cursor + relativedelta(months=step_months)

        return windows

    # ------------------------------------------------------------------
    # 稳定性总分
    # ------------------------------------------------------------------

    def compute_stability_score(
        self, window_metrics: list[dict], min_valid_windows: int = 3
    ) -> float | dict:
        """按 7 项权重计算稳定性总分（返回值 ∈ [0, 100]）。

        权重分配：
          30% OOS Sharpe（Sharpe=2 满分）
          20% OOS 超额收益（超额15%满分）
          15% 回撤惩罚（回撤30%得0分）
          10% IC/IR（IR=1.5 满分）
          10% 正收益窗口比例
          10% 风格阶段一致性（窗口间 Sharpe 标准差）
          5%  换手率惩罚（换手率100%得0分）

        全负 Sharpe 时返回值 < 30。

        当有效窗口数（test_metrics 中 sharpe 非 NaN 的窗口）< min_valid_windows 时，
        返回包含 stability_score=NaN、evaluation_status="invalid" 的字典。

        Args:
            window_metrics: 每个窗口的指标字典列表，每个字典需含 test_metrics 字段。
            min_valid_windows: 最少有效窗口数，默认 3。

        Returns:
            有效时返回稳定性总分 float，范围 [0.0, 100.0]；
            无效时返回 dict，含 stability_score=NaN、evaluation_status="invalid"。
        """
        if not window_metrics:
            return 0.0

        # 统计有效窗口数：test_metrics 中 sharpe 非 NaN 的窗口
        def _is_nan_val(v) -> bool:
            return v is None or (isinstance(v, float) and np.isnan(v))

        valid_count = sum(
            1 for w in window_metrics
            if not _is_nan_val(w.get("test_metrics", {}).get("sharpe"))
        )

        if valid_count < min_valid_windows:
            return {
                "stability_score": float("nan"),
                "evaluation_status": "invalid",
                "invalid_reason": "INSUFFICIENT_VALID_WINDOWS",
            }

        def _safe(val, default=0.0):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)

        def clip01(x: float) -> float:
            return max(0.0, min(1.0, x))

        test_sharpes: list[float] = []
        test_excess_returns: list[float] = []
        test_drawdowns: list[float] = []
        test_irs: list[float] = []
        positive_windows: list[float] = []
        turnover_rates: list[float] = []
        activity_ratios: list[float] = []

        for w in window_metrics:
            tm = w.get("test_metrics", {}) or {}
            sharpe = _safe(tm.get("sharpe"))
            annual_return = _safe(tm.get("annual_return"))
            max_drawdown = _safe(tm.get("max_drawdown"))
            ir = _safe(tm.get("ir"))
            # excess_return: 若无则用 annual_return 近似
            excess_return = _safe(tm.get("excess_return", tm.get("annual_return")))
            turnover = _safe(tm.get("turnover_rate"), default=0.5)
            activity_ratio = clip01(_safe(tm.get("active_ratio"), default=1.0))

            test_sharpes.append(sharpe)
            test_excess_returns.append(excess_return)
            test_drawdowns.append(max_drawdown)
            test_irs.append(ir)
            positive_windows.append(1.0 if annual_return > 0 else 0.0)
            turnover_rates.append(turnover)
            activity_ratios.append(activity_ratio)

        mean_sharpe = float(np.mean(test_sharpes))
        mean_excess = float(np.mean(test_excess_returns))
        mean_drawdown = float(np.mean([abs(d) for d in test_drawdowns]))
        mean_ir = float(np.mean(test_irs))
        positive_ratio = float(np.mean(positive_windows))
        mean_turnover = float(np.mean(turnover_rates))
        mean_activity = float(np.mean(activity_ratios))

        # 单股时若整个测试期从未真正持仓，不应因为“零回撤/零波动”而获得稳定性高分。
        if mean_activity <= 0:
            return 0.0

        sharpe_score = clip01(mean_sharpe / 2.0)
        excess_score = clip01(mean_excess / 0.15)
        drawdown_score = clip01(1.0 - mean_drawdown / 0.30)
        ir_score = clip01(mean_ir / 1.5)

        # 风格阶段一致性：窗口间 Sharpe 标准差越小越好
        if len(test_sharpes) > 1:
            sharpe_std = float(np.std(test_sharpes, ddof=1))
            mean_abs_sharpe = float(np.mean([abs(s) for s in test_sharpes])) + 0.01
            regime_score = clip01(1.0 - sharpe_std / mean_abs_sharpe)
        else:
            regime_score = 1.0

        turnover_score = clip01(1.0 - mean_turnover / 1.0)

        raw_score = (
            0.30 * sharpe_score
            + 0.20 * excess_score
            + 0.15 * drawdown_score
            + 0.10 * ir_score
            + 0.10 * positive_ratio
            + 0.10 * regime_score
            + 0.05 * turnover_score
        )

        score = float(np.clip(raw_score * 100.0, 0.0, 100.0))

        # 全负 Sharpe 时强制压低到 30 以下
        if all(s <= 0 for s in test_sharpes):
            score = min(score, 29.99)

        return score

    # ------------------------------------------------------------------
    # 时序池候选评价
    # ------------------------------------------------------------------

    def evaluate_temporal_pool_candidate(
        self,
        candidate: dict,
        codes: list[str],
        start_date: str,
        end_date: str,
        train_months: int = 12,
        valid_months: int = 3,
        test_months: int = 3,
        step_months: int = 3,
        max_workers: int = 1,
    ) -> dict:
        """时序池候选策略 walk-forward 评价。

        Args:
            candidate: 候选策略配置字典，需含 profile_id 或 config.id 字段
            codes: 股票代码列表
            start_date: 整体起始日期
            end_date: 整体结束日期
            train_months: 训练期月数
            valid_months: 验证期月数
            test_months: 测试期月数
            step_months: 滚动步长月数

        Returns:
            {
                train_metrics: 所有窗口训练期指标均值,
                valid_metrics: 所有窗口验证期指标均值,
                test_metrics: 所有窗口测试期指标均值,
                stability_score: float,
                window_metrics: list[dict],
            }
        """
        profile_id: str = (
            candidate.get("profile_id")
            or candidate.get("id")
            or candidate.get("config", {}).get("id", "")
        )
        params = candidate.get("config", {}).get("params", {}) or candidate.get("params", {})
        top_n: int = int(params.get("top_n", 10))
        score_threshold = params.get("score_threshold")
        rebalance: str = params.get("rebalance", "W-FRI")

        # 若 candidate 包含 factors 定义（搜索生成的 inline 候选），
        # 将其注册为运行时临时 profile，确保打分链能按 profile_id 查到它。
        # 用 try/finally 保证无论是否异常都能清理，避免进程级状态泄漏。
        from backend.services.factor_profile_registry_service import FactorProfileRegistryService
        _registered_runtime_id: str | None = None
        if candidate.get("factors") and profile_id:
            inline_profile = {
                "id": profile_id,
                "mode": candidate.get("mode", "temporal_pool"),
                "description": candidate.get("description", ""),
                "factors": candidate["factors"],
                "params": params,
            }
            FactorProfileRegistryService.register_runtime_profile(inline_profile)
            _registered_runtime_id = profile_id

        hold_days: int = int(params.get("hold_days", 5))

        try:
            windows = self._generate_walk_forward_windows(
                start_date, end_date, train_months, valid_months, test_months, step_months
            )

            if not windows:
                raise ValueError(
                    f"时间范围 {start_date} ~ {end_date} 无法生成完整的 walk-forward 窗口，"
                    f"请确保时间跨度 ≥ {train_months + valid_months + test_months} 个月。"
                )

            window_metrics: list[dict] = []

            for w in windows:
                wid = w["window_id"]
                try:
                    train_metrics = self._tfvs.run_temporal_pool_backtest(
                        codes=codes,
                        profile_id=profile_id,
                        start_date=w["train_start"],
                        end_date=w["train_end"],
                        top_n=top_n,
                        score_threshold=score_threshold,
                        rebalance=rebalance,
                        hold_days=hold_days,
                        max_workers=max_workers,
                    )
                except Exception as exc:
                    logger.warning(f"窗口 {wid} 训练期回测失败，已跳过: {exc}")
                    train_metrics = _empty_metrics()

                try:
                    valid_metrics = self._tfvs.run_temporal_pool_backtest(
                        codes=codes,
                        profile_id=profile_id,
                        start_date=w["valid_start"],
                        end_date=w["valid_end"],
                        top_n=top_n,
                        score_threshold=score_threshold,
                        rebalance=rebalance,
                        hold_days=hold_days,
                        max_workers=max_workers,
                    )
                except Exception as exc:
                    logger.warning(f"窗口 {wid} 验证期回测失败，已跳过: {exc}")
                    valid_metrics = _empty_metrics()

                try:
                    test_metrics = self._tfvs.run_temporal_pool_backtest(
                        codes=codes,
                        profile_id=profile_id,
                        start_date=w["test_start"],
                        end_date=w["test_end"],
                        top_n=top_n,
                        score_threshold=score_threshold,
                        rebalance=rebalance,
                        hold_days=hold_days,
                        max_workers=max_workers,
                    )
                except Exception as exc:
                    logger.warning(f"窗口 {wid} 测试期回测失败，已跳过: {exc}")
                    test_metrics = _empty_metrics()

                # 过拟合检测
                train_sharpe = _safe_float(train_metrics.get("sharpe"))
                test_sharpe = _safe_float(test_metrics.get("sharpe"))
                overfitting_flag = (
                    not np.isnan(train_sharpe)
                    and not np.isnan(test_sharpe)
                    and test_sharpe < 0.3 * train_sharpe
                )

                window_metrics.append(
                    {
                        **w,
                        "train_metrics": train_metrics,
                        "valid_metrics": valid_metrics,
                        "test_metrics": test_metrics,
                        "overfitting_flag": overfitting_flag,
                    }
                )

            stability_result = self.compute_stability_score(window_metrics)

            agg_train = _aggregate_metrics([wm["train_metrics"] for wm in window_metrics])
            agg_valid = _aggregate_metrics([wm["valid_metrics"] for wm in window_metrics])
            agg_test = _aggregate_metrics([wm["test_metrics"] for wm in window_metrics])

            if isinstance(stability_result, dict):
                result = {
                    "train_metrics": agg_train,
                    "valid_metrics": agg_valid,
                    "test_metrics": agg_test,
                    "stability_score": stability_result["stability_score"],
                    "evaluation_status": stability_result.get("evaluation_status", "invalid"),
                    "invalid_reason": stability_result.get("invalid_reason"),
                    "window_metrics": window_metrics,
                }
            else:
                result = {
                    "train_metrics": agg_train,
                    "valid_metrics": agg_valid,
                    "test_metrics": agg_test,
                    "stability_score": stability_result,
                    "window_metrics": window_metrics,
                }

            # 写出 run_manifest.json
            from backend.services.champion_strategy_service import _write_run_manifest
            eval_dir = _EVAL_RESULTS_DIR / str(uuid.uuid4())
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_status = result.get("evaluation_status", "valid")
            quality_gate = [result["invalid_reason"]] if result.get("invalid_reason") else []
            _write_run_manifest(
                eval_dir,
                module="strategy_evaluation",
                data_coverage=1.0,
                quality_gate=quality_gate,
                result_status="completed" if eval_status == "valid" else eval_status,
            )

            return result

        finally:
            # 无论成功还是异常，都清理运行时临时 profile，防止进程级状态泄漏
            if _registered_runtime_id:
                FactorProfileRegistryService.unregister_runtime_profile(_registered_runtime_id)

    # ------------------------------------------------------------------
    # 截面候选评价
    # ------------------------------------------------------------------

    def evaluate_cross_sectional_candidate(
        self,
        candidate: dict,
        universe: str,
        start_date: str,
        end_date: str,
        train_months: int = 12,
        valid_months: int = 3,
        test_months: int = 3,
        step_months: int = 3,
    ) -> dict:
        """截面候选策略 walk-forward 评价。

        复用 vectorbt_backtest_service.cross_sectional_backtest_composite 执行回测。
        universe_df 优先从 candidate['universe_df'] 取；若未提供则通过 DataService
        按 universe 名称加载股票列表并拉取价格+因子数据。

        Args:
            candidate: 候选策略配置字典；可含 'universe_df'（pd.DataFrame）或
                       'universe_codes'（list[str]）以提供真实数据
            universe: universe 名称，用于从 DataService 加载数据（当 candidate 未提供数据时）
            start_date: 整体起始日期
            end_date: 整体结束日期
            train_months: 训练期月数
            valid_months: 验证期月数
            test_months: 测试期月数
            step_months: 滚动步长月数

        Returns:
            与 evaluate_temporal_pool_candidate 相同结构的字典。
        """
        from backend.services.vectorbt_backtest_service import VectorBTBacktestService

        vbt_svc = VectorBTBacktestService()

        config = candidate.get("config", candidate)
        factors = config.get("factors", [])
        factor_weights = {
            f["name"]: f["weight"]
            for f in factors
            if f.get("role", "signal") == "signal" and f.get("weight", 0) > 0
        }
        params = config.get("params", {})
        top_percentile: float = float(params.get("top_percentile", 0.2))

        windows = self._generate_walk_forward_windows(
            start_date, end_date, train_months, valid_months, test_months, step_months
        )

        if not windows:
            raise ValueError(
                f"时间范围 {start_date} ~ {end_date} 无法生成完整的 walk-forward 窗口。"
            )

        # 预加载全量 universe_df（整个回测区间），各窗口按日期切片
        universe_df: "pd.DataFrame | None" = candidate.get("universe_df")
        if universe_df is None:
            universe_df = _load_universe_df(
                universe=universe,
                codes=candidate.get("universe_codes"),
                start_date=start_date,
                end_date=end_date,
                factor_names=list(factor_weights.keys()),
            )

        # 覆盖率门禁：_load_universe_df 返回 invalid_run 字典时直接透传
        if isinstance(universe_df, dict) and universe_df.get("result_status") == "invalid_run":
            return universe_df

        window_metrics: list[dict] = []

        # Short-circuit: if universe_df is empty, no windows can run
        if universe_df is not None and universe_df.empty:
            return {
                "train_metrics": _empty_metrics(),
                "valid_metrics": _empty_metrics(),
                "test_metrics": _empty_metrics(),
                "stability_score": 0.0,
                "window_metrics": [],
            }

        for w in windows:
            wid = w["window_id"]
            train_metrics = _empty_metrics()
            valid_metrics = _empty_metrics()
            test_metrics = _empty_metrics()

            for period, period_start, period_end, store_key in [
                ("train", w["train_start"], w["train_end"], "train_metrics"),
                ("valid", w["valid_start"], w["valid_end"], "valid_metrics"),
                ("test", w["test_start"], w["test_end"], "test_metrics"),
            ]:
                try:
                    import pandas as pd
                    if universe_df is not None and not universe_df.empty:
                        date_col = "date" if "date" in universe_df.columns else universe_df.index.name
                        if date_col and date_col in universe_df.columns:
                            mask = (
                                (universe_df[date_col] >= pd.Timestamp(period_start))
                                & (universe_df[date_col] <= pd.Timestamp(period_end))
                            )
                            period_df = universe_df[mask].copy()
                        else:
                            period_df = universe_df.loc[
                                pd.Timestamp(period_start):pd.Timestamp(period_end)
                            ].copy()
                    else:
                        period_df = pd.DataFrame(columns=["date", "stock_code", "close"])

                    if period_df.empty:
                        logger.warning(
                            f"窗口 {wid} {period} 期 universe_df 为空 "
                            f"({period_start} ~ {period_end})，跳过"
                        )
                    else:
                        result = vbt_svc.cross_sectional_backtest_composite(
                            df=period_df,
                            factor_weights=factor_weights,
                            top_percentile=top_percentile,
                        )
                        metrics = {
                            "annual_return": result.get("annual_return", float("nan")),
                            "sharpe": result.get("sharpe", float("nan")),
                            "max_drawdown": result.get("max_drawdown", float("nan")),
                            "win_rate": float("nan"),
                            "volatility": result.get("volatility", float("nan")),
                            "total_return": result.get("total_return", float("nan")),
                        }
                        if store_key == "train_metrics":
                            train_metrics = metrics
                        elif store_key == "valid_metrics":
                            valid_metrics = metrics
                        else:
                            test_metrics = metrics
                except Exception as exc:
                    logger.warning(f"窗口 {wid} {period} 期截面回测失败，已跳过: {exc}")

            train_sharpe = _safe_float(train_metrics.get("sharpe"))
            test_sharpe = _safe_float(test_metrics.get("sharpe"))
            overfitting_flag = (
                not np.isnan(train_sharpe)
                and not np.isnan(test_sharpe)
                and test_sharpe < 0.3 * train_sharpe
            )

            window_metrics.append(
                {
                    **w,
                    "train_metrics": train_metrics,
                    "valid_metrics": valid_metrics,
                    "test_metrics": test_metrics,
                    "overfitting_flag": overfitting_flag,
                }
            )

        stability_result = self.compute_stability_score(window_metrics)

        agg_train = _aggregate_metrics([wm["train_metrics"] for wm in window_metrics])
        agg_valid = _aggregate_metrics([wm["valid_metrics"] for wm in window_metrics])
        agg_test = _aggregate_metrics([wm["test_metrics"] for wm in window_metrics])

        if isinstance(stability_result, dict):
            result = {
                "train_metrics": agg_train,
                "valid_metrics": agg_valid,
                "test_metrics": agg_test,
                "stability_score": stability_result["stability_score"],
                "evaluation_status": stability_result.get("evaluation_status", "invalid"),
                "invalid_reason": stability_result.get("invalid_reason"),
                "window_metrics": window_metrics,
            }
        else:
            result = {
                "train_metrics": agg_train,
                "valid_metrics": agg_valid,
                "test_metrics": agg_test,
                "stability_score": stability_result,
                "window_metrics": window_metrics,
            }

        # 写出 run_manifest.json
        from backend.services.champion_strategy_service import _write_run_manifest
        eval_dir = _EVAL_RESULTS_DIR / str(uuid.uuid4())
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_status = result.get("evaluation_status", "valid")
        quality_gate = [result["invalid_reason"]] if result.get("invalid_reason") else []
        _write_run_manifest(
            eval_dir,
            module="strategy_evaluation",
            data_coverage=1.0,
            quality_gate=quality_gate,
            result_status="completed" if eval_status == "valid" else eval_status,
        )

        return result


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _empty_metrics() -> dict:
    return {
        "annual_return": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "win_rate": float("nan"),
        "volatility": float("nan"),
        "total_return": float("nan"),
        "active_days": 0.0,
        "trading_days": 0.0,
        "active_ratio": 0.0,
        "rebalance_count": 0.0,
        "active_rebalance_count": 0.0,
    }


def _safe_float(val) -> float:
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    """对多个窗口的指标字典取均值（忽略 NaN）。"""
    if not metrics_list:
        return _empty_metrics()

    keys = list(metrics_list[0].keys())
    result: dict = {}
    for k in keys:
        vals = [_safe_float(m.get(k)) for m in metrics_list]
        valid = [v for v in vals if not np.isnan(v)]
        result[k] = float(np.mean(valid)) if valid else float("nan")
    return result


def _load_universe_df(
    universe: str,
    codes: list[str] | None,
    start_date: str,
    end_date: str,
    factor_names: list[str],
) -> "pd.DataFrame | dict":
    """
    从 DataService 加载截面回测所需的 DataFrame，并计算所需因子列。

    列：date, stock_code, close, [factor_names...]

    优先使用 codes 参数；若未提供则尝试从 universe 名称解析股票列表
    （当前仅支持 CSV 路径或已知 universe 标识，未知时返回空 DataFrame）。

    因子列通过 TemporalScoringService._compute_factors() 计算，
    支持 8 个内置因子；未知因子名会记录警告并跳过。

    当数据覆盖率低于 EVAL_MIN_CODE_COVERAGE 时，返回 invalid_run 字典。
    """
    import pandas as pd
    from backend.services.data_service import data_service
    from backend.core.settings import settings as _settings

    if not codes:
        # 尝试把 universe 当作 CSV 路径
        from pathlib import Path
        p = Path(universe)
        if p.exists() and p.suffix == ".csv":
            try:
                df = pd.read_csv(p)
                if "code" in df.columns:
                    codes = df["code"].astype(str).str.strip().tolist()
            except Exception as exc:
                logger.warning(f"_load_universe_df: 无法从 {universe} 读取 codes: {exc}")

    if not codes:
        logger.warning(
            f"_load_universe_df: universe='{universe}' 无法解析股票列表，"
            "截面回测将返回空结果。请通过 candidate['universe_codes'] 或 "
            "candidate['universe_df'] 提供数据。"
        )
        return pd.DataFrame(columns=["date", "stock_code", "close"])

    # 拉取数据时往前多取 lookback，确保因子计算有足够历史
    from datetime import datetime, timedelta
    lookback_days = 372  # 默认 lookback，temporal_scoring_service 导入延迟到门禁通过后
    fetch_start = (
        datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    report = data_service.get_multiple_stocks_data_with_report(codes, fetch_start, end_date)

    # 覆盖率门禁
    reason_codes = []
    if report.code_coverage < _settings.EVAL_MIN_CODE_COVERAGE:
        reason_codes.append("CODE_COVERAGE_TOO_LOW")

    # 有效日期数门禁
    if report.data:
        all_dates = set()
        for df in report.data.values():
            if df is not None and not df.empty:
                all_dates.update(df.index.tolist())
        valid_date_count = len(all_dates)
    else:
        valid_date_count = 0

    if valid_date_count < _settings.EVAL_MIN_IC_SAMPLES:
        reason_codes.append("INSUFFICIENT_IC_SAMPLES")

    if reason_codes:
        logger.warning(
            f"_load_universe_df: 覆盖率门禁触发 {reason_codes}，"
            f"code_coverage={report.code_coverage:.2%}, valid_dates={valid_date_count}，返回 invalid_run"
        )
        return {
            "result_status": "invalid_run",
            "reason_codes": reason_codes,
            "code_coverage": report.code_coverage,
        }

    # 延迟导入，避免循环依赖（仅在门禁通过后才需要）
    from backend.services.temporal_scoring_service import temporal_scoring_service as _tss
    lookback_days = getattr(_tss, "lookback_days", lookback_days)

    price_data = report.data

    frames = []
    for code, ohlcv_df in price_data.items():
        if ohlcv_df is None or ohlcv_df.empty:
            continue

        # 计算所有 8 个内置因子
        try:
            all_factors = _tss._compute_factors(ohlcv_df)
        except Exception as exc:
            logger.warning(f"_load_universe_df: 股票 {code} 因子计算失败，跳过: {exc}")
            continue

        row = ohlcv_df[["close"]].copy()
        row["stock_code"] = code
        row.index.name = "date"
        row = row.reset_index()

        # 附加请求的因子列
        for fn in factor_names:
            if fn in all_factors:
                factor_series = all_factors[fn]
                # 对齐索引后赋值
                row[fn] = factor_series.reindex(
                    pd.to_datetime(row["date"])
                ).values
            else:
                logger.warning(
                    f"_load_universe_df: 因子 '{fn}' 不在内置因子列表中，"
                    f"股票 {code} 该列将为 NaN"
                )
                row[fn] = float("nan")

        frames.append(row)

    if not frames:
        return pd.DataFrame(columns=["date", "stock_code", "close"])

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])

    # 裁剪到请求的日期范围（去掉 lookback 预热段）
    result = result[
        (result["date"] >= pd.Timestamp(start_date))
        & (result["date"] <= pd.Timestamp(end_date))
    ].reset_index(drop=True)

    return result


def _build_universe_df(universe: str, start_date: str, end_date: str):
    """已废弃占位符，保留以兼容旧调用。"""
    import pandas as pd
    logger.warning(
        f"_build_universe_df: universe='{universe}' 已废弃，请改用 _load_universe_df。"
    )
    return pd.DataFrame(columns=["date", "stock_code", "close"])
