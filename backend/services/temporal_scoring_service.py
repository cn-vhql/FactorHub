"""
时序打分服务模块 - 日频时序百分位评分
"""
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from backend.core.settings import settings
from backend.services.data_service import data_service
from backend.services.factor_service import factor_service

logger = logging.getLogger(__name__)

# 需要从平台预置因子表中查找表达式的 6 个因子名
# price_vwma_ratio / force_index_ma 不在预置表中，在 _compute_factors 里直接用 pandas 实现
_PRESET_FACTOR_NAMES = [
    "price_vs_sma20",
    "momentum_20",
    "bollinger_position",
    "stochastic_d",
    "atr_norm",
    "volume_ma_ratio",
]

# 默认回望天数
_DEFAULT_LOOKBACK_DAYS = 372


def _load_preset_expressions() -> dict[str, str]:
    """
    从 factor_service 预置因子表中查找 _PRESET_FACTOR_NAMES 对应的表达式。
    找不到的因子使用内置兜底表达式，保证服务可启动。
    """
    from backend.core.database import get_db_session
    from backend.repositories.factor_repository import FactorRepository

    _fallback = {
        "price_vs_sma20": "close / SMA(close, timeperiod=20)",
        "momentum_20": "close / close.shift(20) - 1",
        "bollinger_position": (
            "(close - BBANDS(close, timeperiod=20)[2]) / "
            "(BBANDS(close, timeperiod=20)[0] - BBANDS(close, timeperiod=20)[2])"
        ),
        "stochastic_d": (
            "SMA((close - LLV(low, 14)) / (HHV(high, 14) - LLV(low, 14)) * 100, timeperiod=3)"
        ),
        "atr_norm": "ATR(high, low, close, timeperiod=14) / close",
        "volume_ma_ratio": "volume / SMA(volume, timeperiod=10)",
    }
    try:
        db = get_db_session()
        repo = FactorRepository(db)
        result = {}
        for name in _PRESET_FACTOR_NAMES:
            factor = repo.get_by_name(name)
            result[name] = factor.code if factor else _fallback[name]
        db.close()
        return result
    except Exception as exc:
        logger.warning(f"无法从预置因子表加载表达式，使用内置兜底: {exc}")
        return _fallback


class TemporalScoringService:
    """单只股票的因子计算、百分位归一化与评分合成"""

    def __init__(self):
        self.calculator = factor_service.calculator
        self.lookback_days: int = getattr(settings, "SCORING_LOOKBACK_DAYS", _DEFAULT_LOOKBACK_DAYS)
        self.last_refresh_at: Optional[datetime] = None
        self.factor_expression_source: str = "fallback"
        # 从平台预置因子表加载表达式，保持与主链口径一致
        try:
            self._factor_expressions: dict[str, str] = _load_preset_expressions()
            self.factor_expression_source = "db"
        except Exception as exc:
            logger.warning(f"初始化时无法从 DB 加载因子表达式，使用 fallback: {exc}")
            self._factor_expressions = _load_preset_expressions()
            self.factor_expression_source = "fallback"
        self.last_refresh_at = datetime.now()
        # 评分缓存目录
        self.score_cache_dir = Path("data/cache/temporal_scores")
        self.score_cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 因子表达式刷新
    # ------------------------------------------------------------------

    def refresh_factor_expressions(self, force: bool = False) -> None:
        """
        重新从 DB 加载因子表达式。

        Args:
            force: True 时强制刷新；False 时仅在距上次加载超过 1 小时时刷新
        """
        if not force:
            if self.last_refresh_at is not None:
                elapsed = (datetime.now() - self.last_refresh_at).total_seconds()
                if elapsed < 3600:
                    return

        try:
            new_expressions = _load_preset_expressions()
            self._factor_expressions = new_expressions
            self.factor_expression_source = "db"
            self.last_refresh_at = datetime.now()
            logger.info("因子表达式已从 DB 刷新成功")
        except Exception as exc:
            self.factor_expression_source = "fallback"
            logger.warning(f"刷新因子表达式失败，继续使用 fallback 表达式: {exc}")

    # ------------------------------------------------------------------
    # 数据获取
    # ------------------------------------------------------------------

    def _get_history_data(self, code: str, trade_date: str) -> pd.DataFrame:
        """
        获取截至 trade_date 的 lookback_days 历史数据。

        Args:
            code: 股票代码，如 "600519"
            trade_date: 交易日期，格式 "YYYY-MM-DD"

        Returns:
            包含 OHLCV 数据的 DataFrame
        """
        trade_date_dt = datetime.strptime(trade_date, "%Y-%m-%d")
        start_date = (trade_date_dt - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        return data_service.get_stock_data(code, start_date, trade_date)

    # ------------------------------------------------------------------
    # 因子计算
    # ------------------------------------------------------------------

    def _compute_factors(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """
        计算 8 个固定因子，返回各因子的完整时间序列。
        6 个因子表达式从平台预置因子表加载，保持与主链口径一致。
        price_vwma_ratio / force_index_ma 不在预置表中，直接用 pandas 实现。
        """
        factors: dict[str, pd.Series] = {}

        # 从预置因子表加载表达式的 6 个因子
        for name, expr in self._factor_expressions.items():
            factors[name] = self.calculator.calculate(df, expr)

        # price_vwma_ratio：pandas 直接实现（预置表中无此因子）
        vwma = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
        factors["price_vwma_ratio"] = df["close"] / vwma

        # force_index_ma：pandas 直接实现（预置表中无此因子）
        force_index = df["close"].diff(1) * df["volume"]
        factors["force_index_ma"] = force_index.rolling(13).mean()

        return factors

    # ------------------------------------------------------------------
    # 百分位归一化
    # ------------------------------------------------------------------

    def _compute_percentile(self, series: pd.Series, current_value: float) -> float:
        """
        使用 scipy.stats.percentileofscore(kind='rank') 计算百分位。

        Args:
            series: 历史因子时间序列（含当前值）
            current_value: 当日因子值

        Returns:
            百分位，归一化到 [0, 1]

        Raises:
            ValueError: 有效数据点 < 20
        """
        valid_history = series.dropna()
        if len(valid_history) < 20:
            raise ValueError(
                f"历史数据不足：有效数据点 {len(valid_history)} < 20"
            )
        pct_raw = stats.percentileofscore(valid_history, current_value, kind="rank")
        return float(np.clip(pct_raw / 100.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 评分合成
    # ------------------------------------------------------------------

    def _compute_score(self, pcts: dict[str, float]) -> float:
        """
        根据百分位字典计算最终 score（含惩罚项，clip 到 [0, 100]）。

        Args:
            pcts: 因子名 -> 百分位（[0, 1]）的字典

        Returns:
            综合评分，范围 [0, 100]
        """
        raw_score = 100 * (
            0.22 * pcts["price_vs_sma20"]
            + 0.18 * pcts["momentum_20"]
            + 0.16 * pcts["price_vwma_ratio"]
            + 0.14 * pcts["force_index_ma"]
            + 0.14 * pcts["bollinger_position"]
            + 0.16 * pcts["stochastic_d"]
        )

        trend_break = 1 if (pcts["price_vs_sma20"] < 0.35 or pcts["stochastic_d"] < 0.20) else 0
        high_volatility = 1 if pcts["atr_norm"] > 0.80 else 0
        liquidity_risk = 1 if pcts["volume_ma_ratio"] < 0.30 else 0

        score = raw_score - 12 * trend_break - 10 * high_volatility - 8 * liquidity_risk
        return float(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Profile 驱动：因子计算
    # ------------------------------------------------------------------

    def _compute_factors_from_profile(self, df: pd.DataFrame, profile: dict) -> dict:
        """
        按 profile 的 factors 列表动态计算因子值。

        Args:
            df: 包含 OHLCV 数据的 DataFrame
            profile: profile 配置字典，含 factors 列表

        Returns:
            {factor_name: pd.Series} 字典，每个值为完整时间序列
        """
        # 先计算所有 8 个内置因子（复用现有逻辑）
        all_factors = self._compute_factors(df)

        # 按 profile 中的因子名过滤，只返回 profile 需要的因子
        result: dict[str, pd.Series] = {}
        for factor_cfg in profile.get("factors", []):
            name = factor_cfg["name"]
            if name in all_factors:
                result[name] = all_factors[name]
            else:
                # 尝试通过 factor_service 计算未知因子
                try:
                    from backend.services.factor_service import factor_service as _fs
                    # 尝试从预置因子表获取表达式
                    from backend.core.database import get_db_session
                    from backend.repositories.factor_repository import FactorRepository
                    db = get_db_session()
                    repo = FactorRepository(db)
                    factor_obj = repo.get_by_name(name)
                    db.close()
                    if factor_obj:
                        result[name] = _fs.calculator.calculate(df, factor_obj.code)
                    else:
                        logger.warning(f"因子 {name} 在内置因子和预置因子表中均未找到，跳过")
                except Exception as exc:
                    logger.warning(f"计算因子 {name} 失败: {exc}，跳过")

        return result

    # ------------------------------------------------------------------
    # Profile 驱动：评分合成
    # ------------------------------------------------------------------

    def _compute_score_from_profile(self, pcts: dict, profile: dict) -> float:
        """
        按 profile 的 weight/direction/threshold 合成分数。

        规则：
        - signal 因子：score += weight * direction * pct * 100
        - risk_filter 因子：如果 pct > threshold，则 score = 0（过滤掉）

        Args:
            pcts: {factor_name: percentile_value} 字典，每个值 ∈ [0, 1]
            profile: profile 配置字典

        Returns:
            综合评分，范围 [0, 100]
        """
        score = 0.0

        for factor_cfg in profile.get("factors", []):
            name = factor_cfg["name"]
            role = factor_cfg.get("role", "signal")
            pct = pcts.get(name, 0.0)

            if role == "risk_filter":
                threshold = factor_cfg.get("threshold", 0.8)
                direction = factor_cfg.get("direction", 1)
                # direction=-1 表示"越高越危险"（如 atr_norm），pct > threshold 时过滤
                # direction=1 表示"越低越危险"（如 volume_ma_ratio），pct < threshold 时过滤
                if direction == -1 and pct > threshold:
                    return 0.0
                elif direction == 1 and pct < threshold:
                    return 0.0
            else:
                # signal 因子
                weight = factor_cfg.get("weight", 0.0)
                direction = factor_cfg.get("direction", 1)
                score += weight * direction * pct * 100

        return float(np.clip(score, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Profile 驱动：单只股票评分
    # ------------------------------------------------------------------

    def score_one_stock_with_profile(
        self, code: str, trade_date: str, profile_id: str
    ) -> dict:
        """
        按 profile 配置对单只股票打分。

        Args:
            code: 股票代码，如 "600519"
            trade_date: 交易日期，格式 "YYYY-MM-DD"
            profile_id: profile 唯一标识

        Returns:
            包含 date、code、score、factors、pcts 的字典
            score ∈ [0, 100]，pcts 中每个值 ∈ [0, 1]

        Raises:
            KeyError: profile_id 不存在
            ValueError: 数据不足或因子计算失败
        """
        from backend.services.factor_profile_registry_service import FactorProfileRegistryService
        registry = FactorProfileRegistryService()
        profile = registry.get_profile(profile_id)

        percentile_window = profile.get("params", {}).get("percentile_window", 252)

        df = self._get_history_data(code, trade_date)
        factor_series = self._compute_factors_from_profile(df, profile)

        # 解析实际交易日
        trade_date_ts = pd.Timestamp(trade_date)
        available = df.index[df.index <= trade_date_ts]
        if available.empty:
            raise ValueError(f"股票 {code} 在 {trade_date} 之前无可用数据")
        actual_date_ts = available[-1]
        actual_date = actual_date_ts.strftime("%Y-%m-%d")

        # 取实际交易日的因子值
        factors_current: dict[str, float] = {}
        for name, series in factor_series.items():
            if actual_date_ts in series.index:
                val = series.loc[actual_date_ts]
            else:
                val = series.dropna().iloc[-1] if not series.dropna().empty else float("nan")
            factors_current[name] = float(val)

        # 计算百分位
        pcts: dict[str, float] = {}
        for name, series in factor_series.items():
            current_val = factors_current[name]
            if np.isnan(current_val):
                raise ValueError(f"股票 {code} 因子 {name} 当日值为 NaN")
            # 使用 profile 指定的 percentile_window 截取历史窗口
            window_series = series.dropna().iloc[-percentile_window:]
            pcts[name] = self._compute_percentile(window_series, current_val)

        score = self._compute_score_from_profile(pcts, profile)

        return {
            "date": actual_date,
            "code": code,
            "score": round(score, 1),
            "factors": factors_current,
            "pcts": pcts,
        }

    # ------------------------------------------------------------------
    # Profile 驱动：历史批量评分
    # ------------------------------------------------------------------

    def score_stock_history_with_profile(
        self,
        code: str,
        start_date: str,
        end_date: str,
        profile_id: str,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        按 profile 配置计算单只股票历史评分面板。

        Args:
            code: 股票代码
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            profile_id: profile 唯一标识
            window: 滚动窗口大小（交易日数），默认 252

        Returns:
            包含 date, code, score 列的 DataFrame

        Raises:
            KeyError: profile_id 不存在
            ValueError: 有效交易日数 < window + 20
        """
        # 缓存 key 包含 profile_id，避免不同 profile 缓存冲突
        cache_path = self.score_cache_dir / f"{code}_{start_date}_{end_date}_{profile_id}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        from backend.services.factor_profile_registry_service import FactorProfileRegistryService
        registry = FactorProfileRegistryService()
        profile = registry.get_profile(profile_id)

        # 优先使用 profile 自身配置的 percentile_window，保证与单日打分口径一致
        window = profile.get("params", {}).get("percentile_window", window)

        # 一次 IO 拉取足够的历史数据
        # window 是交易日数，换算为自然日需乘以约 1.5（每年约 252 交易日 / 365 自然日）
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        fetch_days = int(window * 1.5) + 60  # 额外 60 天缓冲
        fetch_start = (start_dt - timedelta(days=fetch_days)).strftime("%Y-%m-%d")
        df = data_service.get_stock_data(code, fetch_start, end_date)

        if len(df) < window + 20:
            raise ValueError(
                f"股票 {code} 有效交易日数 {len(df)} < window + 20 ({window + 20})"
            )

        # 计算 profile 所需因子时间序列
        factor_series = self._compute_factors_from_profile(df, profile)
        if not factor_series:
            raise ValueError(f"Profile {profile_id} 未能计算任何因子")

        factor_df = pd.DataFrame(factor_series)
        rolling_rank = factor_df.rolling(window=window, min_periods=20).rank(pct=True)

        # 按 profile 合成分数（向量化）
        score_series = pd.Series(0.0, index=df.index)
        filtered_mask = pd.Series(False, index=df.index)

        for factor_cfg in profile.get("factors", []):
            name = factor_cfg["name"]
            role = factor_cfg.get("role", "signal")
            if name not in rolling_rank.columns:
                continue

            pct_col = rolling_rank[name]

            if role == "risk_filter":
                threshold = factor_cfg.get("threshold", 0.8)
                direction = factor_cfg.get("direction", 1)
                if direction == -1:
                    filtered_mask = filtered_mask | (pct_col > threshold)
                else:
                    filtered_mask = filtered_mask | (pct_col < threshold)
            else:
                weight = factor_cfg.get("weight", 0.0)
                direction = factor_cfg.get("direction", 1)
                score_series = score_series + weight * direction * pct_col * 100

        # 应用风险过滤
        score_series = score_series.where(~filtered_mask, other=0.0)
        score_series = score_series.clip(0, 100)

        result = pd.DataFrame({
            "date": df.index,
            "code": code,
            "score": score_series.values,
        })

        # 只保留 [start_date, end_date] 区间，过滤 NaN
        result = result[
            (result["date"] >= pd.Timestamp(start_date))
            & (result["date"] <= pd.Timestamp(end_date))
        ].dropna(subset=["score"]).reset_index(drop=True)

        # 缓存结果
        result.to_parquet(cache_path, index=False)

        return result

    # ------------------------------------------------------------------
    # 单只股票评分
    # ------------------------------------------------------------------

    def score_one_stock(self, code: str, trade_date: str) -> dict:
        """
        计算单只股票在指定交易日的评分。

        Args:
            code: 股票代码，如 "600519"
            trade_date: 交易日期，格式 "YYYY-MM-DD"

        Returns:
            包含 date、code、score、factors、pcts 的字典

        Raises:
            ValueError: 数据不足或因子计算失败
        """
        df = self._get_history_data(code, trade_date)
        factor_series = self._compute_factors(df)

        # 解析实际交易日：取数据中不晚于 trade_date 的最后一个索引
        # 非交易日（周末/节假日）时 trade_date 不在序列里，回退到最近交易日
        trade_date_ts = pd.Timestamp(trade_date)
        available = df.index[df.index <= trade_date_ts]
        if available.empty:
            raise ValueError(f"股票 {code} 在 {trade_date} 之前无可用数据")
        actual_date_ts = available[-1]
        actual_date = actual_date_ts.strftime("%Y-%m-%d")

        # 取实际交易日的因子值
        factors_current: dict[str, float] = {}
        for name, series in factor_series.items():
            if actual_date_ts in series.index:
                val = series.loc[actual_date_ts]
            else:
                val = series.dropna().iloc[-1] if not series.dropna().empty else float("nan")
            factors_current[name] = float(val)

        # 计算百分位
        pcts: dict[str, float] = {}
        for name, series in factor_series.items():
            current_val = factors_current[name]
            if np.isnan(current_val):
                raise ValueError(f"股票 {code} 因子 {name} 当日值为 NaN")
            pcts[name] = self._compute_percentile(series, current_val)

        score = self._compute_score(pcts)

        return {
            "date": actual_date,   # 实际使用的交易日，非传入的 trade_date
            "code": code,
            "score": round(score, 1),
            "factors": factors_current,
            "pcts": pcts,
        }

    # ------------------------------------------------------------------
    # 批量评分（同步）
    # ------------------------------------------------------------------

    def score_many_stocks(
        self, codes: list[str], trade_date: str
    ) -> tuple[list[dict], list[dict]]:
        """
        同步批量评分（供同步接口调用）。

        Args:
            codes: 股票代码列表
            trade_date: 交易日期，格式 "YYYY-MM-DD"

        Returns:
            (results, errors)
            results: 评分结果列表
            errors:  失败记录列表 [{"code": ..., "error": ...}]
        """
        results: list[dict] = []
        errors: list[dict] = []

        for code in codes:
            try:
                result = self.score_one_stock(code, trade_date)
                results.append(result)
            except Exception as exc:
                logger.warning(f"股票 {code} 评分失败: {exc}")
                errors.append({"code": code, "error": str(exc)})

        return results, errors

    # ------------------------------------------------------------------
    # 历史批量评分（向量化）
    # ------------------------------------------------------------------

    def score_stock_history(
        self,
        code: str,
        start_date: str,
        end_date: str,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        计算单只股票在 [start_date, end_date] 区间内每个交易日的评分。

        Args:
            code: 股票代码
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            window: 滚动窗口大小（交易日数），默认 252

        Returns:
            包含 date, code, score, trend_break, high_volatility, liquidity_risk 列的 DataFrame

        Raises:
            ValueError: 有效交易日数 < window + 20
        """
        # 检查缓存
        cache_path = self.score_cache_dir / f"{code}_{start_date}_{end_date}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        # 一次 IO 拉取足够的历史数据（window 是交易日数，换算为自然日需乘以约 1.5）
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        fetch_days = int(window * 1.5) + 60
        fetch_start = (start_dt - timedelta(days=fetch_days)).strftime("%Y-%m-%d")
        df = data_service.get_stock_data(code, fetch_start, end_date)

        if len(df) < window + 20:
            raise ValueError(
                f"股票 {code} 有效交易日数 {len(df)} < window + 20 ({window + 20})"
            )

        # 计算 8 个因子时间序列
        factor_series = self._compute_factors(df)

        # 向量化滚动百分位（rolling rank）
        factor_df = pd.DataFrame(factor_series)
        rolling_rank = factor_df.rolling(window=window, min_periods=20).rank(pct=True)

        pct_price_vs_sma20 = rolling_rank["price_vs_sma20"]
        pct_momentum_20 = rolling_rank["momentum_20"]
        pct_price_vwma_ratio = rolling_rank["price_vwma_ratio"]
        pct_force_index_ma = rolling_rank["force_index_ma"]
        pct_bollinger_position = rolling_rank["bollinger_position"]
        pct_stochastic_d = rolling_rank["stochastic_d"]
        pct_atr_norm = rolling_rank["atr_norm"]
        pct_volume_ma_ratio = rolling_rank["volume_ma_ratio"]

        raw_score = 100 * (
            0.22 * pct_price_vs_sma20
            + 0.18 * pct_momentum_20
            + 0.16 * pct_price_vwma_ratio
            + 0.14 * pct_force_index_ma
            + 0.14 * pct_bollinger_position
            + 0.16 * pct_stochastic_d
        )

        trend_break = ((pct_price_vs_sma20 < 0.35) | (pct_stochastic_d < 0.20)).astype(int)
        high_volatility = (pct_atr_norm > 0.80).astype(int)
        liquidity_risk = (pct_volume_ma_ratio < 0.30).astype(int)

        score = (raw_score - 12 * trend_break - 10 * high_volatility - 8 * liquidity_risk).clip(0, 100)

        result = pd.DataFrame({
            "date": df.index,
            "code": code,
            "score": score.values,
            "trend_break": trend_break.values,
            "high_volatility": high_volatility.values,
            "liquidity_risk": liquidity_risk.values,
        })

        # 只保留 [start_date, end_date] 区间，并过滤掉 score 为 NaN 的行
        result = result[
            (result["date"] >= pd.Timestamp(start_date))
            & (result["date"] <= pd.Timestamp(end_date))
        ].dropna(subset=["score"]).reset_index(drop=True)

        # 缓存结果
        result.to_parquet(cache_path, index=False)

        return result

    def score_many_stocks_history(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        max_workers: int = 4,
    ) -> pd.DataFrame:
        """
        并发计算多只股票在 [start_date, end_date] 区间内的历史评分。

        Args:
            codes: 股票代码列表
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            max_workers: 线程池大小，默认 4

        Returns:
            所有成功股票的评分 DataFrame（pd.concat 合并）
        """
        self._last_history_errors: list[dict] = []
        frames: list[pd.DataFrame] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_code = {
                executor.submit(self.score_stock_history, code, start_date, end_date): code
                for code in codes
            }
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    frames.append(future.result())
                except Exception as exc:
                    logger.warning(f"股票 {code} 历史评分失败: {exc}")
                    self._last_history_errors.append({"code": code, "error": str(exc)})

        if not frames:
            return pd.DataFrame(columns=["date", "code", "score", "trend_break", "high_volatility", "liquidity_risk"])

        return pd.concat(frames, ignore_index=True)


# 全局单例
temporal_scoring_service = TemporalScoringService()


class TaskManager:
    """异步任务的生命周期管理、并发控制、文件落盘"""

    def __init__(self):
        self.scoring_service = temporal_scoring_service  # 复用全局单例
        self.results_dir: Path = getattr(
            settings, "SCORING_RESULTS_DIR", settings.REPORTS_DIR / "scoring_jobs"
        )
        self.max_workers: int = getattr(settings, "SCORING_MAX_WORKERS", 4)
        self._file_locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------
    # 目录辅助
    # ------------------------------------------------------------------

    def _task_dir(self, task_id: str) -> Path:
        return self.results_dir / task_id

    # ------------------------------------------------------------------
    # 任务创建
    # ------------------------------------------------------------------

    def create_task(
        self,
        codes: list[str],
        trade_date: str,
        profile_id: str | None = None,
        per_stock_profiles: dict[str, str] | None = None,
    ) -> str:
        """
        创建任务目录结构，写入 input_codes.csv 和初始 status.json。
        返回 task_id（UUID）。
        profile_id / per_stock_profiles 若提供则写入 status.json，供 run_task 使用。
        """
        task_id = str(uuid.uuid4())
        task_dir = self._task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)

        # 写入 input_codes.csv（每行一个代码，无表头）
        (task_dir / "input_codes.csv").write_text(
            "\n".join(codes) + "\n", encoding="utf-8"
        )

        # 写入初始 status.json
        now = datetime.now(timezone.utc).isoformat()
        status = {
            "task_id": task_id,
            "status": "pending",
            "trade_date": trade_date,
            "profile_id": profile_id,
            "per_stock_profiles": per_stock_profiles or {},
            "total": len(codes),
            "processed": 0,
            "success_count": 0,
            "failed_count": 0,
            "progress": 0.0,
            "started_at": None,
            "updated_at": now,
            "completed_at": None,
        }
        self._write_status(task_id, status)
        self._file_locks[task_id] = threading.Lock()
        return task_id

    # ------------------------------------------------------------------
    # 文件落盘
    # ------------------------------------------------------------------

    def _write_status(self, task_id: str, status_dict: dict) -> None:
        """原子写入 status.json（写临时文件后 rename）"""
        task_dir = self._task_dir(task_id)
        status_path = task_dir / "status.json"
        tmp_path = task_dir / "status.json.tmp"
        tmp_path.write_text(
            json.dumps(status_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp_path.rename(status_path)

    def _append_score_row(self, task_id: str, result: dict) -> None:
        """追加写入 daily_scores.csv 和 daily_scores_detail.csv"""
        task_dir = self._task_dir(task_id)
        lock = self._file_locks.get(task_id, threading.Lock())

        scores_path = task_dir / "daily_scores.csv"
        detail_path = task_dir / "daily_scores_detail.csv"

        FACTOR_NAMES = [
            "price_vs_sma20", "momentum_20", "price_vwma_ratio", "force_index_ma",
            "bollinger_position", "stochastic_d", "atr_norm", "volume_ma_ratio",
        ]

        with lock:
            # daily_scores.csv
            write_header = not scores_path.exists()
            with open(scores_path, "a", encoding="utf-8", newline="") as f:
                if write_header:
                    f.write("date,code,score\n")
                f.write(f"{result['date']},\"{result['code']}\",{result['score']:.1f}\n")

            # daily_scores_detail.csv
            write_header_detail = not detail_path.exists()
            factor_cols = ",".join(FACTOR_NAMES)
            pct_cols = ",".join(f"pct_{n}" for n in FACTOR_NAMES)
            with open(detail_path, "a", encoding="utf-8", newline="") as f:
                if write_header_detail:
                    f.write(f"date,code,score,{factor_cols},{pct_cols}\n")
                factor_vals = ",".join(
                    str(result["factors"].get(n, "")) for n in FACTOR_NAMES
                )
                pct_vals = ",".join(
                    str(result["pcts"].get(n, "")) for n in FACTOR_NAMES
                )
                f.write(
                    f"{result['date']},\"{result['code']}\",{result['score']:.1f},"
                    f"{factor_vals},{pct_vals}\n"
                )

    def _append_error_row(self, task_id: str, code: str, error: str) -> None:
        """追加写入 errors.csv"""
        task_dir = self._task_dir(task_id)
        lock = self._file_locks.get(task_id, threading.Lock())
        errors_path = task_dir / "errors.csv"
        with lock:
            write_header = not errors_path.exists()
            with open(errors_path, "a", encoding="utf-8", newline="") as f:
                if write_header:
                    f.write("code,error\n")
                safe_error = error.replace('"', '""')
                f.write(f'{code},"{safe_error}"\n')

    # ------------------------------------------------------------------
    # 后台任务执行
    # ------------------------------------------------------------------

    def run_task(
        self,
        task_id: str,
        codes: list[str],
        trade_date: str,
        profile_id: str | None = None,
        per_stock_profiles: dict[str, str] | None = None,
    ) -> None:
        """后台执行任务（由 FastAPI BackgroundTasks 调用）。

        profile_id / per_stock_profiles 优先从参数取，其次从 status.json 读取，
        确保异步路径和自动升级路径都能正确带上 per_stock_profiles 语义。
        """
        try:
            # 更新状态为 running
            status = self.get_status(task_id)
            # 若调用方未传，从已持久化的 status 中读取
            effective_profile_id = profile_id or status.get("profile_id")
            effective_per_stock = per_stock_profiles or status.get("per_stock_profiles") or {}
            status["status"] = "running"
            status["started_at"] = datetime.now(timezone.utc).isoformat()
            status["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_status(task_id, status)

            def _score_one(code: str) -> dict:
                # 优先级：per_stock_profiles[code] > profile_id > 默认评分
                pid = effective_per_stock.get(code) or effective_profile_id
                if pid:
                    return self.scoring_service.score_one_stock_with_profile(
                        code, trade_date, pid
                    )
                return self.scoring_service.score_one_stock(code, trade_date)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_code = {
                    executor.submit(_score_one, code): code
                    for code in codes
                }

                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    status = self.get_status(task_id)
                    try:
                        result = future.result()
                        self._append_score_row(task_id, result)
                        status["success_count"] += 1
                    except Exception as exc:
                        self._append_error_row(task_id, code, str(exc))
                        status["failed_count"] += 1

                    status["processed"] += 1
                    status["progress"] = round(
                        status["processed"] / status["total"] * 100, 1
                    )
                    status["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._write_status(task_id, status)

            # 完成
            status = self.get_status(task_id)
            status["status"] = "completed"
            status["completed_at"] = datetime.now(timezone.utc).isoformat()
            status["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_status(task_id, status)

        except Exception as exc:
            logger.error(f"任务 {task_id} 执行失败: {exc}", exc_info=True)
            try:
                status = self.get_status(task_id)
                status["status"] = "failed"
                status["updated_at"] = datetime.now(timezone.utc).isoformat()
                status["error"] = str(exc)
                self._write_status(task_id, status)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 状态与结果查询
    # ------------------------------------------------------------------

    def get_status(self, task_id: str) -> dict:
        """读取并返回 status.json 内容。task_id 不存在时抛出 FileNotFoundError。"""
        task_dir = self._task_dir(task_id)
        status_path = task_dir / "status.json"
        if not status_path.exists():
            raise FileNotFoundError(f"任务 {task_id} 不存在")
        return json.loads(status_path.read_text(encoding="utf-8"))

    def get_results(self, task_id: str, detail: bool = False) -> list[dict]:
        """
        读取 daily_scores.csv（detail=False）或 daily_scores_detail.csv（detail=True）。
        任务未完成时抛出 ValueError。
        """
        status = self.get_status(task_id)
        if status["status"] != "completed":
            raise ValueError("任务尚未完成")
        task_dir = self._task_dir(task_id)
        csv_path = task_dir / (
            "daily_scores_detail.csv" if detail else "daily_scores.csv"
        )
        if not csv_path.exists():
            return []
        df = pd.read_csv(csv_path, dtype={"code": str})
        return df.to_dict(orient="records")


# TaskManager 全局单例
task_manager = TaskManager()
