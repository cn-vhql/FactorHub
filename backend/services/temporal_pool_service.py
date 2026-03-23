"""
TemporalPoolService
对股票池循环调用单股打分，汇总为池内排序结果，实现 100→10 场景。
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from backend.services.temporal_scoring_service import TemporalScoringService

logger = logging.getLogger(__name__)


def _normalize_code(code: str) -> str:
    """规范化股票代码，确保深市代码保留前导零（6 位）。"""
    code = str(code).strip()
    # 如果是纯数字且不足 6 位，补前导零
    if code.isdigit() and len(code) < 6:
        code = code.zfill(6)
    return code


class TemporalPoolService:
    """对股票池逐只打分并汇总排序的服务。"""

    def __init__(self) -> None:
        self._scoring_service = TemporalScoringService()

    # ------------------------------------------------------------------
    # 单日池内打分
    # ------------------------------------------------------------------

    def score_pool(
        self,
        codes: list[str],
        profile_id: str,
        trade_date: str,
        max_workers: int = 1,
    ) -> pd.DataFrame:
        """对股票池逐只打分，返回含 code/score/rank 的 DataFrame（按 score 降序）。

        Args:
            codes: 股票代码列表
            profile_id: profile 唯一标识
            trade_date: 交易日期，格式 "YYYY-MM-DD"
            max_workers: 并发线程数，=1 时顺序执行，>1 时使用 ThreadPoolExecutor

        Returns:
            包含 code、score、rank 列的 DataFrame，按 score 降序排列。
            rank 为 1-based 降序（score 最高者 rank=1）。
            打分失败的股票不出现在结果中。
        """
        normalized_codes = [_normalize_code(c) for c in codes]
        results: list[dict] = []

        if max_workers == 1:
            # 顺序执行，不启动线程池
            for code in normalized_codes:
                try:
                    result = self._scoring_service.score_one_stock_with_profile(
                        code, trade_date, profile_id
                    )
                    results.append({"code": code, "score": result["score"]})
                except Exception as exc:
                    logger.warning(f"股票 {code} 打分失败，已跳过: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_code = {
                    executor.submit(
                        self._scoring_service.score_one_stock_with_profile,
                        code, trade_date, profile_id
                    ): code
                    for code in normalized_codes
                }
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        result = future.result()
                        results.append({"code": code, "score": result["score"]})
                    except Exception as exc:
                        logger.warning(f"股票 {code} 打分失败，已跳过: {exc}")

        if not results:
            return pd.DataFrame(columns=["code", "score", "rank"])

        df = pd.DataFrame(results)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    # ------------------------------------------------------------------
    # TopN 筛选
    # ------------------------------------------------------------------

    def select_top_n(
        self,
        pool_scores: pd.DataFrame,
        top_n: int,
        score_threshold: float | None = None,
    ) -> pd.DataFrame:
        """从池内评分中选出 TopN。

        Args:
            pool_scores: score_pool 返回的 DataFrame，含 code/score/rank 列
            top_n: 最多返回的股票数量
            score_threshold: 可选，只返回 score >= score_threshold 的股票

        Returns:
            行数不超过 top_n 的 DataFrame。
        """
        result = pool_scores.copy()

        if score_threshold is not None:
            result = result[result["score"] >= score_threshold]

        return result.head(top_n).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 股票池列表
    # ------------------------------------------------------------------

    def list_pools(self) -> list[dict]:
        """列出所有已知的时序股票池配置。

        从 data/champions/inline_profiles/ 目录扫描 mode=temporal_pool 的 profile，
        每个 profile 作为一个池条目返回。
        """
        from pathlib import Path
        import json as _json

        _project_root = Path(__file__).resolve().parent.parent.parent
        inline_dir = _project_root / "data" / "champions" / "inline_profiles"
        if not inline_dir.exists():
            return []

        pools: list[dict] = []
        for path in sorted(inline_dir.glob("*.json")):
            try:
                profile = _json.loads(path.read_text(encoding="utf-8"))
                if profile.get("mode") == "temporal_pool":
                    pools.append({
                        "id": profile.get("id", path.stem),
                        "description": profile.get("description", ""),
                        "source": profile.get("source", ""),
                        "version": profile.get("version", ""),
                    })
            except Exception as exc:
                logger.warning(f"读取 pool profile 失败 {path}: {exc}")
        return pools

    # ------------------------------------------------------------------
    # 历史池内评分面板
    # ------------------------------------------------------------------

    def score_pool_history(
        self,
        codes: list[str],
        profile_id: str,
        start_date: str,
        end_date: str,
        max_workers: int = 1,
    ) -> pd.DataFrame:
        """构建历史池内评分面板。

        对每只股票调用 score_stock_history_with_profile，合并后按每个交易日内
        score 降序排名。

        Args:
            codes: 股票代码列表
            profile_id: profile 唯一标识
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            max_workers: 并发线程数，=1 时顺序执行，>1 时使用 ThreadPoolExecutor

        Returns:
            包含 date、code、score、rank 列的 DataFrame，每个交易日内按 score 降序排名。
        """
        normalized_codes = [_normalize_code(c) for c in codes]
        frames: list[pd.DataFrame] = []

        if max_workers == 1:
            for code in normalized_codes:
                try:
                    df = self._scoring_service.score_stock_history_with_profile(
                        code, start_date, end_date, profile_id
                    )
                    frames.append(df)
                except Exception as exc:
                    logger.warning(f"股票 {code} 历史评分失败，已跳过: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_code = {
                    executor.submit(
                        self._scoring_service.score_stock_history_with_profile,
                        code, start_date, end_date, profile_id
                    ): code
                    for code in normalized_codes
                }
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        frames.append(future.result())
                    except Exception as exc:
                        logger.warning(f"股票 {code} 历史评分失败，已跳过: {exc}")

        if not frames:
            return pd.DataFrame(columns=["date", "code", "score", "rank"])

        combined = pd.concat(frames, ignore_index=True)

        # 每个交易日内按 score 降序排名（rank=1 为最高分）
        combined = combined.sort_values(["date", "score"], ascending=[True, False])
        combined["rank"] = combined.groupby("date")["score"].rank(
            method="first", ascending=False
        ).astype(int)

        return combined.reset_index(drop=True)
