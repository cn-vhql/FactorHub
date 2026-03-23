"""Stage-4: 单票策略搜索与评分。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pipeline_common import (
    bootstrap_runtime,
    ensure_output_dir,
    normalize_stock_code,
    write_json,
    write_text,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为单只股票搜索冠军策略")
    parser.add_argument("--stock", default="600089", help="股票代码")
    parser.add_argument("--start-date", default="2023-01-01", help="搜索开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-03-21", help="搜索结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--search-space-id",
        default="temporal_default",
        help="搜索空间 ID",
    )
    parser.add_argument(
        "--stage2-result",
        default="",
        help="Stage-2 输出的 result.json 路径；为空时自动按 stock/end_date 推断",
    )
    parser.add_argument(
        "--stage3-result",
        default="",
        help="Stage-3 输出的 result.json 路径；为空时自动按 stock/end_date 推断",
    )
    parser.add_argument(
        "--top-factors-source",
        default="stage3",
        choices=["stage2", "stage3", "none"],
        help="注入因子来源：stage3（近期）/ stage2（历史）/ none（仅用全局 config）",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="输出目录；为空时按 stock/end_date 自动生成",
    )
    return parser.parse_args()


def _load_top_factors(result_path: Path) -> list[dict] | None:
    """从 Stage-2/3 的 result.json 中读取 top_factors 列表。"""
    if not result_path.exists():
        logger.warning(f"因子结果文件不存在，跳过注入: {result_path}")
        return None
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        # Stage-2: data["top_factors"]
        # Stage-3: data["top_factors"]（全局）或 data["per_stock_top_factors"][code]["top_factors"]
        top_factors = data.get("top_factors", [])
        if not top_factors:
            logger.warning(f"top_factors 为空，跳过注入: {result_path}")
            return None
        logger.info(f"从 {result_path} 加载 {len(top_factors)} 个 top_factors")
        return top_factors
    except Exception as exc:
        logger.warning(f"读取因子结果文件失败，跳过注入: {result_path} | {exc}")
        return None


def _resolve_stage_result(explicit_path: str, stock_code: str, end_date: str, stage: str) -> Path:
    """解析 Stage-2/3 result.json 路径：优先用显式传入，否则按约定路径推断。"""
    if explicit_path:
        return Path(explicit_path)
    if stage == "stage2":
        # Stage-2 目录名含 start_date，无法精确推断，取最新一个
        base = Path(f"data/pipeline_runs/stage2_single_stock_factor_research")
        candidates = sorted(base.glob(f"{stock_code}_*/result.json"), reverse=True)
        return candidates[0] if candidates else Path(explicit_path or "")
    else:
        return Path(
            f"data/pipeline_runs/stage3_single_stock_recent_factor_mining/"
            f"{stock_code}_{end_date}/result.json"
        )


def build_summary(result: dict, top_factors_source: str, top_factors_count: int) -> str:
    metrics = result.get("metrics", {})
    lines = [
        "# Stage-4 单票策略搜索",
        "",
        f"- 股票: {result['scope_key']}",
        f"- champion_profile: {result['profile_id']}",
        f"- stability_score: {result['stability_score']:.2f}",
        f"- effective_from: {result['effective_from']}",
        f"- report_path: {result['report_path']}",
        f"- 因子来源: {top_factors_source}（{top_factors_count} 个注入因子）",
        "",
        "## 测试期指标",
        "",
        f"- annual_return: {metrics.get('annual_return')}",
        f"- sharpe: {metrics.get('sharpe')}",
        f"- max_drawdown: {metrics.get('max_drawdown')}",
        f"- active_ratio: {metrics.get('active_ratio')}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    bootstrap_runtime()

    from backend.services.champion_strategy_service import ChampionStrategyService

    stock_code = normalize_stock_code(args.stock)
    output_dir = (
        ensure_output_dir(args.output_dir)
        if args.output_dir
        else ensure_output_dir(
            f"data/pipeline_runs/stage4_single_stock_strategy_search/"
            f"{stock_code}_{args.end_date}"
        )
    )

    # 加载 Stage-2/3 top_factors（数据结果依赖，非代码依赖）
    top_factors: list[dict] | None = None
    top_factors_source = "none（全局 config）"
    top_factors_count = 0

    if args.top_factors_source != "none":
        result_path = _resolve_stage_result(
            explicit_path=args.stage3_result if args.top_factors_source == "stage3" else args.stage2_result,
            stock_code=stock_code,
            end_date=args.end_date,
            stage=args.top_factors_source,
        )
        top_factors = _load_top_factors(result_path)
        if top_factors:
            top_factors_source = f"{args.top_factors_source}（{result_path}）"
            top_factors_count = len(top_factors)
        else:
            print(f"[Stage-4] 未找到 {args.top_factors_source} 因子结果，回退到全局 config")

    svc = ChampionStrategyService()
    result = svc.run_for_stock(
        code=stock_code,
        search_space_id=args.search_space_id,
        start_date=args.start_date,
        end_date=args.end_date,
        top_factors=top_factors,
    )

    payload = {
        "metadata": {
            "stage": "stage4_single_stock_strategy_search",
            "stock_code": stock_code,
            "search_space_id": args.search_space_id,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "top_factors_source": args.top_factors_source,
            "top_factors_count": top_factors_count,
        },
        "champion": result,
    }

    write_json(output_dir / "result.json", payload)
    write_text(output_dir / "summary.md", build_summary(result, top_factors_source, top_factors_count))

    print("Stage-4 完成：单票策略搜索")
    print(f"输出目录: {output_dir}")
    print(f"因子来源: {top_factors_source}")
    print(
        f"股票: {stock_code} | champion_profile: {result['profile_id']} | "
        f"stability_score: {result['stability_score']:.2f}"
    )
    print(f"详细评估目录: {result['report_path']}")


if __name__ == "__main__":
    main()
