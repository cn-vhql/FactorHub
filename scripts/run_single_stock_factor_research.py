"""Stage-2: 单票历史因子研究。"""

from __future__ import annotations

import argparse

from pipeline_common import (
    bootstrap_runtime,
    ensure_output_dir,
    normalize_stock_code,
    parse_int_list,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="研究单只股票的历史有效因子")
    parser.add_argument("--stock", default="600089", help="股票代码，如 600519 或 000001.SZ")
    parser.add_argument("--start-date", default="2023-01-01", help="研究开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-03-21", help="研究结束日期 YYYY-MM-DD")
    parser.add_argument("--horizons", default="1,5,10,20", help="未来收益周期列表")
    parser.add_argument("--lookback-days", type=int, default=252, help="因子预热窗口（自然日）")
    parser.add_argument("--top", type=int, default=20, help="输出 Top 因子数量")
    parser.add_argument("--min-samples", type=int, default=60, help="单因子最小样本数")
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.85,
        help="去重相关阈值",
    )
    parser.add_argument(
        "--factors",
        default="",
        help="可选，限定研究因子名，逗号分隔；为空时默认研究全部启用因子",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="输出目录；为空时按 stock/start/end 自动生成",
    )
    return parser.parse_args()


def build_summary(result: dict) -> str:
    meta = result["metadata"]
    top_factors = result["top_factors"][:10]
    lines = [
        "# Stage-2 单票历史因子研究",
        "",
        f"- 股票: {meta['stock_code']}",
        f"- 分析区间: {meta['analysis_start_date']} ~ {meta['analysis_end_date']}",
        f"- 实际结束日: {meta['actual_end_date']}",
        f"- 当前状态: {meta['current_regime']}",
        f"- 研究因子数: {meta['factor_count_scored']}/{meta['factor_count_total']}",
        f"- 选中因子数: {meta['selected_factor_count']}",
        "",
        "## Top 因子",
        "",
    ]
    for idx, item in enumerate(top_factors, start=1):
        direction = "正向" if item["recommended_direction"] >= 0 else "反向"
        lines.append(
            f"{idx}. {item['factor_name']} | score={item['historical_score']:.2f} "
            f"| {direction} | absIC={item['abs_ic_mean']:.4f} | absIR={item['abs_ir']:.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    bootstrap_runtime()

    from backend.services.single_stock_factor_research_service import (
        single_stock_factor_research_service,
    )

    stock_code = normalize_stock_code(args.stock)
    horizons = parse_int_list(args.horizons)
    factor_names = [item.strip() for item in args.factors.split(",") if item.strip()]

    output_dir = (
        ensure_output_dir(args.output_dir)
        if args.output_dir
        else ensure_output_dir(
            f"data/pipeline_runs/stage2_single_stock_factor_research/"
            f"{stock_code}_{args.start_date}_{args.end_date}"
        )
    )

    result = single_stock_factor_research_service.research_factors(
        stock_code=stock_code,
        start_date=args.start_date,
        end_date=args.end_date,
        factor_names=factor_names or None,
        horizons=horizons,
        lookback_days=args.lookback_days,
        top_k=args.top,
        min_samples=args.min_samples,
        correlation_threshold=args.corr_threshold,
    )

    write_json(output_dir / "result.json", result)
    write_csv(output_dir / "top_factors.csv", result["top_factors"])
    write_csv(output_dir / "all_factor_scores.csv", result["all_factor_scores"])
    write_text(output_dir / "summary.md", build_summary(result))

    print("Stage-2 完成：单票历史因子研究")
    print(f"输出目录: {output_dir}")
    print(
        f"股票: {stock_code} | Top 因子数: {len(result['top_factors'])} | "
        f"当前状态: {result['metadata']['current_regime']}"
    )


if __name__ == "__main__":
    main()
