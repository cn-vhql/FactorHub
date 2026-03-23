"""Stage-3: 单票近期有效因子挖掘。"""

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
    parser = argparse.ArgumentParser(description="挖掘单只股票近期可用因子")
    parser.add_argument("--stock", default="600089", help="股票代码")
    parser.add_argument("--end-date", default="2026-03-21", help="评估结束日期 YYYY-MM-DD")
    parser.add_argument("--eval-days", type=int, default=90, help="近期评估窗口（自然日）")
    parser.add_argument("--lookback-days", type=int, default=240, help="因子预热窗口（自然日）")
    parser.add_argument("--horizons", default="1,5,10", help="未来收益周期列表")
    parser.add_argument("--top", type=int, default=20, help="输出 Top 因子数量")
    parser.add_argument("--min-samples", type=int, default=25, help="最小样本数")
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.85,
        help="去重相关阈值",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="输出目录；为空时按 stock/end_date 自动生成",
    )
    return parser.parse_args()


def build_summary(result: dict, stock_code: str) -> str:
    meta = result["metadata"]
    per_stock = result["per_stock_top_factors"].get(stock_code, {})
    lines = [
        "# Stage-3 单票近期因子挖掘",
        "",
        f"- 股票: {stock_code}",
        f"- 请求结束日: {meta['requested_end_date']}",
        f"- 实际结束日: {meta['actual_end_date']}",
        f"- 近期窗口起点: {meta['eval_start_date']}",
        f"- 当前状态: {per_stock.get('current_regime', 'unknown')}",
        f"- 全局 Top 因子数: {len(result.get('top_factors', []))}",
        "",
        "## 单票 Top 因子",
        "",
    ]
    for idx, item in enumerate(per_stock.get("top_factors", [])[:10], start=1):
        direction = "正向" if item["recommended_direction"] >= 0 else "反向"
        lines.append(
            f"{idx}. {item['factor_name']} | score={item['stock_score']:.2f} "
            f"| {direction} | absIC={item['abs_ic_mean']:.4f} | absIR={item['abs_ir']:.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    bootstrap_runtime()

    from backend.services.recent_factor_mining_service import (
        recent_factor_mining_service,
    )

    stock_code = normalize_stock_code(args.stock)
    horizons = parse_int_list(args.horizons)
    output_dir = (
        ensure_output_dir(args.output_dir)
        if args.output_dir
        else ensure_output_dir(
            f"data/pipeline_runs/stage3_single_stock_recent_factor_mining/"
            f"{stock_code}_{args.end_date}"
        )
    )

    result = recent_factor_mining_service.mine_recent_valuable_factors(
        stock_codes=[stock_code],
        end_date=args.end_date,
        eval_days=args.eval_days,
        lookback_days=args.lookback_days,
        horizons=horizons,
        top_k=args.top,
        min_samples=args.min_samples,
        correlation_threshold=args.corr_threshold,
    )

    per_stock = result["per_stock_top_factors"].get(stock_code, {})

    write_json(output_dir / "result.json", result)
    write_csv(output_dir / "top_factors.csv", result["top_factors"])
    write_csv(output_dir / "single_stock_top_factors.csv", per_stock.get("top_factors", []))
    write_text(output_dir / "summary.md", build_summary(result, stock_code))

    print("Stage-3 完成：单票近期因子挖掘")
    print(f"输出目录: {output_dir}")
    print(
        f"股票: {stock_code} | 当前状态: {per_stock.get('current_regime', 'unknown')} | "
        f"Top 因子数: {len(per_stock.get('top_factors', []))}"
    )


if __name__ == "__main__":
    main()
