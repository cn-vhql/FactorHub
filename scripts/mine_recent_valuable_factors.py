"""离线批处理：自动挖掘最近阶段高价值因子。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.recent_factor_mining_service import recent_factor_mining_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动挖掘最近窗口高价值因子")
    parser.add_argument(
        "--stocks",
        nargs="+",
        default=["002594"],
        help="已筛选股票列表，如 000001.SZ 600519.SH",
    )
    parser.add_argument("--end-date", default="2026-03-21", help="评估结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--eval-days", type=int, default=90, help="最近评估窗口（自然日）")
    parser.add_argument("--lookback-days", type=int, default=240, help="因子预热窗口（自然日）")
    parser.add_argument(
        "--horizons",
        default="1,5,10",
        help="未来收益周期，逗号分隔，如 1,5,10",
    )
    parser.add_argument("--top", type=int, default=20, help="输出 Top 因子数量")
    parser.add_argument("--min-samples", type=int, default=25, help="最小样本数")
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.85,
        help="去重相关阈值（绝对值）",
    )
    parser.add_argument(
        "--output",
        default="data/reports/recent_valuable_factors.json",
        help="JSON 输出路径",
    )
    parser.add_argument(
        "--csv-output",
        default="data/reports/recent_valuable_factors_top.csv",
        help="Top 因子 CSV 输出路径",
    )
    return parser.parse_args()


def parse_horizons(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("horizons 解析为空，请至少提供一个周期")
    return values


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)

    result = recent_factor_mining_service.mine_recent_valuable_factors(
        stock_codes=args.stocks,
        end_date=args.end_date,
        eval_days=args.eval_days,
        lookback_days=args.lookback_days,
        horizons=horizons,
        top_k=args.top,
        min_samples=args.min_samples,
        correlation_threshold=args.corr_threshold,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["top_factors"]).to_csv(csv_path, index=False, encoding="utf-8-sig")

    metadata = result.get("metadata", {})
    print("挖掘完成。")
    print(
        f"评估区间: {metadata.get('eval_start_date')} ~ {metadata.get('actual_end_date')} "
        f"| 股票数: {metadata.get('stock_count_with_result')}/{metadata.get('stock_count_requested')} "
        f"| 因子数: {metadata.get('factor_count_scored')}/{metadata.get('factor_count_total')}"
    )
    print(f"JSON 输出: {output_path}")
    print(f"CSV 输出:  {csv_path}")
    print("\nTop 因子:")
    for idx, item in enumerate(result.get("top_factors", []), start=1):
        direction = "正向" if item.get("recommended_direction", 1) >= 0 else "反向"
        print(
            f"{idx:>2}. {item['factor_name']:<30} "
            f"score={item['score']:.2f}  {direction}  "
            f"absIC={item['abs_ic_mean']:.4f}  regimeAbsIC={item['regime_abs_ic_mean']:.4f}"
        )


if __name__ == "__main__":
    main()

