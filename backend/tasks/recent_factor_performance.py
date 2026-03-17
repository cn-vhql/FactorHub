"""离线批处理：拉取库中全部因子并评估最近三个月表现。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.database import get_db_session
from backend.core.settings import settings
from backend.repositories.factor_repository import FactorRepository
from backend.services.analysis_service import AnalysisService
from backend.services.factor_service import factor_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从数据库拉取全部启用因子，评估最近三个月 IC/IR 表现",
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        default=["000001.SZ", "000002.SZ", "600000.SH", "600519.SH", "300750.SZ"],
        help="股票代码列表（默认给出 5 只示例股票）",
    )
    parser.add_argument("--lookback-days", type=int, default=180, help="回看窗口（用于因子计算预热）")
    parser.add_argument("--end-date", default=settings.DEFAULT_END_DATE, help="评估结束日期，默认使用项目配置中的 DEFAULT_END_DATE")
    parser.add_argument("--eval-days", type=int, default=90, help="评估窗口（最近N天）")
    parser.add_argument(
        "--output",
        default="data/reports/recent_factor_performance.json",
        help="输出 JSON 文件路径",
    )
    parser.add_argument("--top", type=int, default=20, help="展示前 N 个因子")
    return parser.parse_args()


def load_all_active_factor_names() -> list[str]:
    db = get_db_session()
    repo = FactorRepository(db)
    factors = repo.get_all(active_only=True)
    db.close()
    return [factor.name for factor in factors]


def main() -> None:
    args = parse_args()
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=args.lookback_days)
    eval_start_dt = end_dt - timedelta(days=args.eval_days)

    factor_names = load_all_active_factor_names()
    if not factor_names:
        raise SystemExit("未在数据库中找到任何启用因子")

    print(f"[1/3] 已读取因子数量: {len(factor_names)}")
    print(f"[2/3] 开始计算股票池因子: {args.stocks}")
    factor_data = factor_service.calculate_factors_for_stocks(
        stock_codes=args.stocks,
        factor_names=factor_names,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
    )

    if not factor_data:
        raise SystemExit("未能计算出任何股票的因子数据，请检查股票代码或数据源")

    latest_dates = [df.index.max() for df in factor_data.values() if not df.empty]
    if not latest_dates:
        raise SystemExit("所有股票都没有可用行情数据")

    actual_end_dt = max(latest_dates).to_pydatetime()
    eval_start_dt = actual_end_dt - timedelta(days=args.eval_days)

    trimmed_data: dict[str, pd.DataFrame] = {}
    for stock_code, df in factor_data.items():
        trimmed = df[df.index >= pd.Timestamp(eval_start_dt)].copy()
        if len(trimmed) >= 20:
            trimmed_data[stock_code] = trimmed

    if not trimmed_data:
        raise SystemExit("最近三个月有效数据不足（<20 条），无法评估")

    print(f"[3/3] 对最近 {args.eval_days} 天执行 IC/IR 评估")
    analysis_service = AnalysisService()
    ic_ir = analysis_service.calculate_ic_ir(trimmed_data, factor_names)
    ic_stats = ic_ir.get("ic_stats", {})

    ranking = []
    for name, stats in ic_stats.items():
        ranking.append(
            {
                "factor_name": name,
                "ic_mean": float(stats.get("IC均值", 0.0)),
                "ic_std": float(stats.get("IC标准差", 0.0)),
                "ir": float(stats.get("IR", 0.0)),
                "positive_ratio": float(stats.get("IC>0占比", 0.0)),
                "abs_ic_mean": float(stats.get("IC绝对值均值", 0.0)),
                "ic_type": stats.get("IC类型", ""),
            }
        )

    ranking.sort(key=lambda x: abs(x["ic_mean"]), reverse=True)

    summary = {
        "metadata": {
            "stocks": args.stocks,
            "factor_count": len(factor_names),
            "stocks_with_data": list(trimmed_data.keys()),
            "lookback_days": args.lookback_days,
            "eval_days": args.eval_days,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "requested_end_date": end_dt.strftime("%Y-%m-%d"),
            "actual_end_date": actual_end_dt.strftime("%Y-%m-%d"),
            "eval_start_date": eval_start_dt.strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
        },
        "top_factors": ranking[: args.top],
        "all_factors": ranking,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评估完成，结果已保存: {output_path}")
    print("\n最近阶段表现最强的因子（按 |IC均值| 排序）:")
    for idx, item in enumerate(summary["top_factors"], start=1):
        print(
            f"{idx:>2}. {item['factor_name']:<30} "
            f"IC均值={item['ic_mean']:+.4f}  IR={item['ir']:+.4f}  "
            f"IC>0占比={item['positive_ratio']:.2%}"
        )


if __name__ == "__main__":
    main()
