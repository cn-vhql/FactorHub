"""
对 magic_formula_top10_pool.csv 中的股票，用所有 temporal_pool profile 打分，
输出每只股票在每个策略下的得分，并同时展示策略配置摘要。

用法：
    python scripts/score_magic_formula_pool.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from backend.services.factor_profile_registry_service import FactorProfileRegistryService
from backend.services.temporal_scoring_service import TemporalScoringService


# ── 参数解析 ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Magic Formula 股票池多策略打分")
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="交易日期，格式 YYYY-MM-DD（默认今日）",
    )
    parser.add_argument(
        "--pool",
        default=str(ROOT / "scripts" / "data" / "magic_formula_top10_pool.csv"),
        help="股票池 CSV 路径",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "scripts" / "data" / f"magic_formula_scores_{datetime.now().strftime('%Y%m%d')}.csv"),
        help="输出 CSV 路径",
    )
    return parser.parse_args()


# ── 策略摘要打印 ──────────────────────────────────────────────────────────────

def print_profile_summary(profiles: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  策略配置摘要")
    print("=" * 70)
    for p in profiles:
        signal_factors = [f for f in p["factors"] if f.get("role") == "signal"]
        risk_factors   = [f for f in p["factors"] if f.get("role") == "risk_filter"]
        params = p.get("params", {})

        print(f"\n【{p['id']}】{p['description']}")
        print(f"  调仓周期: {params.get('rebalance')}  持仓天数: {params.get('hold_days')}  "
              f"TopN: {params.get('top_n')}  分数阈值: {params.get('score_threshold')}  "
              f"百分位窗口: {params.get('percentile_window')}")
        print("  信号因子:")
        for f in signal_factors:
            direction = "↑正向" if f["direction"] == 1 else "↓反向"
            print(f"    {f['name']:25s}  权重={f['weight']:.2f}  {direction}")
        if risk_factors:
            print("  风险过滤因子:")
            for f in risk_factors:
                print(f"    {f['name']:25s}  阈值={f.get('threshold', '-')}")
    print("=" * 70 + "\n")


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    trade_date = args.date

    # 加载股票池（dtype=str 保留前导零，如 000928）
    pool_df = pd.read_csv(args.pool, dtype=str)
    codes = pool_df["code"].str.strip().str.zfill(6).tolist()
    print(f"股票池：{len(codes)} 只股票，交易日期：{trade_date}")

    # 加载所有 temporal_pool profile
    registry = FactorProfileRegistryService()
    profiles = registry.load_profiles("temporal_pool")
    print(f"加载策略：{len(profiles)} 个 profile")

    # 打印策略摘要
    print_profile_summary(profiles)

    # 打分
    scoring_svc = TemporalScoringService()
    rows = []

    for code in codes:
        row = {"code": code}
        for profile in profiles:
            pid = profile["id"]
            try:
                result = scoring_svc.score_one_stock_with_profile(code, trade_date, pid)
                row[pid] = round(result["score"], 2)
            except Exception as exc:
                row[pid] = None
                print(f"  ⚠ {code} / {pid} 打分失败: {exc}")
        rows.append(row)

    # 构建结果 DataFrame
    result_df = pd.DataFrame(rows)
    profile_ids = [p["id"] for p in profiles]

    # 计算平均分和最佳策略
    result_df["avg_score"] = result_df[profile_ids].mean(axis=1).round(2)
    result_df["best_profile"] = result_df[profile_ids].idxmax(axis=1)
    result_df["best_score"] = result_df[profile_ids].max(axis=1).round(2)

    # 按平均分降序排列
    result_df = result_df.sort_values("avg_score", ascending=False).reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))

    # 打印结果
    print("\n" + "=" * 70)
    print(f"  打分结果（交易日期：{trade_date}）")
    print("=" * 70)
    print(result_df.to_string(index=False))
    print("=" * 70)

    # 保存 CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到：{out_path}")

    # 打印各策略下的 Top3
    print("\n各策略 Top3：")
    for pid in profile_ids:
        top3 = result_df.nlargest(3, pid)[["code", pid]]
        codes_str = "  ".join(f"{r['code']}({r[pid]})" for _, r in top3.iterrows())
        print(f"  {pid:20s}: {codes_str}")


if __name__ == "__main__":
    main()
