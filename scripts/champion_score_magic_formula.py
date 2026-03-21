"""
对 magic_formula_top10_pool.csv 中的股票，用所有 temporal_pool profile 打分，
并通过 walk-forward 回测评估每个 profile 的稳定性，选出每只股票的冠军策略。

用法：
    python scripts/champion_score_magic_formula.py [--date YYYY-MM-DD] [--start YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from backend.services.factor_profile_registry_service import FactorProfileRegistryService
from backend.services.temporal_scoring_service import TemporalScoringService
from backend.services.strategy_evaluation_service import StrategyEvaluationService


def parse_args():
    parser = argparse.ArgumentParser(description="Magic Formula 冠军策略打分")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="当日打分日期")
    parser.add_argument("--start", default="2023-01-01", help="walk-forward 回测起始日期")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="walk-forward 回测结束日期")
    parser.add_argument("--pool", default=str(ROOT / "tests" / "magic_formula_top10_pool.csv"))
    parser.add_argument(
        "--out",
        default=str(ROOT / "tests" / f"champion_scores_{datetime.now().strftime('%Y%m%d')}.csv"),
    )
    parser.add_argument("--no-wf", action="store_true", help="跳过 walk-forward 回测，仅输出当日打分")
    return parser.parse_args()


def score_today(codes: list[str], profiles: list[dict], trade_date: str) -> pd.DataFrame:
    """对所有股票 × 所有 profile 打今日分。"""
    scoring_svc = TemporalScoringService()
    rows = []
    profile_ids = [p["id"] for p in profiles]

    for code in codes:
        row: dict = {"code": code}
        for profile in profiles:
            pid = profile["id"]
            try:
                result = scoring_svc.score_one_stock_with_profile(code, trade_date, pid)
                row[pid] = round(result["score"], 2)
            except Exception as exc:
                row[pid] = None
                print(f"  ⚠ {code}/{pid} 打分失败: {exc}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df["today_avg"] = df[profile_ids].mean(axis=1).round(2)
    df["today_best_profile"] = df[profile_ids].idxmax(axis=1)
    df["today_best_score"] = df[profile_ids].max(axis=1).round(2)
    return df


def run_walk_forward(
    codes: list[str],
    profiles: list[dict],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    对每个 profile 跑 walk-forward 回测，返回 stability_score 表。
    行 = profile_id，列 = stability_score / test_sharpe / test_annual_return / test_max_drawdown
    """
    eval_svc = StrategyEvaluationService()
    rows = []

    for profile in profiles:
        pid = profile["id"]
        candidate = {
            "profile_id": pid,
            "config": {"id": pid, "params": profile.get("params", {})},
        }
        print(f"  walk-forward: {pid} ...", end=" ", flush=True)
        try:
            result = eval_svc.evaluate_temporal_pool_candidate(
                candidate=candidate,
                codes=codes,
                start_date=start_date,
                end_date=end_date,
                train_months=12,
                valid_months=3,
                test_months=3,
                step_months=3,
            )
            tm = result.get("test_metrics", {})
            rows.append({
                "profile_id": pid,
                "stability_score": round(result.get("stability_score", 0.0), 2),
                "test_sharpe": round(tm.get("sharpe", float("nan")), 3),
                "test_annual_return": round(tm.get("annual_return", float("nan")), 4),
                "test_max_drawdown": round(tm.get("max_drawdown", float("nan")), 4),
                "n_windows": len(result.get("window_metrics", [])),
            })
            print(f"stability={result.get('stability_score', 0):.1f}")
        except Exception as exc:
            print(f"失败: {exc}")
            rows.append({
                "profile_id": pid,
                "stability_score": 0.0,
                "test_sharpe": float("nan"),
                "test_annual_return": float("nan"),
                "test_max_drawdown": float("nan"),
                "n_windows": 0,
            })

    return pd.DataFrame(rows).set_index("profile_id")


def print_summary(final_df: pd.DataFrame, wf_df: pd.DataFrame | None, profile_ids: list[str]) -> None:
    print("\n" + "=" * 80)
    print("  Magic Formula 冠军策略打分结果")
    print("=" * 80)

    display_cols = ["code", "today_avg", "today_best_profile", "today_best_score"]
    if wf_df is not None:
        display_cols += ["champion_profile", "champion_stability", "champion_today_score"]

    print(final_df[display_cols].to_string(index=False))

    if wf_df is not None:
        print("\n" + "=" * 80)
        print("  各策略 Walk-Forward 稳定性排名")
        print("=" * 80)
        wf_sorted = wf_df.sort_values("stability_score", ascending=False)
        print(wf_sorted[["stability_score", "test_sharpe", "test_annual_return", "test_max_drawdown", "n_windows"]].to_string())

    print("\n各策略今日 Top3：")
    for pid in profile_ids:
        if pid not in final_df.columns:
            continue
        top3 = final_df.nlargest(3, pid)[["code", pid]]
        codes_str = "  ".join(f"{r['code']}({r[pid]})" for _, r in top3.iterrows())
        print(f"  {pid:25s}: {codes_str}")

    print("=" * 80)


def main():
    args = parse_args()

    pool_df = pd.read_csv(args.pool)
    codes = pool_df["code"].astype(str).str.strip().tolist()
    print(f"股票池：{len(codes)} 只  |  打分日期：{args.date}  |  回测区间：{args.start} ~ {args.end}")

    registry = FactorProfileRegistryService()
    profiles = registry.load_profiles("temporal_pool")
    profile_ids = [p["id"] for p in profiles]
    print(f"加载 {len(profiles)} 个 profile：{profile_ids}")

    # ── 今日打分 ──────────────────────────────────────────────────────────────
    print("\n[1/2] 今日打分...")
    score_df = score_today(codes, profiles, args.date)

    # ── Walk-Forward 回测 ─────────────────────────────────────────────────────
    wf_df: pd.DataFrame | None = None
    if not args.no_wf:
        print(f"\n[2/2] Walk-Forward 回测（{args.start} ~ {args.end}）...")
        wf_df = run_walk_forward(codes, profiles, args.start, args.end)

    # ── 合并：冠军策略 = stability_score 最高的 profile ──────────────────────
    final_df = score_df.copy()

    if wf_df is not None:
        best_profile = wf_df["stability_score"].idxmax()
        best_stability = wf_df.loc[best_profile, "stability_score"]

        # 冠军策略对每只股票都一样（整个池子共用一个最优 profile）
        final_df["champion_profile"] = best_profile
        final_df["champion_stability"] = best_stability
        final_df["champion_today_score"] = final_df[best_profile]

    # 按今日平均分排序
    final_df = final_df.sort_values("today_avg", ascending=False).reset_index(drop=True)
    final_df.insert(0, "rank", range(1, len(final_df) + 1))

    # ── 打印 ──────────────────────────────────────────────────────────────────
    print_summary(final_df, wf_df, profile_ids)

    # ── 保存 ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out_path}")

    if wf_df is not None:
        wf_out = out_path.with_name(out_path.stem + "_wf_stability.csv")
        wf_df.reset_index().to_csv(wf_out, index=False, encoding="utf-8-sig")
        print(f"稳定性报告：{wf_out}")


if __name__ == "__main__":
    main()
