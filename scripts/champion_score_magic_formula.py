"""
对 magic_formula_top10_pool.csv 中的每只股票，独立搜索其冠军策略，
再用各自冠军 profile 打当日分，汇总输出 Top10（附失败原因）。

用法：
    python scripts/champion_score_magic_formula.py [--date YYYY-MM-DD] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
    python scripts/champion_score_magic_formula.py --no-wf   # 跳过搜索，直接用已缓存冠军打分
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from backend.services.champion_strategy_service import ChampionStrategyService
from backend.services.temporal_scoring_service import TemporalScoringService


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    today = datetime.now().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Magic Formula 逐股冠军策略打分")
    parser.add_argument("--date",  default=today,         help="当日打分日期 YYYY-MM-DD")
    parser.add_argument("--start", default="2022-01-01",  help="冠军搜索回测起始日期（需 ≥18 个月跨度）")
    parser.add_argument("--end",   default=today,         help="冠军搜索回测结束日期")
    parser.add_argument("--pool",  default=str(ROOT / "scripts" / "data" / "magic_formula_top10_pool.csv"))
    parser.add_argument("--space", default="temporal_default", help="search_space_id")
    parser.add_argument(
        "--out",
        default=str(ROOT / "scripts" / "data" / f"champion_scores_{datetime.now().strftime('%Y%m%d')}.csv"),
    )
    parser.add_argument(
        "--no-wf", action="store_true",
        help="跳过冠军搜索，直接从 registry 读取已缓存冠军策略打分",
    )
    return parser.parse_args()


# ── 逐股冠军搜索 ──────────────────────────────────────────────────────────────

def find_per_stock_champions(
    codes: list[str],
    search_space_id: str,
    start_date: str,
    end_date: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    对每只股票独立运行 run_for_stock，返回：
      champions: {code: profile_id}   — 成功的股票
      search_errors: {code: reason}   — 搜索失败的股票
    """
    svc = ChampionStrategyService()
    champions: dict[str, str] = {}
    search_errors: dict[str, str] = {}

    for i, code in enumerate(codes, 1):
        print(f"  [{i}/{len(codes)}] 搜索 {code} 冠军策略 ...", end=" ", flush=True)
        try:
            result = svc.run_for_stock(
                code=code,
                search_space_id=search_space_id,
                start_date=start_date,
                end_date=end_date,
            )
            pid = result.get("profile_id", "")
            stability = result.get("stability_score", float("nan"))
            champions[code] = pid
            print(f"✓ {pid}  stability={stability:.1f}")
        except Exception as exc:
            search_errors[code] = str(exc)
            print(f"✗ {exc}")

    return champions, search_errors


def load_cached_champions(codes: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """
    从 ChampionRegistryService 读取已缓存的逐股冠军（scope_type=single_stock）。
    返回 (champions, missing)，missing 中的股票没有缓存记录。
    """
    svc = ChampionStrategyService()
    champions: dict[str, str] = {}
    missing: dict[str, str] = {}

    for code in codes:
        champion = svc.get_champion("single_stock", code)
        if champion and champion.get("profile_id"):
            champions[code] = champion["profile_id"]
        else:
            missing[code] = "registry 中无冠军记录"

    return champions, missing


# ── 逐股打分 ──────────────────────────────────────────────────────────────────

def score_with_champions(
    champions: dict[str, str],
    trade_date: str,
) -> tuple[list[dict], dict[str, str]]:
    """
    用各股自己的冠军 profile 打当日分。
    返回 (results, score_errors)。
    """
    scoring_svc = TemporalScoringService()
    results: list[dict] = []
    score_errors: dict[str, str] = {}

    for code, profile_id in champions.items():
        try:
            r = scoring_svc.score_one_stock_with_profile(code, trade_date, profile_id)
            results.append({
                "code": code,
                "champion_profile": profile_id,
                "score": round(r["score"], 2),
            })
        except Exception as exc:
            score_errors[code] = str(exc)

    return results, score_errors


# ── 汇总输出 ──────────────────────────────────────────────────────────────────

def build_summary(
    results: list[dict],
    score_errors: dict[str, str],
    search_errors: dict[str, str],
) -> pd.DataFrame:
    """合并成功结果 + 失败记录，按 score 降序，附 rank。"""
    rows = []

    # 成功的股票
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        rows.append({
            "code": r["code"],
            "champion_profile": r["champion_profile"],
            "score": r["score"],
            "status": "ok",
            "error": "",
        })

    # 打分失败（冠军找到但打分出错）
    for code, reason in score_errors.items():
        rows.append({
            "code": code,
            "champion_profile": "",
            "score": float("nan"),
            "status": "score_error",
            "error": reason,
        })

    # 搜索失败（连冠军都没找到）
    for code, reason in search_errors.items():
        rows.append({
            "code": code,
            "champion_profile": "",
            "score": float("nan"),
            "status": "search_error",
            "error": reason,
        })

    df = pd.DataFrame(rows)
    # rank 只给成功的行
    ok_mask = df["status"] == "ok"
    df.loc[ok_mask, "rank"] = range(1, ok_mask.sum() + 1)
    df["rank"] = df["rank"].astype("Int64")  # nullable int，失败行显示 <NA>
    cols = ["rank", "code", "champion_profile", "score", "status", "error"]
    return df[cols].reset_index(drop=True)


def print_summary(df: pd.DataFrame, trade_date: str) -> None:
    ok = df[df["status"] == "ok"]
    fail = df[df["status"] != "ok"]

    print("\n" + "=" * 70)
    print(f"  逐股冠军策略打分结果  |  日期：{trade_date}")
    print("=" * 70)

    print(f"\nTop {min(10, len(ok))} 股票：")
    top10 = ok.head(10)[["rank", "code", "champion_profile", "score"]]
    print(top10.to_string(index=False))

    if not fail.empty:
        print(f"\n失败股票（{len(fail)} 只）：")
        print(fail[["code", "status", "error"]].to_string(index=False))

    print("=" * 70)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    pool_df = pd.read_csv(args.pool)
    codes = pool_df["code"].astype(str).str.zfill(6).str.strip().tolist()
    print(f"股票池：{len(codes)} 只  |  打分日期：{args.date}")

    # ── Step 1：获取每股冠军 profile ─────────────────────────────────────────
    if args.no_wf:
        print("\n[1/2] 从 registry 读取已缓存冠军策略...")
        champions, search_errors = load_cached_champions(codes)
        print(f"  命中 {len(champions)} 只，缺失 {len(search_errors)} 只")
    else:
        print(f"\n[1/2] 逐股搜索冠军策略（{args.start} ~ {args.end}，space={args.space}）...")
        champions, search_errors = find_per_stock_champions(
            codes, args.space, args.start, args.end
        )
        print(f"  成功 {len(champions)} 只，失败 {len(search_errors)} 只")

    # ── Step 2：用各自冠军打当日分 ───────────────────────────────────────────
    print(f"\n[2/2] 用冠军 profile 打 {args.date} 当日分...")
    results, score_errors = score_with_champions(champions, args.date)
    print(f"  成功 {len(results)} 只，失败 {len(score_errors)} 只")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    df = build_summary(results, score_errors, search_errors)
    print_summary(df, args.date)

    # ── 保存 ─────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out_path}")


if __name__ == "__main__":
    main()
