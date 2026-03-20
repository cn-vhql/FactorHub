"""
验证测试：对神奇公式排名股票池跑时序打分，输出合并结果 CSV。

用法：
    .venv/bin/python tests/test_magic_formula_scoring.py

输出：
    tests/神奇公式_temporal_score_20260320.csv
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.services.temporal_scoring_service import temporal_scoring_service

# ── 配置 ──────────────────────────────────────────────────────────────────────
INPUT_CSV = ROOT / "tests" / "神奇公式排名_a_20260320_183711.csv"
OUTPUT_CSV = ROOT / "tests" / "神奇公式_temporal_score_20260320.csv"
TRADE_DATE = "2026-03-20"
# 每批并发数（避免 akshare 限流，可调小）
BATCH_SIZE = 20
# ─────────────────────────────────────────────────────────────────────────────


def parse_excel_formula(val: str) -> str:
    """去掉 Excel 公式前缀 =""  """
    s = str(val).strip()
    m = re.match(r'^="?([^"]*)"?$', s)
    return m.group(1) if m else s


def load_codes(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df = df.dropna(subset=["代码"])
    df["代码"] = df["代码"].apply(parse_excel_formula)
    df = df[df["代码"].str.match(r"^\d{6}$")]  # 只保留 6 位数字代码
    # 过滤北交所（8/4/9 开头），akshare stock_zh_a_daily 不支持
    df = df[~df["代码"].str.match(r"^[489]")]
    df["复合排名"] = df["复合排名"].apply(parse_excel_formula)
    df["复合排名"] = pd.to_numeric(df["复合排名"], errors="coerce")
    return df.reset_index(drop=True)


def score_in_batches(codes: list[str], trade_date: str) -> tuple[list[dict], list[dict]]:
    all_results, all_errors = [], []
    total = len(codes)
    for i in range(0, total, BATCH_SIZE):
        batch = codes[i: i + BATCH_SIZE]
        print(f"  [{i + len(batch)}/{total}] 正在评分 {batch[0]} ~ {batch[-1]} ...", flush=True)
        t0 = time.time()
        results, errors = temporal_scoring_service.score_many_stocks(batch, trade_date)
        elapsed = time.time() - t0
        print(f"    成功 {len(results)}，失败 {len(errors)}，耗时 {elapsed:.1f}s", flush=True)
        all_results.extend(results)
        all_errors.extend(errors)
    return all_results, all_errors


def main():
    print(f"读取输入文件：{INPUT_CSV}")
    magic_df = load_codes(INPUT_CSV)
    codes = magic_df["代码"].tolist()[:200]   # 只跑前 200
    magic_df = magic_df.iloc[:200].reset_index(drop=True)
    print(f"共 {len(codes)} 只股票（已过滤北交所），交易日 {TRADE_DATE}\n")

    print("开始时序打分...")
    t_start = time.time()
    results, errors = score_in_batches(codes, TRADE_DATE)
    total_elapsed = time.time() - t_start
    print(f"\n完成：成功 {len(results)}，失败 {len(errors)}，总耗时 {total_elapsed:.1f}s\n")

    # 构建评分 DataFrame
    score_df = pd.DataFrame([
        {"代码": r["code"], "评分日期": r["date"], "时序评分": r["score"]}
        for r in results
    ])

    # 合并神奇公式排名
    merged = magic_df.merge(score_df, on="代码", how="left")

    # 按时序评分降序排列，新增时序排名列
    merged_scored = merged.dropna(subset=["时序评分"]).copy()
    merged_scored = merged_scored.sort_values("时序评分", ascending=False)
    merged_scored.insert(0, "时序排名", range(1, len(merged_scored) + 1))

    # 失败的股票追加到末尾
    failed_codes = {e["code"] for e in errors}
    merged_failed = merged[merged["代码"].isin(failed_codes)].copy()
    merged_failed.insert(0, "时序排名", None)

    output_df = pd.concat([merged_scored, merged_failed], ignore_index=True)

    # 保存
    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"结果已写入：{OUTPUT_CSV}")
    print(f"  有效评分行：{len(merged_scored)}")
    print(f"  失败行：{len(merged_failed)}")

    # 简单断言：至少 80% 的股票评分成功
    success_rate = len(results) / len(codes)
    assert success_rate >= 0.8, f"成功率 {success_rate:.1%} 低于 80%，请检查数据获取"
    print(f"\n✓ 成功率 {success_rate:.1%}，验证通过")

    # 打印前 20 名预览
    print("\n── 时序评分 Top 20 ──")
    preview_cols = ["时序排名", "代码", "上市公司", "复合排名", "时序评分", "评分日期"]
    available = [c for c in preview_cols if c in output_df.columns]
    print(output_df[available].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
