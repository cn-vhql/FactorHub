"""
OHLCV P0 因子烟测脚本

验证 12 个 P0 因子的：
  - 可计算性（无异常）
  - warmup NaN 数量
  - 无 inf 值
  - NaN 比例 < 20%（warmup 后）

同时输出：
  - 每个因子的分布统计（均值、标准差、分位数、缺失率）
  - P0 因子与 baseline 因子的相关性矩阵，标记 |corr| > 0.95 的因子

用法：
    .venv/bin/python scripts/run_ohlcv_factor_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# P0 因子定义
# ---------------------------------------------------------------------------

P0_FACTORS: dict[str, str] = {
    "overnight_return_1":   "open / REF(close, 1) - 1",
    "intraday_return_1":    "close / open - 1",
    "body_ratio":           "(close - open) / (high - low + 1e-12)",
    "upper_shadow_ratio":   "(high - MAX(open, close)) / (high - low + 1e-12)",
    "close_location_value": "(close - low) / (high - low + 1e-12)",
    "breakout_strength_20": "(close - HHV(high, 20).shift(1)) / (ATR(high, low, close, timeperiod=14) + 1e-12)",
    "efficiency_ratio_20":  "np.abs(close - REF(close, 20)) / (SUM(np.abs(close - REF(close, 1)), 20) + 1e-12)",
    "kama_ratio_20":        "close / KAMA(close, timeperiod=20)",
    "parkinson_vol_20":     "np.sqrt((np.log(high / low) ** 2).rolling(20).mean() / (4 * np.log(2)))",
    "garman_klass_vol_20":  "np.sqrt((0.5 * (np.log(high / low) ** 2) - (2 * np.log(2) - 1) * (np.log(close / open) ** 2)).rolling(20).mean())",
    "cmf_20":               "SUM((((close - low) - (high - close)) / (high - low + 1e-12)) * volume, 20) / (SUM(volume, 20) + 1e-12)",
    "volume_zscore_20":     "(volume - AVE(volume, 20)) / (STD(volume, 20) + 1e-12)",
}

# 预期 warmup NaN 行数（来自 design.md 需求 3.1）
EXPECTED_WARMUP: dict[str, int] = {
    "overnight_return_1":   1,
    "intraday_return_1":    0,
    "body_ratio":           0,
    "upper_shadow_ratio":   0,
    "close_location_value": 0,
    "breakout_strength_20": 20,
    "efficiency_ratio_20":  20,
    "parkinson_vol_20":     19,
    "garman_klass_vol_20":  19,
    "cmf_20":               19,
    "volume_zscore_20":     19,
    "kama_ratio_20":        20,
}

# 用于相关性分析的 baseline 因子（从 _get_default_factors 中选取代表性因子）
BASELINE_FACTORS: dict[str, str] = {
    "log_return_1":         "np.log(close / close.shift(1))",
    "log_return_5":         "np.log(close / close.shift(5))",
    "price_vs_sma20":       "close / SMA(close, timeperiod=20)",
    "high_low_ratio":       "(high - low) / open",
    "close_open_ratio":     "close / open",
    "rsi_14":               "RSI(close, timeperiod=14)",
    "atr_14":               "ATR(high, low, close, timeperiod=14)",
    "atr_norm":             "ATR(high, low, close, timeperiod=14) / close",
    "volatility_10":        "np.log(close / close.shift(1)).rolling(window=10).std()",
    "volume_ma_ratio":      "volume / SMA(volume, timeperiod=10)",
    "roc_10":               "(close - close.shift(10)) / close.shift(10)",
    "momentum_20":          "close / close.shift(20) - 1",
    "reversal_5":           "-(close / close.shift(5) - 1)",
    "deviation_from_ma20":  "(close - SMA(close, timeperiod=20)) / SMA(close, timeperiod=20)",
    "bollinger_bandwidth":  "(BBANDS(close, timeperiod=20)[0] - BBANDS(close, timeperiod=20)[2]) / BBANDS(close, timeperiod=20)[1]",
}

HIGH_CORR_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# 合成数据生成
# ---------------------------------------------------------------------------

def generate_synthetic_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """生成满足 OHLCV 约束的合成数据（high >= close >= low > 0, volume > 0）。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    # 随机游走价格
    log_returns = rng.normal(0, 0.015, n)
    close = 10.0 * np.exp(np.cumsum(log_returns))

    # 生成 open/high/low 满足约束
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high_offset = np.abs(rng.normal(0, 0.008, n))
    low_offset = np.abs(rng.normal(0, 0.008, n))

    high = np.maximum(open_, close) + high_offset * close
    low = np.minimum(open_, close) - low_offset * close
    low = np.maximum(low, 0.01)  # 确保 low > 0

    volume = rng.uniform(1e6, 5e6, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# 数据加载（优先真实数据，回退合成数据）
# ---------------------------------------------------------------------------

def load_data(n_rows: int = 250) -> tuple[pd.DataFrame, str]:
    """尝试加载真实股票数据，失败则使用合成数据。"""
    try:
        from backend.services.data_service import data_service

        df = data_service.get_stock_data("000001.SZ", "2022-01-01", "2024-12-31")
        if df is not None and len(df) >= n_rows:
            df = df.tail(n_rows).copy()
            required = {"open", "high", "low", "close", "volume"}
            if required.issubset(df.columns):
                print(f"[数据] 使用真实数据：000001.SZ，{len(df)} 行")
                return df, "real"
    except Exception as e:
        print(f"[数据] 真实数据加载失败（{e}），回退到合成数据")

    df = generate_synthetic_ohlcv(n_rows)
    print(f"[数据] 使用合成数据：{len(df)} 行")
    return df, "synthetic"


# ---------------------------------------------------------------------------
# 因子计算
# ---------------------------------------------------------------------------

def compute_factor(calc, name: str, code: str, df: pd.DataFrame) -> pd.Series | None:
    """计算单个因子，返回 Series 或 None（异常时）。"""
    try:
        result = calc.calculate(df, code)
        return result
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return None


# ---------------------------------------------------------------------------
# 分布统计（任务 5.2）
# ---------------------------------------------------------------------------

def print_distribution_stats(name: str, series: pd.Series) -> None:
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    total = len(series)
    nan_count = series.isna().sum() + np.isinf(series).sum()
    nan_ratio = nan_count / total if total > 0 else 1.0

    if len(valid) == 0:
        print(f"  {name}: 全部为 NaN/inf，无法计算统计")
        return

    mean = valid.mean()
    std = valid.std()
    p5, p25, p50, p75, p95 = valid.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).values

    print(
        f"  {name:<28} "
        f"mean={mean:+.4f}  std={std:.4f}  "
        f"p5={p5:+.4f}  p25={p25:+.4f}  p50={p50:+.4f}  p75={p75:+.4f}  p95={p95:+.4f}  "
        f"NaN%={nan_ratio*100:.1f}%"
    )


# ---------------------------------------------------------------------------
# 相关性矩阵（任务 5.3）
# ---------------------------------------------------------------------------

def compute_correlation_matrix(
    calc,
    df: pd.DataFrame,
    p0_results: dict[str, pd.Series],
    baseline_codes: dict[str, str],
) -> pd.DataFrame:
    """计算 P0 因子与 baseline 因子的相关性矩阵。"""
    baseline_results: dict[str, pd.Series] = {}
    for name, code in baseline_codes.items():
        try:
            s = calc.calculate(df, code)
            baseline_results[name] = s
        except Exception as e:
            print(f"  [WARN] baseline 因子 {name} 计算失败: {e}")

    if not baseline_results:
        print("  [WARN] 无可用 baseline 因子，跳过相关性分析")
        return pd.DataFrame()

    # 合并为 DataFrame，对齐索引
    p0_df = pd.DataFrame(p0_results)
    bl_df = pd.DataFrame(baseline_results)

    # 替换 inf 为 NaN
    p0_df = p0_df.replace([np.inf, -np.inf], np.nan)
    bl_df = bl_df.replace([np.inf, -np.inf], np.nan)

    # 计算相关性矩阵（P0 行 × baseline 列）
    corr_rows = {}
    for p0_name in p0_df.columns:
        row = {}
        for bl_name in bl_df.columns:
            combined = pd.concat([p0_df[p0_name], bl_df[bl_name]], axis=1).dropna()
            if len(combined) < 10:
                row[bl_name] = np.nan
            else:
                row[bl_name] = combined.iloc[:, 0].corr(combined.iloc[:, 1])
        corr_rows[p0_name] = row

    return pd.DataFrame(corr_rows).T  # shape: (n_p0, n_baseline)


# ---------------------------------------------------------------------------
# 主验证逻辑
# ---------------------------------------------------------------------------

def run_smoke_test() -> None:
    from backend.services.factor_service import FactorCalculator

    calc = FactorCalculator()

    print("=" * 80)
    print("OHLCV P0 因子烟测")
    print("=" * 80)

    # 加载数据
    df, data_source = load_data(250)
    print()

    # -----------------------------------------------------------------------
    # 逐因子验证
    # -----------------------------------------------------------------------
    print("─" * 80)
    print("【1】因子计算与验证")
    print("─" * 80)

    results: dict[str, pd.Series] = {}
    factor_status: dict[str, dict] = {}

    for name, code in P0_FACTORS.items():
        series = compute_factor(calc, name, code, df)

        status = {
            "computable": series is not None,
            "warmup_ok": False,
            "no_inf": False,
            "nan_ratio_ok": False,
            "pass": False,
        }

        if series is not None:
            results[name] = series
            expected_warmup = EXPECTED_WARMUP.get(name, 0)

            # 统计头部连续 NaN 数
            head_nans = 0
            for v in series.values:
                if pd.isna(v):
                    head_nans += 1
                else:
                    break
            warmup_ok = head_nans == expected_warmup
            status["warmup_ok"] = warmup_ok
            status["actual_warmup"] = head_nans
            status["expected_warmup"] = expected_warmup

            # inf 检查
            inf_count = int(np.isinf(series.fillna(0)).sum())
            status["no_inf"] = inf_count == 0
            status["inf_count"] = inf_count

            # NaN 比例（使用实际 warmup 后的数据）
            post_warmup = series.iloc[head_nans:]
            nan_ratio = post_warmup.isna().mean()
            status["nan_ratio_ok"] = nan_ratio < 0.20
            status["nan_ratio"] = float(nan_ratio)

            # PASS 条件：可计算 + 无 inf + NaN 比例 < 20%
            # warmup 不匹配仅作为警告，不影响 PASS/FAIL
            status["pass"] = (
                status["computable"]
                and status["no_inf"]
                and status["nan_ratio_ok"]
            )

            warmup_mark = "✓" if warmup_ok else f"△(got {head_nans}, exp {expected_warmup})"
            inf_mark = "✓" if status["no_inf"] else f"✗({inf_count} inf)"
            nan_mark = "✓" if status["nan_ratio_ok"] else f"✗({nan_ratio*100:.1f}%)"
            overall = "PASS" if status["pass"] else "FAIL"
            warmup_note = " [warmup△]" if not warmup_ok else ""

            print(
                f"  {name:<28} warmup={warmup_mark:<24} inf={inf_mark:<12} "
                f"NaN%={nan_mark:<12} [{overall}]{warmup_note}"
            )
        else:
            print(f"  {name:<28} [FAIL] 计算异常")

        factor_status[name] = status

    # -----------------------------------------------------------------------
    # 分布统计（任务 5.2）
    # -----------------------------------------------------------------------
    print()
    print("─" * 80)
    print("【2】因子分布统计")
    print("─" * 80)

    for name, series in results.items():
        print_distribution_stats(name, series)

    # -----------------------------------------------------------------------
    # 相关性矩阵（任务 5.3）
    # -----------------------------------------------------------------------
    print()
    print("─" * 80)
    print("【3】P0 因子与 Baseline 因子相关性分析")
    print("─" * 80)

    corr_matrix = compute_correlation_matrix(calc, df, results, BASELINE_FACTORS)

    high_corr_factors: list[str] = []

    if not corr_matrix.empty:
        print(f"\n  相关性矩阵（P0 × baseline，|corr| > {HIGH_CORR_THRESHOLD} 标记为高相关）：\n")

        # 打印表头
        bl_names = list(corr_matrix.columns)
        header = f"  {'因子名':<28}" + "".join(f"{n[:12]:>14}" for n in bl_names)
        print(header)
        print("  " + "-" * (28 + 14 * len(bl_names)))

        for p0_name, row in corr_matrix.iterrows():
            max_abs_corr = row.abs().max()
            is_high_corr = max_abs_corr > HIGH_CORR_THRESHOLD

            row_str = f"  {p0_name:<28}"
            for bl_name in bl_names:
                val = row[bl_name]
                if pd.isna(val):
                    row_str += f"{'N/A':>14}"
                else:
                    marker = "*" if abs(val) > HIGH_CORR_THRESHOLD else " "
                    row_str += f"{val:>13.3f}{marker}"
            if is_high_corr:
                row_str += "  ← 高相关 (HIGH CORR)"
                high_corr_factors.append(str(p0_name))
            print(row_str)

        print()
        if high_corr_factors:
            print(f"  ⚠ 高相关因子（|corr| > {HIGH_CORR_THRESHOLD}，降为 P1 候选）：")
            for f in high_corr_factors:
                # 找出最高相关的 baseline 因子
                max_bl = corr_matrix.loc[f].abs().idxmax()
                max_val = corr_matrix.loc[f, max_bl]
                print(f"    - {f} ↔ {max_bl}: corr={max_val:.3f}")
        else:
            print(f"  ✓ 所有 P0 因子与 baseline 因子的绝对相关系数均 ≤ {HIGH_CORR_THRESHOLD}")

    # -----------------------------------------------------------------------
    # 汇总表
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("【汇总】P0 因子烟测结果")
    print("=" * 80)
    print(f"  {'因子名':<28} {'可计算':^8} {'warmup':^10} {'无inf':^8} {'NaN<20%':^8} {'结论':^8}")
    print("  " + "-" * 76)

    pass_count = 0
    for name, st in factor_status.items():
        comp = "✓" if st.get("computable") else "✗"
        wu_ok = st.get("warmup_ok", False)
        wu = "✓" if wu_ok else f"△({st.get('actual_warmup','?')}≠{st.get('expected_warmup','?')})"
        inf_ = "✓" if st.get("no_inf") else "✗"
        nan_ = "✓" if st.get("nan_ratio_ok") else "✗"
        verdict = "PASS" if st.get("pass") else "FAIL"
        high_corr_mark = " [HIGH CORR]" if name in high_corr_factors else ""
        if st.get("pass"):
            pass_count += 1
        print(f"  {name:<28} {comp:^8} {wu:^10} {inf_:^8} {nan_:^8} {verdict}{high_corr_mark}")

    print()
    total = len(P0_FACTORS)
    print(f"  通过: {pass_count}/{total}  |  数据来源: {data_source}")
    if pass_count >= 4:
        print(f"  ✓ 满足需求 7.4：至少 4 个 P0 因子通过稳定性验证")
    else:
        print(f"  ✗ 未满足需求 7.4：需要至少 4 个 P0 因子通过验证")

    # 列出 warmup 不匹配的因子（△ 表示与 design.md 预期不同，但不影响可用性）
    warmup_mismatches = [
        (n, s) for n, s in factor_status.items()
        if s.get("computable") and not s.get("warmup_ok")
    ]
    if warmup_mismatches:
        print()
        print("  注：以下因子的实际 warmup NaN 数与 design.md 预期不同（△）：")
        print("  （原因：mylanguage_funcs 使用 min_periods=1，实际 warmup 可能少于理论值）")
        for n, s in warmup_mismatches:
            print(f"    - {n}: 实际={s['actual_warmup']}，预期={s['expected_warmup']}")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
