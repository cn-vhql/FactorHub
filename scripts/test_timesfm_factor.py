#!/usr/bin/env python3
"""
快速验证 TimesFM 因子集成是否正常。

前提：TimesFM 因子依赖 autoresearch 的环境（torch 等），有两种运行方式：

  方式一（推荐）：使用 autoresearch 的 venv
    cd /path/to/autoresearch
    uv run python ../FactorHub/scripts/test_timesfm_factor.py --symbol 000001

  方式二：在 FactorHub 环境中先安装 timesfm 依赖
    cd FactorHub
    uv pip install ".[timesfm]"
    uv run python scripts/test_timesfm_factor.py --symbol 000001
"""

import argparse
import sys
from pathlib import Path

# 确保 FactorHub 根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="测试 TimesFM 因子")
    parser.add_argument("--symbol", default="000001", help="股票代码")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5])
    parser.add_argument("--fast", action="store_true", help="使用快速模式（每5日预测一次）")
    args = parser.parse_args()

    # 1. 检查 TimesFM 是否可用
    from backend.services.timesfm_factor_service import timesfm_factor_service
    if not timesfm_factor_service.is_available():
        print("❌ TimesFM 不可用，请检查：")
        print("   1. autoresearch 目录是否存在于 FactorHub 同级目录")
        print("   2. autoresearch/local_timesfm_model 是否存在")
        sys.exit(1)
    print("✅ TimesFM 可用")

    # 2. 获取股票数据
    print(f"正在获取 {args.symbol} 数据 ({args.start} ~ {args.end})...")
    from backend.services.data_service import data_service
    df = data_service.get_stock_data(args.symbol, args.start, args.end)
    print(f"✅ 获取到 {len(df)} 条数据")

    # 3. 计算因子
    print(f"正在计算 TimesFM 因子 (horizon={args.horizon}, fast={args.fast})...")
    if args.fast:
        factor = timesfm_factor_service.compute_fast(df, horizon=args.horizon, context_len=60, step=5)
    else:
        factor = timesfm_factor_service.compute(df, horizon=args.horizon, context_len=60)

    valid = factor.dropna()
    print(f"✅ 因子计算完成，有效值 {len(valid)}/{len(factor)} 个")
    print(f"   均值: {valid.mean():.4f}")
    print(f"   标准差: {valid.std():.4f}")
    print(f"   最小值: {valid.min():.4f}")
    print(f"   最大值: {valid.max():.4f}")
    print("\n最近5个因子值：")
    print(valid.tail())


if __name__ == "__main__":
    main()
