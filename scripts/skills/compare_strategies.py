#!/usr/bin/env python3
"""
Skill: compare_strategies - 对比多个策略

使用示例:
  python compare_strategies.py 000001 "RSI(close,14)" "close/SMA(close,20)" "momentum_20" 2023-01-01 2024-12-31
  python compare_strategies.py 600519 "factor1" "factor2" --start 2023-01-01
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.skills.skills_base import (
    SkillsClient, SkillOutput, create_parser, parse_date_range
)


def format_percentage(value: float) -> str:
    """格式化百分比"""
    if value >= 0:
        return f"+{value*100:.2f}%"
    else:
        return f"{value*100:.2f}%"


def main():
    parser = create_parser(
        "compare_strategies",
        "对比多个策略的性能\n\n" +
        "示例:\n" +
        "  python compare_strategies.py 000001 'RSI(close,14)' 'close/SMA(close,20)' 'momentum_20' 2023-01-01 2024-12-31\n" +
        "  python compare_strategies.py 600519 'factor1' 'factor2' --start 2023-01-01"
    )

    parser.add_argument("stock_code", help="股票代码")
    parser.add_argument("factors", nargs="+", help="因子表达式列表 (至少2个)")
    parser.add_argument("start_date", nargs="?", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("end_date", nargs="?", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--start", dest="start_override", help="开始日期 (覆盖位置参数)")
    parser.add_argument("--end", dest="end_override", help="结束日期 (覆盖位置参数)")
    parser.add_argument("--api-url", default="http://localhost:8000/api", help="API基地址")

    args = parser.parse_args()

    # 检查因子个数
    if len(args.factors) < 2:
        print(SkillOutput.error("至少需要2个因子进行对比"))
        return 1

    # 提取因子列表和日期
    *factors, start_or_end1, start_or_end2 = args.factors if len(args.factors) > 2 else (args.factors + [args.start_date, args.end_date])

    # 重新处理参数
    if len(args.factors) > 2 and args.factors[-1].count("-") == 2:  # 最后一个是日期
        factors = args.factors[:-1]
        start = start_or_end2
        end = args.start_date
    else:
        factors = args.factors
        start = args.start_override or args.start_date
        end = args.end_override or args.end_date

    try:
        start_date, end_date = parse_date_range(start, end)
    except ValueError as e:
        print(SkillOutput.error(str(e)))
        return 1

    client = SkillsClient(args.api_url)

    print(f"\n📊 策略对比: {args.stock_code}")
    print(f"   策略数量: {len(factors)}")
    print(f"   时间范围: {start_date} 至 {end_date}")

    try:
        # 对每个因子进行回测
        print(f"\n⏳ 执行多策略回测...")
        comparison_result = client.post(
            "/backtest/comparison",
            data={
                "stock_code": args.stock_code,
                "factors": factors,
                "start_date": start_date,
                "end_date": end_date
            }
        )

        if not comparison_result.get("success"):
            print(SkillOutput.error(comparison_result.get("detail", "对比失败")))
            return 1

        print("✅ 对比完成\n")

        # 显示结果
        print(SkillOutput.section("策略对比结果"))

        comparison_data = comparison_result.get("data", {})
        results = comparison_data.get("comparison_results", [])

        # 按年化收益排序
        results_sorted = sorted(results, key=lambda x: x.get("metrics", {}).get("annual_return", 0), reverse=True)

        # 准备对比表格
        headers = ["策略因子", "总收益", "年化收益", "夏普比", "最大回撤", "胜率"]
        rows = []

        for i, result in enumerate(results_sorted):
            metrics = result.get("metrics", {})
            factor = result.get("factor", "")
            total_return = metrics.get("total_return", 0)
            annual_return = metrics.get("annual_return", 0)
            sharpe = metrics.get("sharpe_ratio", 0)
            max_dd = metrics.get("max_drawdown", 0)
            win_rate = metrics.get("win_rate", 0)

            # 最佳指标标记
            best_mark = " ⭐" if i == 0 else ""

            rows.append([
                factor[:20] + "..." if len(factor) > 20 else factor,
                format_percentage(total_return),
                format_percentage(annual_return),
                f"{sharpe:.2f}",
                format_percentage(max_dd),
                f"{win_rate*100:.1f}%{best_mark}"
            ])

        print(SkillOutput.table(headers, rows))

        # 最佳策略
        print(SkillOutput.subsection("\n最佳策略"))
        best = results_sorted[0]
        best_metrics = best.get("metrics", {})
        print(f"🏆 {best.get('factor', 'N/A')}")
        print(f"├─ 年化收益: {format_percentage(best_metrics.get('annual_return', 0))}")
        print(f"├─ 夏普比率: {best_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"└─ 最大回撤: {format_percentage(best_metrics.get('max_drawdown', 0))}")

        # 策略特征
        print(SkillOutput.subsection("\n策略特征对比"))
        print(f"├─ 总策略数: {len(results)}")
        print(f"├─ 盈利策略: {sum(1 for r in results if r.get('metrics', {}).get('total_return', 0) > 0)} 个")
        print(f"├─ 平均年化: {format_percentage(sum(r.get('metrics', {}).get('annual_return', 0) for r in results) / len(results))}")

        print("\n✅ 对比分析完成\n")
        return 0

    except Exception as e:
        print(SkillOutput.error(str(e)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
