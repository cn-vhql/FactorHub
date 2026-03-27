#!/usr/bin/env python3
"""
Skill: portfolio_optimize - 组合权重优化

使用示例:
  python portfolio_optimize.py 000001 "RSI(close,14)" "close/SMA(close,20)" "momentum_20" 2023-01-01 2024-12-31 --method max_sharpe
  python portfolio_optimize.py 600519 "factor1" "factor2" --method min_volatility
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
        "portfolio_optimize",
        "组合权重优化\n\n" +
        "示例:\n" +
        "  python portfolio_optimize.py 000001 'RSI(close,14)' 'close/SMA(close,20)' 'momentum_20' 2023-01-01 2024-12-31 --method max_sharpe\n" +
        "  python portfolio_optimize.py 600519 'factor1' 'factor2' --method min_volatility"
    )

    parser.add_argument("stock_code", help="股票代码")
    parser.add_argument("factors", nargs="+", help="因子表达式列表")
    parser.add_argument("start_date", nargs="?", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("end_date", nargs="?", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--start", dest="start_override", help="开始日期 (覆盖位置参数)")
    parser.add_argument("--end", dest="end_override", help="结束日期 (覆盖位置参数)")
    parser.add_argument("--method", choices=["max_sharpe", "min_volatility", "max_return"],
                        default="max_sharpe", help="优化方法 (默认: max_sharpe)")
    parser.add_argument("--api-url", default="http://localhost:8000/api", help="API基地址")

    args = parser.parse_args()

    # 提取因子和日期
    if len(args.factors) >= 2 and args.factors[-1].count("-") == 2:  # 最后一个是日期
        end_date_arg = args.factors[-1]
        factors = args.factors[:-1]
        start_date_arg = args.start_date
    else:
        factors = args.factors
        start_date_arg = args.start_override or args.start_date
        end_date_arg = args.end_override or args.end_date

    try:
        start_date, end_date = parse_date_range(start_date_arg, end_date_arg)
    except ValueError as e:
        print(SkillOutput.error(str(e)))
        return 1

    client = SkillsClient(args.api_url)

    print(f"\n📊 组合优化: {args.stock_code}")
    print(f"   因子个数: {len(factors)}")
    print(f"   优化方法: {args.method}")
    print(f"   时间范围: {start_date} 至 {end_date}")

    try:
        # 调用优化API
        print(f"\n⏳ 执行组合优化...")
        opt_result = client.post(
            "/portfolio/optimize-weights",
            data={
                "stock_code": args.stock_code,
                "factors": factors,
                "method": args.method,
                "start_date": start_date,
                "end_date": end_date
            }
        )

        if not opt_result.get("success"):
            print(SkillOutput.error(opt_result.get("detail", "优化失败")))
            return 1

        print("✅ 优化完成\n")

        # 显示结果
        print(SkillOutput.section("组合优化结果"))

        opt_data = opt_result.get("data", {})

        # 因子权重
        print(SkillOutput.subsection("因子权重配置"))

        weights = opt_data.get("weights", {})
        weight_rows = []

        for factor, weight in weights.items():
            weight_pct = weight * 100
            bar_length = int(weight_pct / 5)
            bar = "📊" * bar_length
            weight_rows.append([
                factor[:25] + "..." if len(factor) > 25 else factor,
                f"{weight_pct:.1f}%",
                bar
            ])

        headers = ["因子", "权重", "可视化"]
        print(SkillOutput.table(headers, weight_rows))

        # 预期性能
        print(SkillOutput.subsection("\n预期性能"))

        metrics = opt_data.get("expected_metrics", {})
        annual_return = metrics.get("annual_return", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        volatility = metrics.get("volatility", 0)
        max_dd = metrics.get("max_drawdown", 0)

        emoji_return = "📈" if annual_return > 0.1 else "⚠️"
        emoji_sharpe = "✅" if sharpe > 0.8 else "⚠️"

        print(SkillOutput.metric("年化收益", format_percentage(annual_return), "", emoji_return))
        print(SkillOutput.metric("夏普比率", sharpe, "", emoji_sharpe))
        print(SkillOutput.metric("年化波动", format_percentage(volatility)))
        print(SkillOutput.metric("最大回撤", format_percentage(max_dd)))

        # 风险指标
        print(SkillOutput.subsection("\n风险指标"))
        print(SkillOutput.metric("信息比率", metrics.get("information_ratio", 0)))
        print(SkillOutput.metric("索提诺比", metrics.get("sortino_ratio", 0)))
        print(SkillOutput.metric("最大期间回撤", format_percentage(metrics.get("max_drawdown", 0))))

        # 对标对比
        print(SkillOutput.subsection("\n对标分析"))
        benchmarks = opt_data.get("benchmark_comparison", {})

        if benchmarks:
            bench_rows = []
            for bench_name, bench_metrics in benchmarks.items():
                bench_rows.append([
                    bench_name,
                    format_percentage(bench_metrics.get("annual_return", 0)),
                    f"{bench_metrics.get('sharpe_ratio', 0):.2f}",
                    format_percentage(bench_metrics.get("max_drawdown", 0))
                ])

            bench_headers = ["对标", "年化收益", "夏普比", "最大回撤"]
            print(SkillOutput.table(bench_headers, bench_rows))

        # 建议
        print(SkillOutput.subsection("\n投资建议"))
        if sharpe > 0.8 and annual_return > 0.1:
            print("✅ 优化配置表现优秀，强烈推荐使用")
        elif sharpe > 0.5 and annual_return > 0.05:
            print("⚠️ 优化配置可行，可考虑与其他策略组合")
        else:
            print("❌ 优化结果不理想，建议调整参数或更换因子")

        print("\n✅ 组合优化完成\n")
        return 0

    except Exception as e:
        print(SkillOutput.error(str(e)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
