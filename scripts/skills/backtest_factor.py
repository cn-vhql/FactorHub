#!/usr/bin/env python3
"""
Skill: backtest_factor - 因子回测

使用示例:
  python backtest_factor.py 000001 "close/SMA(close, 20)" 2023-01-01 2024-12-31 --percentile 50
  python backtest_factor.py 600519 "RSI(close, 14)" --direction long --api-url http://localhost:8000/api
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
        "backtest_factor",
        "因子回测\n\n" +
        "示例:\n" +
        "  python backtest_factor.py 000001 'close/SMA(close, 20)' 2023-01-01 2024-12-31 --percentile 50\n" +
        "  python backtest_factor.py 600519 'RSI(close, 14)' --direction long"
    )

    parser.add_argument("stock_code", help="股票代码")
    parser.add_argument("expression", help="因子表达式")
    parser.add_argument("start_date", nargs="?", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("end_date", nargs="?", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--start", dest="start_override", help="开始日期 (覆盖位置参数)")
    parser.add_argument("--end", dest="end_override", help="结束日期 (覆盖位置参数)")
    parser.add_argument("--percentile", type=int, default=50, help="分位数 (默认: 50)")
    parser.add_argument("--direction", choices=["long", "short"], default="long", help="交易方向 (默认: long)")
    parser.add_argument("--api-url", default="http://localhost:8000/api", help="API基地址")

    args = parser.parse_args()

    start = args.start_override or args.start_date
    end = args.end_override or args.end_date

    try:
        start_date, end_date = parse_date_range(start, end)
    except ValueError as e:
        print(SkillOutput.error(str(e)))
        return 1

    client = SkillsClient(args.api_url)

    print(f"\n🔄 回测因子: {args.stock_code} - {args.expression}")
    print(f"   时间范围: {start_date} 至 {end_date}")
    print(f"   交易规则: {args.direction.upper()} @ 分位数{args.percentile}")

    try:
        # 调用回测API
        print(f"\n⏳ 执行回测...")
        backtest_result = client.post(
            "/backtest/single",
            data={
                "stock_codes": [args.stock_code],
                "factors": [args.expression],
                "strategy_type": "single_factor",
                "start_date": start_date,
                "end_date": end_date,
                "percentile": args.percentile,
                "direction": args.direction
            }
        )

        if not backtest_result.get("success"):
            print(SkillOutput.error(backtest_result.get("detail", "回测失败")))
            return 1

        print("✅ 回测完成\n")

        # 显示结果
        print(SkillOutput.section("回测结果"))

        backtest_data = backtest_result.get("data", {})

        # 基础信息
        print(SkillOutput.subsection("基础信息"))
        print(f"├─ 股票代码: {args.stock_code}")
        print(f"├─ 因子表达式: {args.expression}")
        print(f"├─ 回测期间: {start_date} 至 {end_date}")
        print(f"├─ 交易规则: {args.expression} > {args.percentile}分位数 时持{args.direction.upper()}头")

        # 性能指标
        print(SkillOutput.subsection("\n性能指标"))

        metrics = backtest_data.get("metrics", {})

        total_return = metrics.get("total_return", 0)
        annual_return = metrics.get("annual_return", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown", 0)
        win_rate = metrics.get("win_rate", 0)

        emoji_return = "📈" if total_return > 0 else "📉"
        emoji_sharpe = "✅" if sharpe > 0.8 else "⚠️" if sharpe > 0.5 else "❌"

        print(SkillOutput.metric("总收益", format_percentage(total_return), "", emoji_return))
        print(SkillOutput.metric("年化收益", format_percentage(annual_return)))
        print(SkillOutput.metric("夏普比率", sharpe, "", emoji_sharpe))
        print(SkillOutput.metric("最大回撤", format_percentage(max_dd)))
        print(SkillOutput.metric("胜率", f"{win_rate*100:.1f}%"))

        # 交易统计
        print(SkillOutput.subsection("\n交易统计"))

        trades = backtest_data.get("trade_statistics", {})
        trade_count = trades.get("total_trades", 0)
        avg_duration = trades.get("avg_holding_days", 0)
        avg_pnl = trades.get("avg_pnl_percent", 0)

        print(SkillOutput.metric("总交易数", trade_count, "笔"))
        print(SkillOutput.metric("平均持仓", f"{avg_duration:.1f}", "天"))
        print(SkillOutput.metric("平均每笔", format_percentage(avg_pnl)))

        # 风险评级
        print(SkillOutput.subsection("\n风险评级"))

        if sharpe > 1.0 and abs(max_dd) < 0.15:
            risk_level = "低 ✅"
        elif sharpe > 0.5 and abs(max_dd) < 0.25:
            risk_level = "中 ⚠️"
        else:
            risk_level = "高 ❌"

        print(f"├─ 风险等级: {risk_level}")
        print(f"├─ 夏普比率: {sharpe:.2f}")
        print(f"└─ 最大回撤: {format_percentage(max_dd)}")

        # 建议
        print(SkillOutput.subsection("\n投资建议"))
        if total_return > 0.15 and sharpe > 0.8:
            print("✅ 策略表现优秀，可考虑实盘应用")
        elif total_return > 0.05 and sharpe > 0.5:
            print("⚠️ 策略可行，建议与其他策略组合")
        else:
            print("❌ 策略效果不理想，建议优化参数或更换因子")

        print("\n✅ 回测分析完成\n")
        return 0

    except Exception as e:
        print(SkillOutput.error(str(e)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
