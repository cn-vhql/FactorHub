#!/usr/bin/env python3
"""
Skill: factor_analyze - 因子分析 (IC/IR/稳定性)

使用示例:
  python factor_analyze.py 000001 "RSI(close, 14)" 2023-01-01 2024-12-31
  python factor_analyze.py 000001 "close/SMA(close, 20)" --horizons 1,5,10
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.skills.skills_base import (
    SkillsClient, SkillOutput, create_parser, parse_date_range
)


def main():
    parser = create_parser(
        "factor_analyze",
        "因子分析 - 计算IC/IR和稳定性\n\n" +
        "示例:\n" +
        "  python factor_analyze.py 000001 'RSI(close, 14)' 2023-01-01 2024-12-31\n" +
        "  python factor_analyze.py 000001 'close/SMA(close, 20)' --horizons 1,5,10"
    )

    parser.add_argument("stock_code", help="股票代码")
    parser.add_argument("expression", help="因子表达式")
    parser.add_argument("start_date", nargs="?", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("end_date", nargs="?", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--start", dest="start_override", help="开始日期 (覆盖位置参数)")
    parser.add_argument("--end", dest="end_override", help="结束日期 (覆盖位置参数)")
    parser.add_argument("--horizons", default="1", help="分析周期 (默认: 1, 示例: 1,5,10)")
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

    print(f"\n📊 因子分析: {args.stock_code} - {args.expression}")
    print(f"   时间范围: {start_date} 至 {end_date}")

    try:
        # Step 1: 计算因子
        print(f"\n⏳ 步骤1: 计算因子值...")
        calc_result = client.post(
            "/analysis/calculate",
            data={
                "factors": [args.expression],
                "stock_codes": [args.stock_code],
                "start_date": start_date,
                "end_date": end_date
            }
        )

        if not calc_result.get("success"):
            print(SkillOutput.error(calc_result.get("detail", "计算因子失败")))
            return 1

        print("✅ 因子计算完成")

        # Step 2: IC/IR分析
        print(f"⏳ 步骤2: 计算IC/IR...")
        ic_result = client.post(
            "/analysis/ic",
            data={
                "factors": [args.expression],
                "stock_codes": [args.stock_code],
                "start_date": start_date,
                "end_date": end_date,
                "horizons": [int(h.strip()) for h in args.horizons.split(",")]
            }
        )

        if not ic_result.get("success"):
            print(SkillOutput.error(ic_result.get("detail", "IC分析失败")))
            return 1

        print("✅ IC/IR计算完成")

        # Step 3: 稳定性分析
        print(f"⏳ 步骤3: 计算稳定性...")
        stability_result = client.post(
            "/analysis/stability",
            data={
                "factors": [args.expression],
                "stock_codes": [args.stock_code],
                "start_date": start_date,
                "end_date": end_date
            }
        )

        if not stability_result.get("success"):
            print(SkillOutput.error(stability_result.get("detail", "稳定性分析失败")))
            return 1

        print("✅ 稳定性计算完成")

        # 显示结果
        print(SkillOutput.section("因子分析报告"))

        # 基础信息
        print(SkillOutput.subsection("基础信息"))
        calc_data = calc_result.get("data", {})
        print(f"├─ 股票代码: {args.stock_code}")
        print(f"├─ 因子表达式: {args.expression}")
        print(f"├─ 分析周期: {start_date} 至 {end_date}")
        print(f"├─ 数据点数: {calc_data.get('data_points', 0)}")

        # IC/IR分析
        print(SkillOutput.subsection("\nIC/IR分析"))
        ic_data = ic_result.get("data", {})

        ic_mean = ic_data.get("ic_mean", 0)
        ir_ratio = ic_data.get("ir_ratio", 0)

        emoji_ic = "✅" if ic_mean > 0.03 else "⚠️" if ic_mean > 0 else "❌"
        emoji_ir = "✅" if ir_ratio > 0.5 else "⚠️" if ir_ratio > 0.3 else "❌"

        print(SkillOutput.metric("IC均值", ic_mean, "", emoji_ic))
        print(SkillOutput.metric("IC标准差", ic_data.get("ic_std", 0)))
        print(SkillOutput.metric("IR比率", ir_ratio, "", emoji_ir))
        print(SkillOutput.metric("IC正比例", f"{ic_data.get('positive_rate', 0)*100:.1f}%"))

        # 稳定性分析
        print(SkillOutput.subsection("\n稳定性分析"))
        stability_data = stability_result.get("data", {})

        score = stability_data.get("stability_score", 0)
        grade = stability_data.get("stability_grade", "C")
        grade_emoji = {"A": "⭐⭐⭐", "B": "⭐⭐", "C": "⭐", "D": "○"}.get(grade, "○")

        print(SkillOutput.metric("稳定性评分", score))
        print(SkillOutput.metric("稳定性等级", f"{grade} {grade_emoji}"))
        print(SkillOutput.metric("衰减指数", stability_data.get("decay_index", 0)))

        # 综合评估
        print(SkillOutput.subsection("\n综合评估"))
        if ic_mean > 0.03 and ir_ratio > 0.5 and grade in ["A", "B"]:
            print("✅ 因子质量优秀，强烈推荐使用")
        elif ic_mean > 0.01 and grade in ["B", "C"]:
            print("⚠️ 因子有效，建议与其他因子组合使用")
        else:
            print("❌ 因子效果不理想，建议优化或更换")

        print("\n✅ 分析完成\n")
        return 0

    except Exception as e:
        print(SkillOutput.error(str(e)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
