#!/usr/bin/env python3
"""
Skill: validate_formula - 验证因子公式的有效性

使用示例:
  python validate_formula.py "(close - SMA(close, 20)) / SMA(close, 20)" 000001 2023-01-01 2024-12-31
  python validate_formula.py "RSI(close, 14)" 000001
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
        "validate_formula",
        "验证因子公式的有效性\n\n" +
        "示例:\n" +
        "  python validate_formula.py '(close - SMA(close, 20)) / SMA(close, 20)' 000001 2023-01-01 2024-12-31\n" +
        "  python validate_formula.py 'RSI(close, 14)' 000001"
    )

    parser.add_argument("formula", help="因子公式表达式")
    parser.add_argument("stock_code", nargs="?", default="000001", help="测试股票代码 (默认: 000001)")
    parser.add_argument("start_date", nargs="?", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("end_date", nargs="?", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--start", dest="start_override", help="开始日期 (覆盖位置参数)")
    parser.add_argument("--end", dest="end_override", help="结束日期 (覆盖位置参数)")
    parser.add_argument("--api-url", default="http://localhost:8000/api", help="API基地址")

    args = parser.parse_args()

    # 确定日期
    start = args.start_override or args.start_date or "2023-01-01"
    end = args.end_override or args.end_date

    try:
        start_date, end_date = parse_date_range(start, end)
    except ValueError as e:
        print(SkillOutput.error(str(e)))
        return 1

    client = SkillsClient(args.api_url)

    print(f"\n✓ 公式验证: {args.formula}")
    print(f"  测试股票: {args.stock_code}")
    print(f"  时间范围: {start_date} 至 {end_date}")

    try:
        # 调用API验证公式
        result = client.post(
            "/factors/validate",
            data={
                "formula": args.formula,
                "stock_code": args.stock_code,
                "start_date": start_date,
                "end_date": end_date
            }
        )

        if not result.get("success"):
            detail = result.get("detail", "验证失败")
            print(SkillOutput.error(detail))
            return 1

        validation = result.get("data", {})

        # 显示验证结果
        print(SkillOutput.section("验证结果"))

        # 语法检查
        print(SkillOutput.subsection("语法检查"))
        syntax_valid = validation.get("syntax_valid", False)
        status = "✅ 通过" if syntax_valid else "❌ 失败"
        print(f"├─ 结果: {status}")
        if validation.get("syntax_error"):
            print(f"├─ 错误: {validation['syntax_error']}")

        # 函数库检查
        print(SkillOutput.subsection("\n函数库检查"))
        functions = validation.get("functions", {})
        all_valid = True
        for func_name, func_info in functions.items():
            is_valid = func_info.get("available", False)
            status = "✅" if is_valid else "❌"
            source = func_info.get("source", "unknown")
            print(f"├─ {func_name}(): {status} ({source})")
            if not is_valid:
                all_valid = False

        # 执行检查
        print(SkillOutput.subsection("\n执行检查"))
        execution = validation.get("execution", {})
        exec_success = execution.get("success", False)
        status = "✅ 成功" if exec_success else "❌ 失败"
        print(f"├─ 样本执行: {status}")

        if exec_success:
            values = execution.get("values", [])
            print(f"├─ 数据点数: {len(values)}")
            if values:
                print(f"├─ 值范围: [{min(values):.4f}, {max(values):.4f}]")
                null_count = sum(1 for v in values if v is None or (isinstance(v, float) and str(v) == 'nan'))
                if null_count > 0:
                    print(f"├─ 缺失值: {null_count}行 ({100*null_count/len(values):.1f}%) ⚠️")
        else:
            error = execution.get("error", "未知错误")
            print(f"├─ 错误: {error}")

        # 统计特征
        if exec_success and "statistics" in validation:
            print(SkillOutput.subsection("\n统计特征"))
            stats = validation["statistics"]
            print(SkillOutput.metric("平均值", stats.get("mean", 0)))
            print(SkillOutput.metric("标准差", stats.get("std", 0)))
            print(SkillOutput.metric("偏度", stats.get("skewness", 0)))
            print(SkillOutput.metric("峰度", stats.get("kurtosis", 0)))

        # 整体评估
        print(SkillOutput.subsection("\n整体评估"))
        if syntax_valid and all_valid and exec_success:
            print("✅ 公式有效，建议用于分析\n")
            return 0
        elif syntax_valid and exec_success:
            print("⚠️ 公式可执行但包含未知函数，使用前请核实\n")
            return 0
        else:
            print("❌ 公式存在问题，请修正后重试\n")
            return 1

    except Exception as e:
        print(SkillOutput.error(str(e)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
