#!/usr/bin/env python3
"""
Skill: get_stock_data - 获取股票OHLCV数据

使用示例:
  python get_stock_data.py 000001 2024-01-01 2024-03-27
  python get_stock_data.py 600519 --start 2023-01-01 --end 2024-12-31
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.skills.skills_base import (
    SkillsClient, SkillOutput, create_parser, parse_date_range
)


def main():
    parser = create_parser(
        "get_stock_data",
        "获取股票OHLCV数据\n\n" +
        "示例:\n" +
        "  python get_stock_data.py 000001 2024-01-01 2024-03-27\n" +
        "  python get_stock_data.py 600519 --start 2023-01-01 --end 2024-12-31"
    )

    parser.add_argument("stock_code", help="股票代码 (如: 000001, 600519)")
    parser.add_argument("start_date", nargs="?", help="开始日期 (YYYY-MM-DD), 默认2020-01-01")
    parser.add_argument("end_date", nargs="?", help="结束日期 (YYYY-MM-DD), 默认今天")
    parser.add_argument("--start", dest="start_override", help="开始日期 (覆盖位置参数)")
    parser.add_argument("--end", dest="end_override", help="结束日期 (覆盖位置参数)")
    parser.add_argument("--api-url", default="http://localhost:8000/api", help="API基地址")

    args = parser.parse_args()

    # 确定日期
    start = args.start_override or args.start_date
    end = args.end_override or args.end_date

    try:
        start_date, end_date = parse_date_range(start, end)
    except ValueError as e:
        print(SkillOutput.error(str(e)))
        return 1

    client = SkillsClient(args.api_url)

    print(f"\n📊 获取股票数据: {args.stock_code}")
    print(f"   时间范围: {start_date} 至 {end_date}")

    try:
        # 调用API获取数据
        result = client.get(
            f"/data/stock/{args.stock_code}",
            params={
                "start_date": start_date,
                "end_date": end_date
            }
        )

        if not result.get("success"):
            print(SkillOutput.error(result.get("detail", "获取数据失败")))
            return 1

        data = result.get("data", {})
        ohlcv = data.get("ohlcv", [])

        if not ohlcv:
            print(SkillOutput.error("未获得任何数据，请检查股票代码和日期范围"))
            return 1

        # 显示数据表格
        print(SkillOutput.section("数据明细"))

        # 准备表格数据（只显示前20条）
        headers = ["日期", "开盘", "最高", "最低", "收盘", "成交量(万)"]
        rows = []

        for item in ohlcv[-20:]:  # 最新的20条
            rows.append([
                item.get("date", ""),
                f"{item.get('open', 0):.2f}",
                f"{item.get('high', 0):.2f}",
                f"{item.get('low', 0):.2f}",
                f"{item.get('close', 0):.2f}",
                f"{item.get('volume', 0) / 10000:,.0f}" if item.get('volume') else "0"
            ])

        rows.reverse()  # 从早到晚
        print(SkillOutput.table(headers, rows))

        # 显示统计信息
        print(SkillOutput.section("数据统计"))

        closes = [d.get('close', 0) for d in ohlcv]
        volumes = [d.get('volume', 0) for d in ohlcv]

        print(SkillOutput.subsection("基础统计"))
        print(SkillOutput.metric("交易日数", len(ohlcv)))
        print(SkillOutput.metric("价格范围", f"{min(closes):.2f} - {max(closes):.2f}"))
        print(SkillOutput.metric("平均收盘价", sum(closes) / len(closes), "元", ""))
        print(SkillOutput.metric("平均成交量", sum(volumes) / len(volumes), "股", ""))

        # 涨跌统计
        ups = sum(1 for i in range(1, len(ohlcv)) if ohlcv[i]['close'] > ohlcv[i-1]['close'])
        downs = len(ohlcv) - 1 - ups

        print(SkillOutput.subsection("\n涨跌统计"))
        print(SkillOutput.metric("上涨天数", ups, "天", "📈"))
        print(SkillOutput.metric("下跌天数", downs, "天", "📉"))
        print(SkillOutput.metric("涨跌比", f"{ups}/{downs}" if downs > 0 else f"{ups}/0"))

        print("\n✅ 数据获取成功\n")
        return 0

    except Exception as e:
        print(SkillOutput.error(str(e)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
