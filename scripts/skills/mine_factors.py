#!/usr/bin/env python3
"""
Skill: mine_factors - 遗传算法因子挖掘

使用示例:
  python mine_factors.py 000001 --base "RSI(close,14),close/SMA(close,20)" --generations 10 2023-01-01 2024-12-31
  python mine_factors.py 600519 --base "RSI(14)" --gens 5
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.skills.skills_base import (
    SkillsClient, SkillOutput, create_parser, parse_date_range
)


def main():
    parser = create_parser(
        "mine_factors",
        "遗传算法因子挖掘\n\n" +
        "示例:\n" +
        "  python mine_factors.py 000001 --base 'RSI(close,14),close/SMA(close,20)' --generations 10 2023-01-01 2024-12-31\n" +
        "  python mine_factors.py 600519 --base 'RSI(14)' --gens 5"
    )

    parser.add_argument("stock_code", help="股票代码")
    parser.add_argument("--base", required=True, help="基础因子列表 (逗号分隔)")
    parser.add_argument("--generations", "--gens", dest="generations", type=int, default=10, help="进化代数 (默认: 10)")
    parser.add_argument("--population-size", type=int, default=50, help="种群大小 (默认: 50)")
    parser.add_argument("start_date", nargs="?", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("end_date", nargs="?", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--start", dest="start_override", help="开始日期 (覆盖位置参数)")
    parser.add_argument("--end", dest="end_override", help="结束日期 (覆盖位置参数)")
    parser.add_argument("--api-url", default="http://localhost:8000/api", help="API基地址")

    args = parser.parse_args()

    start = args.start_override or args.start_date
    end = args.end_override or args.end_date

    try:
        start_date, end_date = parse_date_range(start, end)
    except ValueError as e:
        print(SkillOutput.error(str(e)))
        return 1

    base_factors = [f.strip() for f in args.base.split(",")]

    client = SkillsClient(args.api_url)

    print(f"\n🧬 因子挖掘: {args.stock_code}")
    print(f"   基础因子: {', '.join(base_factors)}")
    print(f"   进化代数: {args.generations}")
    print(f"   时间范围: {start_date} 至 {end_date}")

    try:
        # 提交挖掘任务
        print(f"\n⏳ 提交挖掘任务...")
        submit_result = client.post(
            "/mining/genetic",
            data={
                "stock_code": args.stock_code,
                "base_factors": base_factors,
                "population_size": args.population_size,
                "n_generations": args.generations,
                "start_date": start_date,
                "end_date": end_date
            }
        )

        if not submit_result.get("success"):
            print(SkillOutput.error(submit_result.get("detail", "提交失败")))
            return 1

        task_id = submit_result.get("data", {}).get("task_id")
        print(f"✅ 任务已提交 (ID: {task_id})")

        # 轮询任务状态
        print(f"\n⏳ 监控挖掘进度...")
        max_wait = 3600  # 最多等待1小时
        start_time = time.time()
        check_interval = 5  # 初始检查间隔5秒

        while time.time() - start_time < max_wait:
            try:
                status_result = client.get(f"/mining/status/{task_id}")

                if not status_result.get("success"):
                    print(SkillOutput.error("获取状态失败"))
                    break

                status_data = status_result.get("data", {})
                current_gen = status_data.get("current_generation", 0)
                best_fitness = status_data.get("best_fitness", 0)
                status = status_data.get("status", "running")

                # 显示进度
                if status == "completed":
                    print(f"\n✅ 挖掘完成!")
                    break
                elif status == "running":
                    progress = (current_gen / args.generations) * 100
                    print(f"  Generation {current_gen}/{args.generations}: 最优适应度={best_fitness:.4f} ({progress:.0f}%)")
                    check_interval = min(10, check_interval + 1)  # 逐步增加检查间隔
                else:
                    print(f"  状态: {status}")

                time.sleep(check_interval)

            except Exception as e:
                print(f"  ⚠️ 获取状态出错: {e}")
                break

        # 获取结果
        print(f"\n⏳ 获取挖掘结果...")
        results_result = client.get(f"/mining/results/{task_id}")

        if not results_result.get("success"):
            print(SkillOutput.error(results_result.get("detail", "获取结果失败")))
            return 1

        results_data = results_result.get("data", {})
        best_factors = results_data.get("best_factors", [])

        # 显示结果
        print(SkillOutput.section("因子挖掘结果"))

        print(SkillOutput.subsection("发现因子 (Top 5)"))
        for i, factor_info in enumerate(best_factors[:5], 1):
            factor = factor_info.get("expression", "N/A")
            ic = factor_info.get("ic_mean", 0)
            ir = factor_info.get("ir_ratio", 0)
            fitness = factor_info.get("fitness", 0)

            print(f"\n{i}. {factor}")
            print(f"   ├─ IC均值: {ic:.4f}")
            print(f"   ├─ IR比率: {ir:.4f}")
            print(f"   └─ 适应度: {fitness:.4f}")

        # 统计信息
        print(SkillOutput.subsection("\n挖掘统计"))
        print(f"├─ 发现因子总数: {len(best_factors)}")
        print(f"├─ 最优适应度: {results_data.get('best_fitness', 0):.4f}")
        print(f"├─ 平均适应度: {results_data.get('avg_fitness', 0):.4f}")
        print(f"└─ 总用时: {results_data.get('elapsed_seconds', 0):.1f}秒")

        # 建议
        if len(best_factors) > 0:
            top_ic = best_factors[0].get("ic_mean", 0)
            if top_ic > 0.05:
                print(SkillOutput.subsection("\n💡 建议"))
                print("✅ 发现了高质量因子，强烈推荐用于实盘")
            elif top_ic > 0.03:
                print(SkillOutput.subsection("\n💡 建议"))
                print("⚠️ 发现了有效因子，建议进一步验证后使用")

        print("\n✅ 因子挖掘完成\n")
        return 0

    except Exception as e:
        print(SkillOutput.error(str(e)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
