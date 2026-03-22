"""Stage-1: 离线因子库审计与 baseline 因子选择。"""

from __future__ import annotations

import argparse

from pipeline_common import (
    bootstrap_runtime,
    ensure_output_dir,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线审计当前因子库覆盖情况")
    parser.add_argument(
        "--search-space-id",
        default="temporal_default",
        help="用于抽取 baseline 因子集合的 search_space_id",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="包含未启用因子一起审计",
    )
    parser.add_argument(
        "--output-dir",
        default="data/pipeline_runs/stage1_factor_library_audit/latest",
        help="输出目录",
    )
    return parser.parse_args()


def build_summary(result: dict) -> str:
    meta = result["metadata"]
    summary = result["summary"]
    lines = [
        "# Stage-1 因子库审计",
        "",
        f"- 生成时间: {meta['generated_at']}",
        f"- 审计因子数: {meta['factor_count_total']}",
        f"- baseline 因子数: {meta['baseline_factor_count_present']}/{meta['baseline_factor_count']}",
        f"- 核心域覆盖: {summary['covered_domain_count']}/{meta['required_domain_count']}",
        f"- 核心域充足: {summary['sufficient_domain_count']}/{meta['required_domain_count']}",
        "",
        "## 缺口",
        "",
        f"- 缺失域: {', '.join(summary['missing_domains']) or '无'}",
        f"- 覆盖不足域: {', '.join(summary['insufficient_domains']) or '无'}",
        f"- baseline 缺失因子: {', '.join(summary['missing_baseline_factors']) or '无'}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    bootstrap_runtime(load_preset_factors=False)

    from backend.services.factor_library_audit_service import (
        factor_library_audit_service,
    )

    output_dir = ensure_output_dir(args.output_dir)

    result = factor_library_audit_service.audit_factor_library(
        active_only=not args.include_inactive,
        search_space_id=args.search_space_id,
    )

    write_json(output_dir / "result.json", result)
    write_csv(output_dir / "factor_inventory.csv", result["factor_inventory"])
    write_csv(output_dir / "domain_coverage.csv", result["domain_coverage"])
    write_csv(output_dir / "category_coverage.csv", result["category_coverage"])
    write_csv(
        output_dir / "baseline_selected_factors.csv",
        result["baseline_selected_factors"],
    )
    write_text(output_dir / "summary.md", build_summary(result))

    print("Stage-1 完成：离线因子库审计")
    print(f"输出目录: {output_dir}")
    print(f"因子总数: {result['metadata']['factor_count_total']}")
    print(
        "核心域覆盖: "
        f"{result['summary']['covered_domain_count']}/"
        f"{result['metadata']['required_domain_count']}"
    )


if __name__ == "__main__":
    main()
