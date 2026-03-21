"""
ChampionStrategyService
独立运行完整的 champion 搜索流程：
  加载配置 → 生成候选 → walk-forward 评价 → 过滤失效候选 → 排序 → 选出 champion → 落盘

支持三种作用域：
  - run_for_stock(code, ...)
  - run_for_pool(codes, ...)
  - run_for_cross_sectional(universe, ...)

任务状态落盘到 data/reports/strategy_search/{task_id}/
"""
from __future__ import annotations

import csv
import json
import logging
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TASK_RESULTS_DIR = Path("data/reports/strategy_search")


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _months_between(start: str, end: str) -> float:
    """计算两个日期之间的月数（近似）。"""
    s = _parse_date(start)
    e = _parse_date(end)
    return (e.year - s.year) * 12 + (e.month - s.month) + (e.day - s.day) / 30.0


def _safe_float(val, default: float = float("nan")) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class ChampionStrategyService:
    """独立运行 champion 搜索任务，选出最优策略并落盘。"""

    def __init__(self) -> None:
        from backend.services.strategy_search_service import StrategySearchService
        from backend.services.strategy_evaluation_service import StrategyEvaluationService
        from backend.services.champion_registry_service import ChampionRegistryService

        self._search_svc = StrategySearchService()
        self._eval_svc = StrategyEvaluationService()
        self._registry = ChampionRegistryService()

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def run_for_stock(
        self,
        code: str,
        search_space_id: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        """为单只股票运行完整搜索流程，返回 champion 配置。"""
        return self._run(
            scope_type="single_stock",
            scope_key=code,
            search_space_id=search_space_id,
            start_date=start_date,
            end_date=end_date,
            codes=[code],
            universe=None,
            max_workers=1,
        )

    def run_for_pool(
        self,
        codes: list[str],
        search_space_id: str,
        start_date: str,
        end_date: str,
        max_workers: int = 1,
    ) -> dict:
        """为股票组运行完整搜索流程，返回 champion 配置。"""
        from backend.services.champion_registry_service import ChampionRegistryService

        group_hash = ChampionRegistryService._scope_key_to_hash(codes)
        return self._run(
            scope_type="stock_group",
            scope_key=group_hash,
            search_space_id=search_space_id,
            start_date=start_date,
            end_date=end_date,
            codes=codes,
            universe=None,
            max_workers=max_workers,
        )

    def run_for_cross_sectional(
        self,
        universe: str,
        search_space_id: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        """为截面 universe 运行完整搜索流程，返回 champion 配置。"""
        return self._run(
            scope_type="cross_sectional_universe",
            scope_key=universe,
            search_space_id=search_space_id,
            start_date=start_date,
            end_date=end_date,
            codes=None,
            universe=universe,
            max_workers=1,
        )

    def get_champion(self, scope_type: str, scope_key: str) -> dict | None:
        """查询当前 champion 配置，不存在时返回 None。"""
        return self._registry.load(scope_type, scope_key)

    def apply_champion(
        self, scope_type: str, scope_key: str, champion: dict
    ) -> None:
        """将 champion 写入 registry，供打分接口直接调用。"""
        self._registry.save(scope_type, scope_key, champion)

    # ------------------------------------------------------------------
    # 核心搜索流程
    # ------------------------------------------------------------------

    def _run(
        self,
        scope_type: str,
        scope_key: str,
        search_space_id: str,
        start_date: str,
        end_date: str,
        codes: Optional[list[str]],
        universe: Optional[str],
        max_workers: int,
    ) -> dict:
        """执行完整搜索流程，返回 champion 字典。"""
        # 1. 验证时间跨度 >= 18 个月
        months = _months_between(start_date, end_date)
        if months < 18:
            raise ValueError(
                f"时间跨度不足 18 个月（实际约 {months:.1f} 个月）。"
                f"请将 start_date 和 end_date 之间的跨度设置为至少 18 个月，"
                f"以保证能生成完整的 walk-forward 窗口。"
            )

        # 2. 生成 task_id，创建任务目录
        task_id = str(uuid.uuid4())
        task_dir = _TASK_RESULTS_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # 3. 写入 task_status.json（status=running）和 input_scope.json
        task_status = {
            "task_id": task_id,
            "scope_type": scope_type,
            "scope_key": scope_key,
            "search_space_id": search_space_id,
            "status": "running",
            "started_at": _now_iso(),
            "completed_at": None,
            "champion_id": None,
            "error": None,
        }
        _write_json(task_dir / "task_status.json", task_status)

        input_scope = {
            "scope_type": scope_type,
            "scope_key": scope_key,
            "search_space_id": search_space_id,
            "start_date": start_date,
            "end_date": end_date,
            "codes": codes,
            "universe": universe,
            "max_workers": max_workers,
        }
        _write_json(task_dir / "input_scope.json", input_scope)

        try:
            champion = self._execute_search(
                task_id=task_id,
                task_dir=task_dir,
                scope_type=scope_type,
                scope_key=scope_key,
                search_space_id=search_space_id,
                start_date=start_date,
                end_date=end_date,
                codes=codes,
                universe=universe,
                max_workers=max_workers,
            )
        except Exception as exc:
            # 11. 任务失败：更新 task_status.json
            task_status["status"] = "failed"
            task_status["error"] = str(exc)
            _write_json(task_dir / "task_status.json", task_status)
            raise

        # 11. 任务完成：更新 task_status.json
        task_status["status"] = "completed"
        task_status["completed_at"] = _now_iso()
        task_status["champion_id"] = champion.get("champion_id")
        _write_json(task_dir / "task_status.json", task_status)

        return champion

    def _execute_search(
        self,
        task_id: str,
        task_dir: Path,
        scope_type: str,
        scope_key: str,
        search_space_id: str,
        start_date: str,
        end_date: str,
        codes: Optional[list[str]],
        universe: Optional[str],
        max_workers: int,
    ) -> dict:
        """执行搜索核心逻辑，返回 champion 字典。"""

        # 4. 调用 StrategySearchService 生成候选策略
        if scope_type == "single_stock":
            candidates = self._search_svc.search_for_stock(
                code=scope_key,
                search_space_id=search_space_id,
                start_date=start_date,
                end_date=end_date,
            )
        elif scope_type == "stock_group":
            candidates = self._search_svc.search_for_pool(
                codes=codes or [],
                search_space_id=search_space_id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            candidates = self._search_svc.search_for_cross_sectional(
                universe=universe or scope_key,
                search_space_id=search_space_id,
                start_date=start_date,
                end_date=end_date,
            )

        logger.info(f"[{task_id}] 生成候选策略 {len(candidates)} 个")

        # 5. 对每个候选调用 StrategyEvaluationService 评价
        evaluated: list[dict] = []
        for i, candidate in enumerate(candidates):
            try:
                if scope_type == "cross_sectional_universe":
                    result = self._eval_svc.evaluate_cross_sectional_candidate(
                        candidate=candidate,
                        universe=universe or scope_key,
                        start_date=start_date,
                        end_date=end_date,
                    )
                else:
                    eval_codes = codes if codes else [scope_key]
                    result = self._eval_svc.evaluate_temporal_pool_candidate(
                        candidate=candidate,
                        codes=eval_codes,
                        start_date=start_date,
                        end_date=end_date,
                        max_workers=max_workers,
                    )
                evaluated.append({**candidate, **result})
                logger.info(
                    f"[{task_id}] 候选 {i+1}/{len(candidates)} 评价完成，"
                    f"stability_score={result.get('stability_score', 0):.2f}"
                )
            except Exception as exc:
                logger.warning(f"[{task_id}] 候选 {i+1}/{len(candidates)} 评价失败，已跳过: {exc}")

        # 6. 如果所有候选评价失败，抛 RuntimeError
        if not evaluated:
            raise RuntimeError(
                f"所有 {len(candidates)} 个候选策略评价均失败，"
                "请检查数据质量或缩短时间范围。"
            )

        # 7. 过滤：优先过滤 test Sharpe <= 0 或年化收益 < -20% 的候选
        valid_candidates = [
            c for c in evaluated
            if _safe_float(c.get("test_metrics", {}).get("sharpe"), 0.0) > 0
            and _safe_float(c.get("test_metrics", {}).get("annual_return"), 0.0) > -0.20
        ]

        # 过滤后无有效候选时降级取最优
        if not valid_candidates:
            logger.warning(
                f"[{task_id}] 过滤后无有效候选，降级为从所有已评价候选中选出最优"
            )
            valid_candidates = evaluated

        # 8. 按 stability_score 降序排序，选出 champion
        valid_candidates.sort(
            key=lambda c: _safe_float(c.get("stability_score"), 0.0),
            reverse=True,
        )
        best = valid_candidates[0]

        champion_id = str(uuid.uuid4())
        champion = {
            "champion_id": champion_id,
            "scope_type": scope_type,
            "scope_key": scope_key,
            "mode": best.get("mode", "temporal_pool"),
            "profile_id": best.get("profile_id") or best.get("id", ""),
            "config": {
                k: v for k, v in best.items()
                if k not in (
                    "train_metrics", "valid_metrics", "test_metrics",
                    "stability_score", "window_metrics",
                )
            },
            "metrics": best.get("test_metrics", {}),
            "stability_score": _safe_float(best.get("stability_score"), 0.0),
            "selected_at": _now_iso(),
            "effective_from": end_date,
            "report_path": str(task_dir),
        }

        # 9. 调用 ChampionRegistryService.save 保存 champion
        self._registry.save(scope_type, scope_key, champion)

        # 10. 写入结果文件
        _write_candidate_leaderboard(task_dir, evaluated)
        _write_candidate_configs(task_dir, evaluated)
        _write_best_strategy(task_dir, champion)
        _write_selection_report(task_dir, champion, evaluated, task_id)

        logger.info(
            f"[{task_id}] Champion 选出：profile_id={champion['profile_id']}，"
            f"stability_score={champion['stability_score']:.2f}"
        )

        return champion


# ------------------------------------------------------------------
# 文件写入辅助函数
# ------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    """将字典写入 JSON 文件（UTF-8，缩进 2）。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_candidate_leaderboard(task_dir: Path, evaluated: list[dict]) -> None:
    """写入 candidate_leaderboard.csv，按 stability_score 降序排列。"""
    sorted_candidates = sorted(
        evaluated,
        key=lambda c: _safe_float(c.get("stability_score"), 0.0),
        reverse=True,
    )

    fieldnames = [
        "candidate_id",
        "profile_id",
        "stability_score",
        "test_sharpe",
        "test_annual_return",
        "test_max_drawdown",
        "train_sharpe",
        "valid_sharpe",
        "mode",
        "source",
    ]

    rows = []
    for c in sorted_candidates:
        test_m = c.get("test_metrics") or {}
        train_m = c.get("train_metrics") or {}
        valid_m = c.get("valid_metrics") or {}
        rows.append({
            "candidate_id": c.get("candidate_id", ""),
            "profile_id": c.get("profile_id") or c.get("id", ""),
            "stability_score": _safe_float(c.get("stability_score"), 0.0),
            "test_sharpe": _safe_float(test_m.get("sharpe")),
            "test_annual_return": _safe_float(test_m.get("annual_return")),
            "test_max_drawdown": _safe_float(test_m.get("max_drawdown")),
            "train_sharpe": _safe_float(train_m.get("sharpe")),
            "valid_sharpe": _safe_float(valid_m.get("sharpe")),
            "mode": c.get("mode", ""),
            "source": c.get("source", ""),
        })

    csv_path = task_dir / "candidate_leaderboard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_candidate_configs(task_dir: Path, evaluated: list[dict]) -> None:
    """写入 candidate_configs.json，包含所有候选的完整配置。"""
    configs = []
    for c in evaluated:
        config_entry = {
            k: v for k, v in c.items()
            if k not in ("window_metrics",)
        }
        configs.append(config_entry)
    _write_json(task_dir / "candidate_configs.json", {"candidates": configs})


def _write_best_strategy(task_dir: Path, champion: dict) -> None:
    """写入 best_strategy.json。"""
    _write_json(task_dir / "best_strategy.json", champion)


def _write_selection_report(
    task_dir: Path,
    champion: dict,
    evaluated: list[dict],
    task_id: str,
) -> None:
    """写入 selection_report.md，包含搜索摘要和 champion 信息。"""
    now = _now_iso()
    n_total = len(evaluated)
    stability = _safe_float(champion.get("stability_score"), 0.0)
    test_m = champion.get("metrics") or {}
    test_sharpe = _safe_float(test_m.get("sharpe"))
    test_return = _safe_float(test_m.get("annual_return"))
    test_dd = _safe_float(test_m.get("max_drawdown"))

    # 统计过滤情况
    n_valid = sum(
        1 for c in evaluated
        if _safe_float((c.get("test_metrics") or {}).get("sharpe"), 0.0) > 0
        and _safe_float((c.get("test_metrics") or {}).get("annual_return"), 0.0) > -0.20
    )

    lines = [
        "# Champion 策略选择报告",
        "",
        f"**任务 ID**: `{task_id}`",
        f"**生成时间**: {now}",
        "",
        "## 搜索摘要",
        "",
        f"- 候选策略总数：{n_total}",
        f"- 通过过滤的候选数：{n_valid}（test Sharpe > 0 且年化收益 > -20%）",
        f"- 降级模式：{'是' if n_valid == 0 else '否'}",
        "",
        "## Champion 策略",
        "",
        f"- **Champion ID**: `{champion.get('champion_id', '')}`",
        f"- **Profile ID**: `{champion.get('profile_id', '')}`",
        f"- **作用域类型**: {champion.get('scope_type', '')}",
        f"- **作用域标识**: {champion.get('scope_key', '')}",
        f"- **稳定性总分**: {stability:.2f}",
        f"- **生效日期**: {champion.get('effective_from', '')}",
        "",
        "## 测试期指标",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Sharpe | {test_sharpe:.4f} |",
        f"| 年化收益 | {test_return:.4f} |",
        f"| 最大回撤 | {test_dd:.4f} |",
        "",
        "## 候选排行榜（前 10）",
        "",
        "| 排名 | Candidate ID | Stability Score | Test Sharpe | Test Annual Return |",
        "|------|-------------|-----------------|-------------|-------------------|",
    ]

    sorted_candidates = sorted(
        evaluated,
        key=lambda c: _safe_float(c.get("stability_score"), 0.0),
        reverse=True,
    )
    for rank, c in enumerate(sorted_candidates[:10], 1):
        cid = c.get("candidate_id", "")[:8]
        score = _safe_float(c.get("stability_score"), 0.0)
        tm = c.get("test_metrics") or {}
        sharpe = _safe_float(tm.get("sharpe"))
        ret = _safe_float(tm.get("annual_return"))
        lines.append(f"| {rank} | `{cid}` | {score:.2f} | {sharpe:.4f} | {ret:.4f} |")

    lines.append("")
    lines.append("---")
    lines.append(f"*报告由 ChampionStrategyService 自动生成，任务目录：`{task_dir}`*")
    lines.append("")

    report_path = task_dir / "selection_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
