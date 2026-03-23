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
import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TASK_RESULTS_DIR = Path("data/reports/strategy_search")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _candidate_is_usable(candidate: dict, scope_type: str) -> bool:
    test_metrics = candidate.get("test_metrics") or {}
    sharpe = _safe_float(test_metrics.get("sharpe"), 0.0)
    annual_return = _safe_float(test_metrics.get("annual_return"), 0.0)

    if sharpe <= 0 or annual_return <= -0.20:
        return False

    if scope_type != "single_stock":
        return True

    active_ratio = _safe_float(test_metrics.get("active_ratio"), float("nan"))
    if active_ratio == active_ratio:
        return active_ratio > 0

    active_days = _safe_float(test_metrics.get("active_days"), float("nan"))
    if active_days == active_days:
        return active_days > 0

    return True


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
        top_factors: Optional[list[dict]] = None,
    ) -> dict:
        """为单只股票运行完整搜索流程，返回 champion 配置。

        Args:
            top_factors: 可选，Stage-2/3 筛选出的个股有效因子列表。
                         有则注入到候选 profile 的 signal 因子；无则使用全局 config。
        """
        return self._run(
            scope_type="single_stock",
            scope_key=code,
            search_space_id=search_space_id,
            start_date=start_date,
            end_date=end_date,
            codes=[code],
            universe=None,
            max_workers=1,
            top_factors=top_factors,
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
        """将 champion 写入 registry，供打分接口直接调用。

        若 champion 携带 inline_profile（搜索产出的 profile 定义），
        同步持久化到磁盘，确保 FactorProfileRegistryService.get_profile() 可查到。
        """
        # 持久化 inline profile（若存在）
        inline_profile = champion.get("inline_profile")
        if inline_profile:
            _persist_inline_profile(inline_profile)
        elif champion.get("profile_id") and champion.get("config", {}).get("factors"):
            # 兼容：champion 本身带 factors 定义但未包装成 inline_profile 字段
            profile_id = champion["profile_id"]
            cfg = champion["config"]
            inline_profile = {
                "id": profile_id,
                "mode": champion.get("mode", "temporal_pool"),
                "description": cfg.get("description", f"手动应用 champion（{scope_type}:{scope_key}）"),
                "source": "apply_champion",
                "version": cfg.get("version", "1.0"),
                "factors": cfg["factors"],
                "params": cfg.get("params", {}),
            }
            _persist_inline_profile(inline_profile)

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
        top_factors: Optional[list[dict]] = None,
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
                top_factors=top_factors,
            )
        except Exception as exc:
            # 11. 任务失败：更新 task_status.json
            task_status["status"] = "failed"
            task_status["error"] = str(exc)
            _write_json(task_dir / "task_status.json", task_status)
            _write_run_manifest(task_dir, module="champion_strategy", data_coverage=1.0, quality_gate=[], result_status="failed")
            raise

        # 11. 任务完成：更新 task_status.json
        task_status["status"] = "completed"
        task_status["completed_at"] = _now_iso()
        task_status["champion_id"] = champion.get("champion_id")
        _write_json(task_dir / "task_status.json", task_status)
        _write_run_manifest(task_dir, module="champion_strategy", data_coverage=1.0, quality_gate=[], result_status="completed")

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
        top_factors: Optional[list[dict]] = None,
    ) -> dict:
        """执行搜索核心逻辑，返回 champion 字典。"""

        # 4. 调用 StrategySearchService 生成候选策略
        if scope_type == "single_stock":
            candidates = self._search_svc.search_for_stock(
                code=scope_key,
                search_space_id=search_space_id,
                start_date=start_date,
                end_date=end_date,
                top_factors=top_factors,
            )
        elif scope_type == "stock_group":
            candidates = self._search_svc.search_for_pool(
                codes=codes or [],
                search_space_id=search_space_id,
                start_date=start_date,
                end_date=end_date,
                top_factors=top_factors,
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

        # 7. 过滤：排除 evaluation_status != "valid" 的候选（无效评估结果）
        #    同时排除 test Sharpe <= 0 或年化收益 < -20% 的候选
        valid_candidates = [
            c for c in evaluated
            if c.get("evaluation_status", "valid") == "valid"
            and _candidate_is_usable(c, scope_type)
        ]

        # 收集所有候选的失败原因，用于报告
        failure_reasons = _collect_failure_reasons(evaluated, scope_type)

        # 过滤后无有效候选时，任务失败（不降级）
        if not valid_candidates:
            _write_candidate_leaderboard(task_dir, evaluated)
            _write_candidate_configs(task_dir, evaluated)
            _write_failure_report(task_dir, evaluated, failure_reasons, task_id)
            if scope_type == "single_stock":
                raise RuntimeError(
                    "未找到可用的 single_stock champion：所有候选在测试期要么无持仓，"
                    "要么 Sharpe<=0，要么年化收益<-20%，要么评估状态无效。"
                )
            raise RuntimeError(
                f"过滤后无 evaluation_status=valid 的候选（共 {len(evaluated)} 个已评价候选）。"
                f"失败原因：{failure_reasons}"
            )

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
            "profile_id": best.get("profile_id") or best.get("id") or best.get("candidate_id", ""),
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

        # 若 champion 来自 inline 搜索候选（有 factors 定义），
        # 将完整 profile 定义持久化到磁盘，确保进程重启后打分链仍可复用。
        if best.get("factors"):
            profile_id = champion["profile_id"]
            inline_profile = {
                "id": profile_id,
                "mode": best.get("mode", "temporal_pool"),
                "description": best.get("description", f"搜索产出 champion（{scope_type}:{scope_key}）"),
                "source": "search_result",
                "version": best.get("version", "1.0"),
                "factors": best["factors"],
                "params": best.get("params", {}),
            }
            champion["inline_profile"] = inline_profile
            # 持久化到独立目录，供 FactorProfileRegistryService 查找
            _persist_inline_profile(inline_profile)

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

def _collect_failure_reasons(evaluated: list[dict], scope_type: str) -> list[dict]:
    """收集所有候选的失败原因，用于报告。"""
    reasons = []
    for c in evaluated:
        candidate_id = c.get("candidate_id", c.get("profile_id") or c.get("id", ""))
        eval_status = c.get("evaluation_status", "valid")
        invalid_reason = c.get("invalid_reason", "")
        test_m = c.get("test_metrics") or {}
        sharpe = _safe_float(test_m.get("sharpe"), 0.0)
        annual_return = _safe_float(test_m.get("annual_return"), 0.0)
        active_ratio = _safe_float(test_m.get("active_ratio"), float("nan"))

        failure_codes = []
        if eval_status == "invalid":
            failure_codes.append(invalid_reason or "EVALUATION_STATUS_INVALID")
        if sharpe <= 0:
            failure_codes.append("SHARPE_NON_POSITIVE")
        if annual_return <= -0.20:
            failure_codes.append("ANNUAL_RETURN_TOO_LOW")
        if scope_type == "single_stock":
            if active_ratio == active_ratio and active_ratio <= 0:
                failure_codes.append("NO_ACTIVE_POSITION")

        reasons.append({
            "candidate_id": candidate_id,
            "evaluation_status": eval_status,
            "stability_score": _safe_float(c.get("stability_score")),
            "failure_codes": failure_codes,
        })
    return reasons


def _write_failure_report(
    task_dir: Path,
    evaluated: list[dict],
    failure_reasons: list[dict],
    task_id: str,
) -> None:
    """写入 failure_report.md，包含所有候选的失败原因。"""
    now = _now_iso()
    n_total = len(evaluated)

    lines = [
        "# Champion 搜索失败报告",
        "",
        f"**任务 ID**: `{task_id}`",
        f"**生成时间**: {now}",
        f"**结果**: 无 evaluation_status=valid 的候选，任务失败",
        "",
        "## 候选失败原因汇总",
        "",
        f"共评价 {n_total} 个候选，全部未通过过滤：",
        "",
        "| 候选 ID | 评估状态 | 稳定性分 | 失败原因 |",
        "|---------|---------|---------|---------|",
    ]

    for r in failure_reasons:
        cid = str(r.get("candidate_id", ""))[:12]
        status = r.get("evaluation_status", "")
        score = r.get("stability_score", float("nan"))
        score_str = f"{score:.2f}" if score == score else "NaN"
        codes = ", ".join(r.get("failure_codes", [])) or "无"
        lines.append(f"| `{cid}` | {status} | {score_str} | {codes} |")

    lines.append("")
    lines.append("---")
    lines.append(f"*报告由 ChampionStrategyService 自动生成，任务目录：`{task_dir}`*")
    lines.append("")

    report_path = task_dir / "failure_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


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
        "test_active_ratio",
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
            "test_active_ratio": _safe_float(test_m.get("active_ratio")),
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


def _persist_inline_profile(profile: dict) -> None:
    """将搜索产出的 inline profile 持久化到 <project_root>/data/champions/inline_profiles/{id}.json。

    FactorProfileRegistryService 会在磁盘 profile 查不到时扫描此目录，
    确保进程重启后 score_one_stock_with_profile 仍能按 profile_id 找到定义。
    使用与 FactorProfileRegistryService 相同的项目根目录基准，避免 cwd 不一致导致
    写入路径与读取路径不匹配。
    """
    profile_id = profile.get("id", "")
    if not profile_id:
        return
    # 与 FactorProfileRegistryService._PROJECT_ROOT 保持一致：此文件在 backend/services/，向上两级
    _project_root = Path(__file__).resolve().parent.parent.parent
    inline_dir = _project_root / "data" / "champions" / "inline_profiles"
    inline_dir.mkdir(parents=True, exist_ok=True)
    path = inline_dir / f"{profile_id}.json"
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_run_manifest(
    task_dir: Path,
    module: str,
    data_coverage: float,
    quality_gate: list[str],
    result_status: str,
) -> None:
    """写入 run_manifest.json，包含评估任务的追溯信息。

    Args:
        task_dir: 任务产物目录
        module: 模块/服务名称（如 "champion_strategy"、"temporal_filter_validation"）
        data_coverage: 数据覆盖率（0.0 到 1.0）
        quality_gate: 原因码列表（空列表表示通过）
        result_status: 结果状态（如 "completed"、"failed"、"invalid_run"、"no_valid_candidate"）
    """
    from backend.services.cache_service import SCORING_SERVICE_VERSION
    from backend.core.settings import settings

    # 生成 settings 关键配置的 sha256 指纹（前 12 位）
    key_settings = {
        "EVAL_MIN_CODE_COVERAGE": settings.EVAL_MIN_CODE_COVERAGE,
        "EVAL_MIN_IC_SAMPLES": settings.EVAL_MIN_IC_SAMPLES,
        "SCORING_SERVICE_VERSION": SCORING_SERVICE_VERSION,
    }
    settings_json = json.dumps(key_settings, sort_keys=True)
    settings_fingerprint = hashlib.sha256(settings_json.encode()).hexdigest()[:12]

    manifest = {
        "run_id": str(uuid.uuid4()),
        "module": module,
        "generated_at": _now_iso(),
        "schema_version": "1.0",
        "settings_fingerprint": settings_fingerprint,
        "data_coverage": data_coverage,
        "quality_gate": {
            "passed": len(quality_gate) == 0,
            "reason_codes": quality_gate,
        },
        "result_status": result_status,
    }
    _write_json(task_dir / "run_manifest.json", manifest)


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
    test_active_ratio = _safe_float(test_m.get("active_ratio"))

    # 统计过滤情况
    scope_type = champion.get("scope_type", "")
    n_valid = sum(
        1 for c in evaluated
        if c.get("evaluation_status", "valid") == "valid"
        and _candidate_is_usable(c, scope_type)
    )
    if scope_type == "single_stock":
        filter_desc = "evaluation_status=valid、test Sharpe > 0、年化收益 > -20%，且测试期存在实际持仓"
    else:
        filter_desc = "evaluation_status=valid、test Sharpe > 0 且年化收益 > -20%"

    lines = [
        "# Champion 策略选择报告",
        "",
        f"**任务 ID**: `{task_id}`",
        f"**生成时间**: {now}",
        "",
        "## 搜索摘要",
        "",
        f"- 候选策略总数：{n_total}",
        f"- 通过过滤的候选数：{n_valid}（{filter_desc}）",
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
        f"| 活跃比例 | {test_active_ratio:.4f} |",
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
