"""
因子库审计服务。

目标：
1. 离线检查当前因子库是否覆盖了核心时序因子域；
2. 输出可直接用于后续搜索/筛选的 baseline 因子清单；
3. 为 Stage-1「离线选因子」提供独立、可复用的服务层入口。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from backend.core.database import get_db_session
from backend.repositories.factor_repository import FactorRepository
from backend.services.factor_profile_registry_service import FactorProfileRegistryService


class FactorLibraryAuditService:
    """审计当前因子库覆盖情况，并给出 baseline 因子清单。"""

    REQUIRED_DOMAINS = [
        "trend",
        "momentum",
        "reversal",
        "volatility",
        "volume_flow",
        "price_position",
        "risk",
        "pattern_regime",
    ]

    DOMAIN_LABELS = {
        "trend": "趋势",
        "momentum": "动量",
        "reversal": "反转",
        "volatility": "波动率",
        "volume_flow": "量价资金流",
        "price_position": "价格位置",
        "risk": "风险约束",
        "pattern_regime": "形态/状态",
        "sentiment": "情绪",
        "other": "其他",
    }

    MIN_REQUIRED_COUNT = {
        "trend": 4,
        "momentum": 4,
        "reversal": 4,
        "volatility": 3,
        "volume_flow": 3,
        "price_position": 3,
        "risk": 3,
        "pattern_regime": 3,
    }

    CATEGORY_DOMAIN_MAP = {
        "价格收益率": ["trend", "price_position"],
        "动量趋势": ["trend", "momentum"],
        "波动率风险": ["volatility", "risk"],
        "成交量资金流": ["volume_flow"],
        "结构模式": ["pattern_regime"],
        "动量加速度": ["momentum"],
        "反转信号": ["reversal"],
        "技术形态": ["pattern_regime"],
        "市场情绪": ["sentiment"],
        "风险指标": ["risk"],
        "均线系统": ["trend"],
        "价格位置": ["price_position"],
        "资金流动": ["volume_flow"],
    }

    def __init__(self) -> None:
        self._registry = FactorProfileRegistryService()

    def _infer_domains(self, factor: dict[str, Any]) -> list[str]:
        """根据分类和名称推断高层因子域。"""
        category = str(factor.get("category", "")).strip()
        name = str(factor.get("name", "")).strip().lower()
        domains = set(self.CATEGORY_DOMAIN_MAP.get(category, []))

        if any(token in name for token in ("ma", "sma", "ema", "trend", "macd", "adx")):
            domains.add("trend")
        if any(token in name for token in ("momentum", "roc", "rsi", "cci")):
            domains.add("momentum")
        if any(token in name for token in ("reversal", "deviation", "stochastic")):
            domains.add("reversal")
        if any(token in name for token in ("volatility", "atr", "bollinger", "drawdown", "var_")):
            domains.add("volatility")
        if any(token in name for token in ("volume", "obv", "money_flow", "force_index", "vwma")):
            domains.add("volume_flow")
        if any(token in name for token in ("percentile", "distance_to_", "price_range", "price_vs_")):
            domains.add("price_position")
        if any(token in name for token in ("risk", "drawdown", "var_", "skewness", "kurtosis")):
            domains.add("risk")
        if any(token in name for token in ("cross", "candles", "gap_", "regime", "new_high", "new_low")):
            domains.add("pattern_regime")

        if not domains:
            domains.add("other")

        return sorted(domains)

    def _load_factors(self, active_only: bool) -> list[dict[str, Any]]:
        db = get_db_session()
        repo = FactorRepository(db)
        try:
            return [factor.to_dict() for factor in repo.get_all(active_only=active_only)]
        finally:
            db.close()

    def _load_baseline_factor_names(self, search_space_id: str) -> list[str]:
        """收集后续 pipeline 的 baseline 因子集合。"""
        baseline_names: set[str] = set()

        try:
            search_space = self._registry.load_search_space(search_space_id)
            baseline_names.update(search_space.get("candidate_factors", []))
        except KeyError:
            pass

        for profile in self._registry.load_profiles("temporal_pool"):
            for factor_cfg in profile.get("factors", []):
                name = factor_cfg.get("name")
                if name:
                    baseline_names.add(name)

        return sorted(baseline_names)

    def audit_factor_library(
        self,
        active_only: bool = True,
        search_space_id: str = "temporal_default",
    ) -> dict[str, Any]:
        """执行因子库审计。"""
        factors = self._load_factors(active_only=active_only)
        baseline_names = self._load_baseline_factor_names(search_space_id)

        inventory: list[dict[str, Any]] = []
        for factor in factors:
            domains = self._infer_domains(factor)
            inventory.append(
                {
                    "name": factor["name"],
                    "category": factor.get("category", ""),
                    "source": factor.get("source", ""),
                    "formula_type": factor.get("formula_type", "expression"),
                    "is_active": bool(factor.get("is_active", True)),
                    "description": factor.get("description", ""),
                    "domains": domains,
                    "in_baseline_seed": factor["name"] in baseline_names,
                }
            )

        inventory.sort(key=lambda row: (row["category"], row["name"]))

        domain_counter: Counter[str] = Counter()
        domain_sources: dict[str, Counter[str]] = defaultdict(Counter)
        domain_categories: dict[str, set[str]] = defaultdict(set)

        for row in inventory:
            for domain in row["domains"]:
                domain_counter[domain] += 1
                domain_sources[domain][row["source"]] += 1
                if row["category"]:
                    domain_categories[domain].add(row["category"])

        domain_coverage: list[dict[str, Any]] = []
        covered_domains = 0
        sufficient_domains = 0

        for domain in self.REQUIRED_DOMAINS:
            count = domain_counter.get(domain, 0)
            minimum = self.MIN_REQUIRED_COUNT[domain]
            covered = count > 0
            sufficient = count >= minimum
            if covered:
                covered_domains += 1
            if sufficient:
                sufficient_domains += 1

            domain_coverage.append(
                {
                    "domain": domain,
                    "domain_label": self.DOMAIN_LABELS.get(domain, domain),
                    "factor_count": count,
                    "minimum_required": minimum,
                    "covered": covered,
                    "sufficient": sufficient,
                    "preset_count": domain_sources[domain].get("preset", 0),
                    "user_count": domain_sources[domain].get("user", 0),
                    "categories": sorted(domain_categories[domain]),
                }
            )

        category_counts: Counter[tuple[str, str]] = Counter(
            (row["category"], row["source"]) for row in inventory
        )
        category_coverage = [
            {
                "category": category,
                "source": source,
                "factor_count": count,
            }
            for (category, source), count in sorted(category_counts.items())
        ]

        inventory_names = {row["name"] for row in inventory}
        baseline_selected_factors = [
            row for row in inventory if row["name"] in baseline_names
        ]
        missing_baseline_factors = sorted(set(baseline_names) - inventory_names)

        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "active_only": active_only,
                "search_space_id": search_space_id,
                "factor_count_total": len(inventory),
                "baseline_factor_count": len(baseline_names),
                "baseline_factor_count_present": len(baseline_selected_factors),
                "required_domain_count": len(self.REQUIRED_DOMAINS),
            },
            "summary": {
                "covered_domain_count": covered_domains,
                "sufficient_domain_count": sufficient_domains,
                "coverage_ratio": round(covered_domains / len(self.REQUIRED_DOMAINS), 4),
                "sufficiency_ratio": round(
                    sufficient_domains / len(self.REQUIRED_DOMAINS), 4
                ),
                "missing_domains": [
                    item["domain"]
                    for item in domain_coverage
                    if not item["covered"]
                ],
                "insufficient_domains": [
                    item["domain"]
                    for item in domain_coverage
                    if item["covered"] and not item["sufficient"]
                ],
                "missing_baseline_factors": missing_baseline_factors,
            },
            "domain_coverage": domain_coverage,
            "category_coverage": category_coverage,
            "baseline_selected_factors": baseline_selected_factors,
            "factor_inventory": inventory,
        }


factor_library_audit_service = FactorLibraryAuditService()
