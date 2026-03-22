"""
StrategySearchService
三层候选策略生成：模板枚举 → 受约束随机搜索 → 逐步精简。
"""
from __future__ import annotations

import copy
import random
import uuid
from typing import Optional

import numpy as np

from backend.services.factor_profile_registry_service import FactorProfileRegistryService


class StrategySearchService:
    """生成候选策略列表，支持三层搜索：
    1. 模板枚举（_template_enum）
    2. 受约束随机搜索（_constrained_random）
    3. 逐步精简（_forward_selection）
    """

    def __init__(self) -> None:
        self._registry = FactorProfileRegistryService()

    # ------------------------------------------------------------------
    # 第一层：模板枚举
    # ------------------------------------------------------------------

    def _inject_top_factors(
        self,
        base_profiles: list[dict],
        top_factors: list[dict],
    ) -> list[dict]:
        """将 Stage-2/3 筛选出的 top_factors 注入到 base_profiles 的 signal 因子位置。

        每个 base_profile 保留其结构（params、risk_filter 因子），
        仅替换 signal 因子为 top_factors 中的因子（均匀分配权重）。
        top_factors 格式：[{"factor_name": str, "recommended_direction": int, ...}, ...]

        返回注入后的新 profile 列表（不修改原始 base_profiles）。
        """
        if not top_factors:
            return base_profiles

        injected: list[dict] = []
        for profile in base_profiles:
            risk_factors = [
                f for f in profile.get("factors", []) if f.get("role") != "signal"
            ]
            n_signals = len(top_factors)
            if n_signals == 0:
                injected.append(copy.deepcopy(profile))
                continue

            # 均匀分配权重，最后一个修正精度
            base_w = round(1.0 / n_signals, 6)
            signal_factors: list[dict] = []
            for i, tf in enumerate(top_factors):
                w = base_w if i < n_signals - 1 else round(1.0 - base_w * (n_signals - 1), 6)
                signal_factors.append({
                    "name": tf["factor_name"],
                    "direction": int(tf.get("recommended_direction", 1)),
                    "weight": w,
                    "role": "signal",
                })

            new_profile = copy.deepcopy(profile)
            new_profile["factors"] = signal_factors + [copy.deepcopy(f) for f in risk_factors]
            new_profile["description"] = (
                profile.get("description", profile.get("id", ""))
                + "（个股因子注入）"
            )
            # 注入后的 profile 必须有唯一 id，避免不同搜索任务共享同一模板 id
            # 导致 inline_profiles/<id>.json 互相覆盖
            new_profile["id"] = f"{profile.get('id', 'profile')}_{uuid.uuid4().hex[:8]}"
            injected.append(new_profile)

        return injected

    def _template_enum(self, base_profiles: list[dict]) -> list[dict]:
        """枚举所有内置 profile 作为候选，返回 base_profiles 的深拷贝。

        每个候选添加 candidate_id（UUID）和 source="search_result"。
        """
        candidates: list[dict] = []
        for profile in base_profiles:
            candidate = copy.deepcopy(profile)
            candidate["candidate_id"] = str(uuid.uuid4())
            candidate["source"] = "search_result"
            candidates.append(candidate)
        return candidates

    # ------------------------------------------------------------------
    # 第二层：受约束随机搜索
    # ------------------------------------------------------------------

    def _constrained_random(self, search_space: dict, n: int) -> list[dict]:
        """受约束随机搜索，生成 n 个候选策略。

        约束：
        - signal 因子权重之和 ∈ [0.999, 1.001]（Dirichlet 采样保证）
        - 至少 1 个 risk_filter 因子
        - 因子总数在 [min_factors, max_factors] 范围内
        - 同组因子数不超过 max_per_group
        - 单因子权重 ≤ 0.40

        超过 n*10 次尝试后返回已生成候选，不抛异常。
        """
        candidates: list[dict] = []
        max_attempts = n * 10
        attempts = 0

        factor_groups: dict[str, list[str]] = search_space.get("factor_groups", {})
        risk_factors: list[str] = search_space.get("risk_factors", [])
        min_factors: int = search_space.get("min_factors", 4)
        max_factors: int = search_space.get("max_factors", 8)
        max_per_group: int = search_space.get("max_per_group", 2)
        mode: str = search_space.get("mode", "temporal_pool")

        rebalance_choices: list = search_space.get("rebalance_choices", ["W-FRI"])
        hold_days_choices: list = search_space.get("hold_days_choices", [5])
        top_n_choices: list = search_space.get("top_n_choices", [10])
        score_threshold_choices: list = search_space.get("score_threshold_choices", [65])

        while len(candidates) < n and attempts < max_attempts:
            attempts += 1

            # Step 1: 从每个组随机选取因子
            selected_signal: list[str] = []
            selected_risk: list[str] = []

            for group_name, group_factors in factor_groups.items():
                # 区分该组中的 signal 因子和 risk 因子
                group_signal = [f for f in group_factors if f not in risk_factors]
                group_risk = [f for f in group_factors if f in risk_factors]

                # signal 因子：从组内随机选 0~max_per_group 个
                if group_signal:
                    max_from_group = min(max_per_group, len(group_signal))
                    n_from_group = random.randint(0, max_from_group)
                    selected_signal.extend(random.sample(group_signal, n_from_group))

                # risk 因子：从组内随机选 0~全部
                if group_risk:
                    n_risk = random.randint(0, len(group_risk))
                    selected_risk.extend(random.sample(group_risk, n_risk))

            # 确保至少 1 个 risk_filter 因子
            if len(selected_risk) == 0:
                if risk_factors:
                    selected_risk = [random.choice(risk_factors)]
                else:
                    continue

            total_factors = len(selected_signal) + len(selected_risk)

            # 约束：因子总数在 [min_factors, max_factors]
            if total_factors < min_factors:
                continue
            if total_factors > max_factors:
                # 随机裁剪 signal 因子（保留 risk 因子）
                max_signal = max_factors - len(selected_risk)
                if max_signal <= 0:
                    continue
                selected_signal = random.sample(selected_signal, min(len(selected_signal), max_signal))
                total_factors = len(selected_signal) + len(selected_risk)
                if total_factors < min_factors:
                    continue

            # 至少需要 1 个 signal 因子才能分配权重
            if len(selected_signal) == 0:
                continue

            # Step 2: Dirichlet 采样 signal 因子权重
            alpha = np.ones(len(selected_signal))
            weights = np.random.dirichlet(alpha)

            # 约束：单因子权重 ≤ 0.40
            if float(np.max(weights)) > 0.40:
                continue

            # Step 3: 随机选择参数
            params = {
                "rebalance": random.choice(rebalance_choices),
                "hold_days": random.choice(hold_days_choices) if hold_days_choices else 5,
                "top_n": random.choice(top_n_choices),
                "score_threshold": random.choice(score_threshold_choices),
                "percentile_window": 252,
            }

            # Step 4: 构建 profile 候选
            factors_list: list[dict] = []
            for fname, w in zip(selected_signal, weights.tolist()):
                factors_list.append({
                    "name": fname,
                    "direction": 1,
                    "weight": round(w, 6),
                    "role": "signal",
                })

            # 归一化确保权重和精确为 1
            signal_weight_sum = sum(f["weight"] for f in factors_list)
            if signal_weight_sum > 0:
                for f in factors_list:
                    f["weight"] = round(f["weight"] / signal_weight_sum, 6)
                # 修正最后一个因子以确保精确归一化
                total_w = sum(f["weight"] for f in factors_list[:-1])
                factors_list[-1]["weight"] = round(1.0 - total_w, 6)

            for fname in selected_risk:
                factors_list.append({
                    "name": fname,
                    "direction": -1,
                    "weight": 0.0,
                    "role": "risk_filter",
                    "threshold": 0.80,
                })

            candidate_id = str(uuid.uuid4())
            candidate = {
                "candidate_id": candidate_id,
                "id": f"search_{candidate_id[:8]}",
                "mode": mode,
                "description": f"随机搜索候选 {candidate_id[:8]}",
                "source": "search_result",
                "version": "1.0",
                "factors": factors_list,
                "params": params,
            }
            candidates.append(candidate)

        return candidates

    # ------------------------------------------------------------------
    # 第三层：逐步精简
    # ------------------------------------------------------------------

    def _forward_selection(
        self, top_candidates: list[dict], search_space: dict
    ) -> list[dict]:
        """在最优候选上做逐步精简：去掉权重最低的 signal 因子（如果因子数 > min_factors）。

        返回精简后的新候选列表（不修改原始候选）。
        """
        min_factors: int = search_space.get("min_factors", 4)
        refined: list[dict] = []

        for candidate in top_candidates:
            signal_factors = [
                f for f in candidate.get("factors", []) if f.get("role") == "signal"
            ]
            risk_factors = [
                f for f in candidate.get("factors", []) if f.get("role") != "signal"
            ]
            total_factors = len(signal_factors) + len(risk_factors)

            # 只有因子总数 > min_factors 时才精简
            if total_factors <= min_factors or len(signal_factors) <= 1:
                continue

            # 找到权重最低的 signal 因子并移除
            min_weight_factor = min(signal_factors, key=lambda f: f.get("weight", 0.0))
            new_signal = [f for f in signal_factors if f is not min_weight_factor]

            if len(new_signal) == 0:
                continue

            # 重新归一化权重
            total_w = sum(f.get("weight", 0.0) for f in new_signal)
            if total_w <= 0:
                continue

            new_signal_normalized = []
            for i, f in enumerate(new_signal):
                new_f = copy.deepcopy(f)
                if i < len(new_signal) - 1:
                    new_f["weight"] = round(f.get("weight", 0.0) / total_w, 6)
                else:
                    # 最后一个因子修正精度
                    prev_sum = sum(nf["weight"] for nf in new_signal_normalized)
                    new_f["weight"] = round(1.0 - prev_sum, 6)
                new_signal_normalized.append(new_f)

            new_candidate = copy.deepcopy(candidate)
            new_candidate["candidate_id"] = str(uuid.uuid4())
            new_candidate["id"] = f"refined_{new_candidate['candidate_id'][:8]}"
            new_candidate["description"] = f"精简候选（来自 {candidate.get('id', '')}）"
            new_candidate["source"] = "search_result"
            new_candidate["factors"] = new_signal_normalized + [copy.deepcopy(f) for f in risk_factors]
            refined.append(new_candidate)

        return refined

    # ------------------------------------------------------------------
    # 组合三层搜索
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        mode: str,
        base_profiles: list[dict],
        search_space: dict,
        n_candidates: int,
    ) -> list[dict]:
        """组合三层搜索生成候选策略列表。

        流程：
        1. 模板枚举（第一层）
        2. 受约束随机搜索补充到 n_candidates（第二层）
        3. 对前几个候选做逐步精简（第三层）

        每个候选均有唯一 candidate_id（UUID）和 source="search_result"。
        """
        # 第一层：模板枚举
        candidates = self._template_enum(base_profiles)

        # 第二层：随机搜索补充
        n_random = max(0, n_candidates - len(candidates))
        if n_random > 0:
            random_candidates = self._constrained_random(search_space, n_random)
            candidates.extend(random_candidates)

        # 第三层：对前 min(5, len(candidates)) 个候选做精简
        n_refine = min(5, len(candidates))
        if n_refine > 0:
            refined = self._forward_selection(candidates[:n_refine], search_space)
            candidates.extend(refined)

        return candidates

    # ------------------------------------------------------------------
    # 作用域入口
    # ------------------------------------------------------------------

    def search_for_stock(
        self,
        code: str,
        search_space_id: str,
        start_date: str,
        end_date: str,
        top_factors: Optional[list[dict]] = None,
    ) -> list[dict]:
        """为单只股票生成候选策略列表。

        Args:
            top_factors: 可选，Stage-2/3 筛选出的个股有效因子列表。
                         有则用其替换 base_profiles 的 signal 因子；无则使用全局 config。
        """
        search_space = self._registry.load_search_space(search_space_id)
        mode = search_space.get("mode", "temporal_pool")
        base_profiles = self._registry.load_profiles(mode)
        if top_factors:
            base_profiles = self._inject_top_factors(base_profiles, top_factors)
        n_candidates = search_space.get("n_candidates", 50)
        return self.generate_candidates(mode, base_profiles, search_space, n_candidates)

    def search_for_pool(
        self,
        codes: list[str],
        search_space_id: str,
        start_date: str,
        end_date: str,
        top_factors: Optional[list[dict]] = None,
    ) -> list[dict]:
        """为股票组生成候选策略列表。

        Args:
            top_factors: 可选，Stage-2/3 筛选出的有效因子列表。
                         有则用其替换 base_profiles 的 signal 因子；无则使用全局 config。
        """
        search_space = self._registry.load_search_space(search_space_id)
        mode = search_space.get("mode", "temporal_pool")
        base_profiles = self._registry.load_profiles(mode)
        if top_factors:
            base_profiles = self._inject_top_factors(base_profiles, top_factors)
        n_candidates = search_space.get("n_candidates", 50)
        return self.generate_candidates(mode, base_profiles, search_space, n_candidates)

    def search_for_cross_sectional(
        self,
        universe: str,
        search_space_id: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """为截面 universe 生成候选策略列表。"""
        search_space = self._registry.load_search_space(search_space_id)
        mode = search_space.get("mode", "cross_sectional_filter")
        base_profiles = self._registry.load_profiles(mode)
        n_candidates = search_space.get("n_candidates", 30)
        return self.generate_candidates(mode, base_profiles, search_space, n_candidates)
