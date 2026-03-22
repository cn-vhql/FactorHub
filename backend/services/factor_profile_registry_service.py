"""
FactorProfileRegistryService
加载、校验、分发 profile 配置和 search space 配置。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 项目根目录（此文件位于 backend/services/，向上两级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TEMPORAL_POOL_FILE = _PROJECT_ROOT / "config" / "temporal_pool_profiles.json"
_CROSS_SECTIONAL_FILE = _PROJECT_ROOT / "config" / "cross_sectional_profiles.json"
_SEARCH_SPACE_FILE = _PROJECT_ROOT / "config" / "strategy_search_space.json"

_VALID_MODES = {"temporal_pool", "cross_sectional_filter"}


class FactorProfileRegistryService:
    """加载、校验、分发 profile 配置和 search space 配置的服务。

    支持热重载：每次调用时检测配置文件 mtime，文件变更后自动重新加载。
    支持运行时注册临时 inline profile（用于搜索候选评估，不持久化）。
    """

    # 进程级临时 profile 注册表，供搜索候选评估使用
    _runtime_profiles: dict[str, dict] = {}

    def __init__(self) -> None:
        # 缓存：{文件路径: (mtime, 数据)}
        self._profile_cache: dict[str, tuple[float, list[dict]]] = {}
        # search space 缓存：(mtime, 数据)
        self._search_space_cache: tuple[float, dict] | None = None

    @classmethod
    def register_runtime_profile(cls, profile: dict) -> str:
        """将 inline profile 注册到运行时临时表，返回其 id。

        若 profile 无 'id' 字段则自动生成一个。不持久化到磁盘。
        """
        import uuid as _uuid
        pid = profile.get("id") or f"_runtime_{_uuid.uuid4().hex[:8]}"
        profile = dict(profile)
        profile["id"] = pid
        cls._runtime_profiles[pid] = profile
        return pid

    @classmethod
    def unregister_runtime_profile(cls, profile_id: str) -> None:
        """从运行时临时表中移除 profile（可选清理）。"""
        cls._runtime_profiles.pop(profile_id, None)

    # ------------------------------------------------------------------
    # 内部：文件加载与热重载
    # ------------------------------------------------------------------

    def _load_json_file(self, path: Path) -> dict:
        """读取 JSON 文件，返回解析后的字典。"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_mtime(self, path: Path) -> float:
        return os.path.getmtime(path)

    def _load_profiles_from_file(self, path: Path) -> list[dict]:
        """从单个 profiles JSON 文件加载 profile 列表，带 mtime 缓存。"""
        key = str(path)
        current_mtime = self._get_mtime(path)
        cached = self._profile_cache.get(key)
        if cached is not None and cached[0] == current_mtime:
            return cached[1]
        data = self._load_json_file(path)
        profiles: list[dict] = data.get("profiles", [])
        self._profile_cache[key] = (current_mtime, profiles)
        return profiles

    def _all_profiles(self) -> list[dict]:
        """返回两个配置文件中所有 profile 的合并列表（带热重载）。"""
        temporal = self._load_profiles_from_file(_TEMPORAL_POOL_FILE)
        cross = self._load_profiles_from_file(_CROSS_SECTIONAL_FILE)
        return temporal + cross

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def load_profiles(self, mode: str) -> list[dict]:
        """加载指定 mode 的所有 profile。

        Args:
            mode: "temporal_pool" 或 "cross_sectional_filter"

        Returns:
            与 mode 匹配的 profile 列表。
        """
        return [p for p in self._all_profiles() if p.get("mode") == mode]

    def get_profile(self, profile_id: str) -> dict:
        """按 profile_id 获取单个 profile。

        查找顺序：
        1. 进程内运行时临时表（评估期间注册的 inline 候选）
        2. 磁盘配置文件（temporal_pool_profiles.json / cross_sectional_profiles.json）
        3. 持久化 inline profile 目录（data/champions/inline_profiles/）

        Args:
            profile_id: profile 的唯一标识。

        Returns:
            对应的完整 profile 配置字典。

        Raises:
            KeyError: 当 profile_id 在所有来源中均不存在时。
        """
        # 1. 进程内运行时临时表（评估期间短暂存活）
        if profile_id in self.__class__._runtime_profiles:
            return self.__class__._runtime_profiles[profile_id]

        # 2. 磁盘配置文件
        for profile in self._all_profiles():
            if profile.get("id") == profile_id:
                return profile

        # 3. 持久化 inline profile 目录（搜索产出的 champion profile）
        inline_path = _PROJECT_ROOT / "data" / "champions" / "inline_profiles" / f"{profile_id}.json"
        if inline_path.exists():
            try:
                return json.loads(inline_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"读取 inline profile 文件失败 {inline_path}: {exc}")

        raise KeyError(f"Profile '{profile_id}' not found")

    def load_search_space(self, space_id: str) -> dict:
        """加载指定 search space 配置。

        Args:
            space_id: search space 的唯一标识。

        Returns:
            对应的 search space 配置字典。

        Raises:
            KeyError: 当 space_id 不存在时。
        """
        current_mtime = self._get_mtime(_SEARCH_SPACE_FILE)
        if (
            self._search_space_cache is None
            or self._search_space_cache[0] != current_mtime
        ):
            data = self._load_json_file(_SEARCH_SPACE_FILE)
            self._search_space_cache = (current_mtime, data.get("search_spaces", {}))

        spaces: dict = self._search_space_cache[1]
        if space_id not in spaces:
            raise KeyError(f"Search space '{space_id}' not found")
        return spaces[space_id]

    def validate_profile(self, profile: dict) -> None:
        """校验 profile 合法性。

        Args:
            profile: 待校验的 profile 字典。

        Raises:
            ValueError: 当 profile 不合法时，错误信息说明具体原因。
        """
        # 1. 必填字段
        required_fields = ["id", "mode", "factors", "params"]
        missing = [f for f in required_fields if f not in profile]
        if missing:
            raise ValueError(f"Profile 缺少必填字段: {', '.join(missing)}")

        # 2. mode 合法性
        mode = profile["mode"]
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Profile mode '{mode}' 非法，必须是 {sorted(_VALID_MODES)} 之一"
            )

        # 3. signal 因子权重之和
        factors: list[dict] = profile["factors"]
        signal_weights = [
            f.get("weight", 0.0)
            for f in factors
            if f.get("role", "signal") == "signal"
        ]
        if signal_weights:
            weight_sum = sum(signal_weights)
            if not (0.999 <= weight_sum <= 1.001):
                raise ValueError(
                    f"signal 因子权重之和为 {weight_sum:.6f}，必须在 [0.999, 1.001] 范围内"
                )

        # 4. params.top_n
        params: dict = profile["params"]
        if "top_n" in params:
            top_n = params["top_n"]
            if not (1 <= top_n <= 100):
                raise ValueError(
                    f"params.top_n={top_n} 超出范围，必须在 [1, 100] 内"
                )

        # 5. params.percentile_window
        if "percentile_window" in params:
            pw = params["percentile_window"]
            if not (20 <= pw <= 504):
                raise ValueError(
                    f"params.percentile_window={pw} 超出范围，必须在 [20, 504] 内"
                )

    def list_profile_ids(self, mode: Optional[str] = None) -> list[str]:
        """列出所有 profile_id，可按 mode 过滤。

        Args:
            mode: 可选，过滤指定 mode 的 profile。为 None 时返回所有。

        Returns:
            profile_id 字符串列表。
        """
        profiles = self._all_profiles()
        if mode is not None:
            profiles = [p for p in profiles if p.get("mode") == mode]
        return [p["id"] for p in profiles if "id" in p]
