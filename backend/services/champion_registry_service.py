"""
ChampionRegistryService: 以文件方式持久化 champion 配置。

存储路径：
- data/champions/single_stock/{code}.json
- data/champions/stock_group/{group_hash}.json
- data/champions/cross_sectional/{universe}.json
"""

import hashlib
import json
from pathlib import Path


# scope_type 到子目录的映射
_SCOPE_DIR_MAP = {
    "single_stock": "single_stock",
    "stock_group": "stock_group",
    "cross_sectional_universe": "cross_sectional",
}

_BASE_DIR = Path("data/champions")


class ChampionRegistryService:
    """文件方式持久化 champion 配置，不依赖数据库。"""

    def __init__(self, base_dir: Path | None = None):
        self._base_dir = base_dir or _BASE_DIR

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _validate_scope_key(self, scope_key: str) -> None:
        """路径安全校验：拒绝含 '/' 或 '..' 的 scope_key。"""
        if "/" in scope_key or ".." in scope_key:
            raise ValueError(
                f"scope_key 包含非法字符（'/' 或 '..'）：{scope_key!r}"
            )

    def _resolve_path(self, scope_type: str, scope_key: str) -> Path:
        """根据 scope_type 和 scope_key 解析文件路径。"""
        if scope_type not in _SCOPE_DIR_MAP:
            raise ValueError(
                f"未知 scope_type：{scope_type!r}，"
                f"合法值为 {list(_SCOPE_DIR_MAP.keys())}"
            )
        self._validate_scope_key(scope_key)
        subdir = _SCOPE_DIR_MAP[scope_type]
        return self._base_dir / subdir / f"{scope_key}.json"

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def save(self, scope_type: str, scope_key: str, champion: dict) -> Path:
        """保存 champion 配置，返回写入文件的 Path 对象。

        Args:
            scope_type: "single_stock" | "stock_group" | "cross_sectional_universe"
            scope_key: 作用域标识（股票代码 / group_hash / universe 名称）
            champion: champion 配置字典

        Returns:
            写入文件的 Path 对象

        Raises:
            ValueError: scope_key 含非法字符或 scope_type 非法
            IOError: 目录无写权限或磁盘满
        """
        path = self._resolve_path(scope_type, scope_key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(champion, ensure_ascii=False, indent=2), encoding="utf-8")
        except PermissionError as exc:
            raise IOError(f"无法写入 {path}：{exc}") from exc
        return path

    def load(self, scope_type: str, scope_key: str) -> dict | None:
        """加载 champion 配置，文件不存在时返回 None。

        Args:
            scope_type: "single_stock" | "stock_group" | "cross_sectional_universe"
            scope_key: 作用域标识

        Returns:
            champion 配置字典，或 None（文件不存在时）
        """
        path = self._resolve_path(scope_type, scope_key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_champions(self, scope_type: str | None = None) -> list[dict]:
        """列出所有 champion，可按 scope_type 过滤。

        每个返回的 champion 字典中包含 `scope_type` 字段。

        Args:
            scope_type: 若指定，则仅返回该 scope_type 的 champion；
                        若为 None，则返回所有 scope_type 的 champion。

        Returns:
            champion 字典列表
        """
        scope_types = (
            [scope_type] if scope_type is not None else list(_SCOPE_DIR_MAP.keys())
        )

        results: list[dict] = []
        for st in scope_types:
            subdir = _SCOPE_DIR_MAP.get(st)
            if subdir is None:
                continue
            dir_path = self._base_dir / subdir
            if not dir_path.exists():
                continue
            for json_file in sorted(dir_path.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    # 注入 scope_type 字段，方便调用方识别
                    data["scope_type"] = st
                    results.append(data)
                except (json.JSONDecodeError, OSError):
                    # 跳过损坏的文件，不影响整体结果
                    continue
        return results

    def delete(self, scope_type: str, scope_key: str) -> bool:
        """删除 champion 配置文件。

        Args:
            scope_type: "single_stock" | "stock_group" | "cross_sectional_universe"
            scope_key: 作用域标识

        Returns:
            True 表示删除成功，False 表示文件不存在（不抛异常）
        """
        path = self._resolve_path(scope_type, scope_key)
        if not path.exists():
            return False
        path.unlink()
        return True

    # ------------------------------------------------------------------
    # 静态工具
    # ------------------------------------------------------------------

    @staticmethod
    def _scope_key_to_hash(codes: list[str]) -> str:
        """将股票列表转为稳定 hash（排序后 MD5），确保顺序无关。

        Args:
            codes: 股票代码列表（顺序任意）

        Returns:
            32 位小写十六进制 MD5 字符串
        """
        sorted_codes = sorted(codes)
        content = ",".join(sorted_codes)
        return hashlib.md5(content.encode("utf-8")).hexdigest()
