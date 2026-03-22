from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def bootstrap_runtime(load_preset_factors: bool = True) -> None:
    """确保脚本独立运行时具备数据库和预置因子。"""
    from backend.core.database import init_db

    init_db()
    if not load_preset_factors:
        return

    try:
        from backend.services.factor_service import factor_service
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "运行该 stage 缺少必要依赖，请先安装 TA-Lib 等量化计算依赖。"
        ) from exc

    factor_service.load_preset_factors()


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: str | Path, data: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_csv(path: str | Path, rows: list[dict] | pd.DataFrame) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def write_text(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("至少需要提供一个整数")
    return values


def normalize_stock_code(code: str) -> str:
    value = str(code).strip()
    if value.endswith(".0"):
        value = value[:-2]
    if value.isdigit():
        value = value.zfill(6)
    return value


def load_codes_from_csv(csv_path: str | Path, code_column: str = "code") -> list[str]:
    df = pd.read_csv(csv_path, dtype=str)
    if code_column not in df.columns:
        raise ValueError(f"CSV 缺少列: {code_column}")
    return [
        normalize_stock_code(code)
        for code in df[code_column].dropna().astype(str).tolist()
    ]


def safe_slug(raw: str) -> str:
    value = raw.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    return "".join(ch for ch in value if ch.isalnum() or ch in {"_", "-", "."}) or "run"
