"""
公式执行结果类型定义。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class PlotSeries:
    """公式绘图输出线。"""

    name: str
    values: pd.Series
    render_type: Optional[str] = None


@dataclass(frozen=True)
class FormulaExecutionResult:
    """公式执行结果。"""

    primary: pd.Series
    plots: list[PlotSeries] = field(default_factory=list)
    formula_type: str = "mylanguage"
