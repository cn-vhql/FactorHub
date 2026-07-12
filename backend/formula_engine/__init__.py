"""
统一公式引擎出口。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from backend.formula_engine.mylanguage_engine import MyLanguageEngine
from backend.formula_engine.python_engine import SafePythonEngine
from backend.formula_engine.runtime import (
    FormulaExecutionError,
    infer_formula_type,
    normalize_formula_type,
)
from backend.formula_engine.types import FormulaExecutionResult, PlotSeries


@dataclass
class FormulaEngineManager:
    """公式执行管理器。"""

    mylanguage_engine: MyLanguageEngine = MyLanguageEngine()
    python_engine: SafePythonEngine = SafePythonEngine()

    def infer_formula_type(self, code: str) -> str:
        return infer_formula_type(code)

    def validate(self, code: str, formula_type: Optional[str] = None) -> str:
        engine_type = normalize_formula_type(formula_type, code)
        self._select_engine(engine_type).validate(code)
        return engine_type

    def execute(self, df: pd.DataFrame, code: str, formula_type: Optional[str] = None) -> pd.Series:
        engine_type = normalize_formula_type(formula_type, code)
        return self._select_engine(engine_type).execute(df, code)

    def execute_with_metadata(
        self,
        df: pd.DataFrame,
        code: str,
        formula_type: Optional[str] = None,
    ) -> FormulaExecutionResult:
        engine_type = normalize_formula_type(formula_type, code)
        return self._select_engine(engine_type).execute_with_metadata(df, code)

    def parse_mylanguage_expression(self, expression: str) -> Any:
        return self.mylanguage_engine.parse_expression(expression)

    def parse_mylanguage_program(self, code: str) -> Any:
        return self.mylanguage_engine.parse_program(code)

    def _select_engine(self, engine_type: str) -> Any:
        if engine_type == "python":
            return self.python_engine
        if engine_type == "mylanguage":
            return self.mylanguage_engine
        raise FormulaExecutionError(f"未知公式类型: {engine_type}")


formula_engine_manager = FormulaEngineManager()

__all__ = [
    "FormulaEngineManager",
    "FormulaExecutionResult",
    "PlotSeries",
    "formula_engine_manager",
]
