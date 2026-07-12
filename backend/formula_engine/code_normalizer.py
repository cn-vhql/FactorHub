"""
公式代码规范化工具。
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Any

from backend.formula_engine import formula_engine_manager
from backend.formula_engine.mylanguage_engine import (
    AssignmentNode,
    BinaryNode,
    CallNode,
    ExpressionStatementNode,
    IdentifierNode,
    IndexNode,
    LiteralNode,
    UnaryNode,
)
from backend.formula_engine.runtime import infer_formula_type


MYLANGUAGE_OUTPUT_PATTERN = re.compile(r"(^|[;\n])\s*[A-Za-z_][A-Za-z0-9_]*\s*:", re.MULTILINE)


def normalize_formula_code(code: str, formula_type: Optional[str] = None) -> str:
    """将混合公式代码规范化为可执行表达式。"""
    stripped = (code or "").strip()
    if not stripped:
        return stripped

    resolved_type = formula_type or infer_formula_type(stripped)
    if resolved_type == "mylanguage":
        primary = extract_mylanguage_primary_expression(stripped)
        return primary or stripped

    normalized = _normalize_legacy_python_escapes(stripped)
    while True:
        replaced = _replace_embedded_mylanguage_segments(normalized)
        if replaced == normalized:
            break
        normalized = replaced
    return normalized


def extract_mylanguage_primary_expression(code: str) -> Optional[str]:
    """提取麦语言程序中的主输出表达式。"""
    stripped = (code or "").strip()
    if not stripped:
        return None

    program = formula_engine_manager.parse_mylanguage_program(stripped)
    assignments: Dict[str, str] = {}
    output_candidates: list[tuple[str, str]] = []
    last_expression: Optional[str] = None

    for statement in program.statements:
        if isinstance(statement, AssignmentNode):
            rendered = _render_mylanguage_expression(statement.expression, assignments)
            assignments[statement.name] = rendered
            assignments[statement.name.upper()] = rendered
            last_expression = rendered
            if statement.output:
                output_candidates.append((statement.name, rendered))
        elif isinstance(statement, ExpressionStatementNode):
            last_expression = _render_mylanguage_expression(statement.expression, assignments)

    for name, rendered in output_candidates:
        if name.upper() == "XG":
            return rendered
    if output_candidates:
        return output_candidates[-1][1]
    return last_expression


def _replace_embedded_mylanguage_segments(code: str) -> str:
    replacements: list[tuple[int, int, str]] = []
    stack: list[int] = []

    for index, char in enumerate(code):
        if char == "(":
            stack.append(index)
            continue
        if char != ")" or not stack:
            continue

        start = stack.pop()
        inner = code[start + 1 : index].strip()
        if not _looks_like_mylanguage_program(inner):
            continue

        try:
            primary = extract_mylanguage_primary_expression(inner)
        except Exception:  # noqa: BLE001 - 非法片段直接跳过
            continue

        if not primary:
            continue
        replacements.append((start, index + 1, f"({primary})"))

    if not replacements:
        return code

    result = code
    for start, end, replacement in reversed(replacements):
        result = result[:start] + replacement + result[end:]
    return result


def _looks_like_mylanguage_program(code: str) -> bool:
    return ":=" in code or bool(MYLANGUAGE_OUTPUT_PATTERN.search(code))


def _normalize_legacy_python_escapes(code: str) -> str:
    """清理历史错误保存的 Python 转义字符。"""
    normalized = code
    if '\\"' in normalized:
        normalized = normalized.replace('\\"', '"')
    if "\\'" in normalized:
        normalized = normalized.replace("\\'", "'")
    return normalized


def _render_mylanguage_expression(node: Any, assignments: Dict[str, str]) -> str:
    if isinstance(node, LiteralNode):
        return repr(node.value)

    if isinstance(node, IdentifierNode):
        resolved = assignments.get(node.name) or assignments.get(node.name.upper())
        return f"({resolved})" if resolved is not None else node.name

    if isinstance(node, UnaryNode):
        operand = _render_mylanguage_expression(node.operand, assignments)
        if node.operator == "+":
            return f"(+{operand})"
        if node.operator == "-":
            return f"(-{operand})"
        raise ValueError(f"不支持的一元运算符: {node.operator}")

    if isinstance(node, BinaryNode):
        operator = {
            "=": "==",
            "<>": "!=",
            "AND": "&",
            "OR": "|",
            "^": "**",
        }.get(node.operator, node.operator)
        left = _render_mylanguage_expression(node.left, assignments)
        right = _render_mylanguage_expression(node.right, assignments)
        return f"({left} {operator} {right})"

    if isinstance(node, CallNode):
        callee = _render_mylanguage_expression(node.callee, assignments)
        args = [_render_mylanguage_expression(arg, assignments) for arg in node.args]
        kwargs = [
            f"{key}={_render_mylanguage_expression(value, assignments)}"
            for key, value in node.kwargs.items()
        ]
        return f"{callee}({', '.join(args + kwargs)})"

    if isinstance(node, IndexNode):
        target = _render_mylanguage_expression(node.target, assignments)
        index = _render_mylanguage_expression(node.index, assignments)
        return f"{target}[{index}]"

    raise ValueError(f"不支持的麦语言节点类型: {type(node)}")
