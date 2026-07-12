"""
受限 Python 指标执行器。
"""
from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from backend.formula_engine.runtime import (
    FormulaExecutionError,
    FormulaRuntime,
    ensure_bool_scalar,
    elementwise_binary,
    elementwise_unary,
    normalize_result,
    resolve_attribute,
)
from backend.formula_engine.types import FormulaExecutionResult


class SafePythonEvaluator:
    """安全 Python 表达式求值器。"""

    def __init__(self, runtime: FormulaRuntime, variables: Optional[Dict[str, Any]] = None):
        self.runtime = runtime
        self.variables = variables or {}

    def evaluate_expression(self, expression: str) -> Any:
        tree = ast.parse(expression, mode="eval")
        return self.visit(tree.body)

    def evaluate_node(self, node: ast.AST) -> Any:
        return self.visit(node)

    def visit(self, node: ast.AST) -> Any:
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise FormulaExecutionError(f"Python 指标不支持 AST 节点: {type(node).__name__}")
        return method(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        return self.runtime.resolve_name(node.id, self.variables)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        return elementwise_binary(self._binop(node.op), left, right, self.runtime.index)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        values = [self.visit(value) for value in node.values]
        operator = "AND" if isinstance(node.op, ast.And) else "OR"
        result = values[0]
        for value in values[1:]:
            result = elementwise_binary(operator, result, value, self.runtime.index)
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        return elementwise_unary(self._unary(node.op), value)

    def visit_Compare(self, node: ast.Compare) -> Any:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise FormulaExecutionError("Python 指标暂不支持链式比较")
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        return elementwise_binary(self._compare(node.ops[0]), left, right, self.runtime.index)

    def visit_Call(self, node: ast.Call) -> Any:
        callee = self.visit(node.func)
        if not callable(callee):
            raise FormulaExecutionError("Python 指标调用目标不可执行")
        args = [self.visit(arg) for arg in node.args]
        kwargs = {keyword.arg: self.visit(keyword.value) for keyword in node.keywords}
        return callee(*args, **kwargs)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        target = self.visit(node.value)
        return resolve_attribute(target, node.attr)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        target = self.visit(node.value)
        key = self._resolve_subscript_key(node.slice)

        if isinstance(target, pd.DataFrame):
            return target[key]
        if isinstance(target, pd.Series):
            if isinstance(key, slice):
                return target.iloc[key]
            if isinstance(key, int):
                return target.iloc[key]
            return target.loc[key]
        if isinstance(target, (tuple, list, dict)):
            return target[key]
        if hasattr(target, "__getitem__"):
            return target[key]
        raise FormulaExecutionError(f"对象 {type(target)} 不支持下标访问")

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(element) for element in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(element) for element in node.elts)

    def visit_Dict(self, node: ast.Dict) -> Any:
        return {self.visit(key): self.visit(value) for key, value in zip(node.keys, node.values)}

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        condition = ensure_bool_scalar(self.visit(node.test))
        return self.visit(node.body) if condition else self.visit(node.orelse)

    def _resolve_subscript_key(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Slice):
            lower = self.visit(node.lower) if node.lower else None
            upper = self.visit(node.upper) if node.upper else None
            step = self.visit(node.step) if node.step else None
            return slice(lower, upper, step)
        return self.visit(node)

    def _binop(self, operator: ast.AST) -> str:
        mapping = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.Pow: "^",
            ast.BitAnd: "&",
            ast.BitOr: "|",
        }
        for ast_type, symbol in mapping.items():
            if isinstance(operator, ast_type):
                return symbol
        raise FormulaExecutionError(f"Python 指标不支持二元运算符: {type(operator).__name__}")

    def _unary(self, operator: ast.AST) -> str:
        if isinstance(operator, ast.UAdd):
            return "+"
        if isinstance(operator, ast.USub):
            return "-"
        if isinstance(operator, ast.Not):
            return "NOT"
        raise FormulaExecutionError(f"Python 指标不支持一元运算符: {type(operator).__name__}")

    def _compare(self, operator: ast.AST) -> str:
        mapping = {
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Eq: "==",
            ast.NotEq: "!=",
        }
        for ast_type, symbol in mapping.items():
            if isinstance(operator, ast_type):
                return symbol
        raise FormulaExecutionError(f"Python 指标不支持比较运算符: {type(operator).__name__}")


class SafePythonEngine:
    """支持受限 Python 表达式和函数指标。"""

    SAFE_GLOBALS = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "int": int,
        "float": float,
        "bool": bool,
        "str": str,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "range": range,
    }

    def validate(self, code: str) -> None:
        stripped = code.strip()
        if not stripped:
            raise FormulaExecutionError("Python 指标代码不能为空")
        if stripped.startswith("def "):
            ast.parse(code, mode="exec")
        else:
            ast.parse(code, mode="eval")

    def execute(self, df: pd.DataFrame, code: str) -> pd.Series:
        return self.execute_with_metadata(df, code).primary

    def execute_with_metadata(self, df: pd.DataFrame, code: str) -> FormulaExecutionResult:
        runtime = FormulaRuntime(df.copy())
        stripped = code.strip()

        if stripped.startswith("def "):
            value = self._execute_function(runtime, code)
        else:
            evaluator = SafePythonEvaluator(runtime, variables=self._initial_variables(runtime))
            value = evaluator.evaluate_expression(code)

        return FormulaExecutionResult(
            primary=normalize_result(value, runtime.index),
            plots=[],
            formula_type="python",
        )

    def _initial_variables(self, runtime: FormulaRuntime) -> Dict[str, Any]:
        variables = dict(self.SAFE_GLOBALS)
        variables.update(runtime.base_variables)
        variables.update(runtime.safe_modules)
        variables.update(runtime.function_registry)
        return variables

    def _execute_function(self, runtime: FormulaRuntime, code: str) -> Any:
        module = ast.parse(code, mode="exec")
        function = None
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                function = node
                break
        if function is None or function.name != "calculate_factor":
            raise FormulaExecutionError("Python 指标必须定义 calculate_factor(df) 函数")
        if len(function.args.args) != 1 or function.args.args[0].arg != "df":
            raise FormulaExecutionError("calculate_factor 函数必须仅接收一个 df 参数")

        variables = self._initial_variables(runtime)
        return self._execute_statements(runtime, variables, function.body)

    def _execute_statements(self, runtime: FormulaRuntime, variables: Dict[str, Any], statements: Iterable[ast.stmt]) -> Any:
        evaluator = SafePythonEvaluator(runtime, variables)

        for statement in statements:
            if isinstance(statement, ast.Expr):
                if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                    continue
                evaluator.evaluate_node(statement.value)
                continue

            if isinstance(statement, ast.Import):
                self._handle_import(statement, variables)
                continue

            if isinstance(statement, ast.ImportFrom):
                raise FormulaExecutionError("Python 指标不允许使用 from ... import ...")

            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    raise FormulaExecutionError("Python 指标仅支持简单变量赋值")
                variables[statement.targets[0].id] = evaluator.evaluate_node(statement.value)
                continue

            if isinstance(statement, ast.AugAssign):
                if not isinstance(statement.target, ast.Name):
                    raise FormulaExecutionError("Python 指标仅支持简单变量增强赋值")
                current = variables.get(statement.target.id)
                if current is None:
                    raise FormulaExecutionError(f"变量 {statement.target.id} 未定义")
                variables[statement.target.id] = elementwise_binary(
                    evaluator._binop(statement.op),
                    current,
                    evaluator.evaluate_node(statement.value),
                    runtime.index,
                )
                continue

            if isinstance(statement, ast.Return):
                return evaluator.evaluate_node(statement.value)

            if isinstance(statement, ast.If):
                branch = statement.body if ensure_bool_scalar(evaluator.evaluate_node(statement.test)) else statement.orelse
                result = self._execute_statements(runtime, variables, branch)
                if result is not None:
                    return result
                continue

            if isinstance(statement, ast.Try):
                try:
                    result = self._execute_statements(runtime, variables, statement.body)
                    if result is not None:
                        return result
                    if statement.orelse:
                        result = self._execute_statements(runtime, variables, statement.orelse)
                        if result is not None:
                            return result
                except Exception as exc:  # noqa: BLE001 - 这里需要显式捕获公式异常
                    handled = False
                    for handler in statement.handlers:
                        handled = True
                        if handler.name:
                            variables[handler.name] = exc
                        result = self._execute_statements(runtime, variables, handler.body)
                        if result is not None:
                            return result
                        break
                    if not handled:
                        raise
                finally:
                    if statement.finalbody:
                        result = self._execute_statements(runtime, variables, statement.finalbody)
                        if result is not None:
                            return result
                continue

            if isinstance(statement, ast.Pass):
                continue

            raise FormulaExecutionError(f"Python 指标不支持语句类型: {type(statement).__name__}")

        return None

    def _handle_import(self, statement: ast.Import, variables: Dict[str, Any]) -> None:
        for alias in statement.names:
            if alias.name == "pandas":
                variables[alias.asname or "pandas"] = variables["pd"]
                continue
            if alias.name == "numpy":
                variables[alias.asname or "numpy"] = variables["np"]
                continue
            raise FormulaExecutionError(f"Python 指标不允许导入模块: {alias.name}")
