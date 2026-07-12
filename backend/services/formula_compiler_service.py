"""
公式编译器服务 - 可视化因子构建
"""
import ast
import re
from typing import Dict, List, Optional, Any, Tuple

from backend.formula_engine import formula_engine_manager
from backend.formula_engine.mylanguage_engine import (
    BinaryNode,
    CallNode,
    IdentifierNode,
    IndexNode,
    LiteralNode,
    UnaryNode,
)


class FormulaCompilerService:
    """公式编译器服务类 - 将可视化公式树编译为可执行代码"""

    # 支持的因子元素
    AVAILABLE_ELEMENTS = {
        # 价格数据
        "price": {
            "open": "开盘价",
            "high": "最高价",
            "low": "最低价",
            "close": "收盘价",
            "volume": "成交量",
            "amount": "成交额",
            "turnover": "换手率",
            "amplitude": "振幅",
            "pct_change": "涨跌幅",
            "date": "交易日期",
        },
        # 技术指标
        "indicators": {
            "MA": "简单移动平均",
            "SMA": "简单移动平均",
            "EMA": "指数移动平均",
            "RSI": "相对强弱指标",
            "MACD": "MACD",
            "ADX": "平均趋向指标",
            "CCI": "顺势指标",
            "ATR": "平均真实波幅",
            "BBANDS": "布林带",
            "OBV": "能量潮",
            "REF": "引用历史值",
            "HHV": "区间最高值",
            "LLV": "区间最低值",
            "RANGEPOS": "区间分位",
            "IF": "条件分支",
            "CROSS": "上穿信号",
        },
        # 运算符
        "operators": {
            "+": "加",
            "-": "减",
            "*": "乘",
            "/": "除",
            ">": "大于",
            "<": "小于",
            ">=": "大于等于",
            "<=": "小于等于",
            "==": "等于",
            "!=": "不等于",
            "AND": "且",
            "OR": "或",
        },
        # 统计函数
        "statistics": {
            "mean": "均值",
            "std": "标准差",
            "max": "最大值",
            "min": "最小值",
            "median": "中位数",
            "rank": "排名",
            "zscore": "Z-score标准化",
        },
    }

    def __init__(self):
        pass

    def compile_formula(self, formula_tree: Dict) -> str:
        """
        将公式树编译为可执行代码

        Args:
            formula_tree: 公式树结构

        Returns:
            可执行的Python代码

        公式树示例:
        {
            "type": "operation",
            "operator": "/",
            "left": {
                "type": "column",
                "value": "close"
            },
            "right": {
                "type": "function",
                "name": "SMA",
                "args": [
                    {
                        "type": "column",
                        "value": "close"
                    },
                    {
                        "type": "literal",
                        "value": 20
                    }
                ]
            }
        }
        """
        try:
            code = self._compile_node(formula_tree)

            # 包装为完整的表达式
            return code

        except Exception as e:
            raise ValueError(f"公式编译失败: {e}")

    def _compile_node(self, node: Dict) -> str:
        """递归编译节点"""
        node_type = node.get("type")

        if node_type == "column":
            # 数据列
            return node["value"]

        elif node_type == "literal":
            # 字面量
            value = node["value"]
            if isinstance(value, str):
                return f'"{value}"'
            return str(value)

        elif node_type == "attribute":
            target = self._compile_node(node["target"])
            return f"{target}.{node['name']}"

        elif node_type == "index":
            target = self._compile_node(node["target"])
            index = self._compile_node(node["index"])
            return f"{target}[{index}]"

        elif node_type == "method":
            target = self._compile_node(node["target"])
            method_name = node["name"]
            args = node.get("args", [])
            kwargs = node.get("kwargs", {})
            compiled_args = [self._compile_node(arg) for arg in args]
            compiled_kwargs = [
                f"{key}={self._compile_node(value)}"
                for key, value in kwargs.items()
            ]
            all_args = compiled_args + compiled_kwargs
            return f"{target}.{method_name}({', '.join(all_args)})"

        elif node_type == "function":
            # 函数调用
            func_name = node["name"]
            args = node.get("args", [])
            kwargs = node.get("kwargs", {})

            # 编译参数
            compiled_args = [self._compile_node(arg) for arg in args]
            compiled_kwargs = {
                key: self._compile_node(value)
                for key, value in kwargs.items()
            }

            # 特殊处理技术指标
            if func_name in ["MA", "SMA", "EMA", "RSI", "MACD", "ADX", "CCI", "ATR", "BBANDS", "OBV"]:
                if func_name == "MA":
                    self._require_positional_args(func_name, compiled_args, 1)
                    self._require_args(func_name, compiled_args, 2, compiled_kwargs)
                    return self._build_function_call(
                        "MA",
                        [compiled_args[0]],
                        {"timeperiod": compiled_kwargs.get("timeperiod", compiled_args[1])},
                    )
                if func_name == "SMA":
                    self._require_positional_args(func_name, compiled_args, 1)
                    if "timeperiod" in compiled_kwargs:
                        self._require_args(func_name, compiled_args, 1, compiled_kwargs)
                        return self._build_function_call(
                            "SMA",
                            [compiled_args[0]],
                            {"timeperiod": compiled_kwargs["timeperiod"]},
                        )
                    self._require_args(func_name, compiled_args, 2, compiled_kwargs)
                    return self._build_function_call(
                        "SMA",
                        [compiled_args[0]],
                        {"timeperiod": compiled_args[1]},
                    )
                elif func_name == "EMA":
                    self._require_positional_args(func_name, compiled_args, 1)
                    if "timeperiod" in compiled_kwargs:
                        self._require_args(func_name, compiled_args, 1, compiled_kwargs)
                        return self._build_function_call(
                            "EMA",
                            [compiled_args[0]],
                            {"timeperiod": compiled_kwargs["timeperiod"]},
                        )
                    self._require_args(func_name, compiled_args, 2, compiled_kwargs)
                    return self._build_function_call(
                        "EMA",
                        [compiled_args[0]],
                        {"timeperiod": compiled_args[1]},
                    )
                elif func_name == "RSI":
                    self._require_positional_args(func_name, compiled_args, 1)
                    if "timeperiod" in compiled_kwargs:
                        self._require_args(func_name, compiled_args, 1, compiled_kwargs)
                        return self._build_function_call(
                            "RSI",
                            [compiled_args[0]],
                            {"timeperiod": compiled_kwargs["timeperiod"]},
                        )
                    self._require_args(func_name, compiled_args, 2, compiled_kwargs)
                    return self._build_function_call(
                        "RSI",
                        [compiled_args[0]],
                        {"timeperiod": compiled_args[1]},
                    )
                elif func_name == "MACD":
                    self._require_positional_args(func_name, compiled_args, 1)
                    self._require_args(func_name, compiled_args, 1, compiled_kwargs)
                    fastperiod = compiled_kwargs.get("fastperiod", compiled_args[1] if len(compiled_args) > 1 else "12")
                    slowperiod = compiled_kwargs.get("slowperiod", compiled_args[2] if len(compiled_args) > 2 else "26")
                    signalperiod = compiled_kwargs.get("signalperiod", compiled_args[3] if len(compiled_args) > 3 else "9")
                    return (
                        self._build_function_call(
                            "MACD",
                            [compiled_args[0]],
                            {
                                "fastperiod": fastperiod,
                                "slowperiod": slowperiod,
                                "signalperiod": signalperiod,
                            },
                        )
                        + "[0]"
                    )
                elif func_name == "BBANDS":
                    self._require_positional_args(func_name, compiled_args, 1)
                    self._require_args(func_name, compiled_args, 1, compiled_kwargs)
                    timeperiod = compiled_kwargs.get("timeperiod", compiled_args[1] if len(compiled_args) > 1 else "20")
                    return self._build_function_call(
                        "BBANDS",
                        [compiled_args[0]],
                        {"timeperiod": timeperiod},
                    ) + "[1]"
                elif func_name == "ATR":
                    self._require_positional_args(func_name, compiled_args, 3)
                    self._require_args(func_name, compiled_args, 3, compiled_kwargs)
                    timeperiod = compiled_kwargs.get("timeperiod", compiled_args[3] if len(compiled_args) > 3 else None)
                    if timeperiod is None:
                        raise ValueError("函数 ATR 至少需要 4 个参数，或显式传入 timeperiod")
                    return self._build_function_call(
                        "ATR",
                        [compiled_args[0], compiled_args[1], compiled_args[2]],
                        {"timeperiod": timeperiod},
                    )
                elif func_name == "ADX":
                    self._require_positional_args(func_name, compiled_args, 3)
                    self._require_args(func_name, compiled_args, 3, compiled_kwargs)
                    timeperiod = compiled_kwargs.get("timeperiod", compiled_args[3] if len(compiled_args) > 3 else None)
                    if timeperiod is None:
                        raise ValueError("函数 ADX 至少需要 4 个参数，或显式传入 timeperiod")
                    return self._build_function_call(
                        "ADX",
                        [compiled_args[0], compiled_args[1], compiled_args[2]],
                        {"timeperiod": timeperiod},
                    )
                elif func_name == "CCI":
                    self._require_positional_args(func_name, compiled_args, 3)
                    self._require_args(func_name, compiled_args, 3, compiled_kwargs)
                    timeperiod = compiled_kwargs.get("timeperiod", compiled_args[3] if len(compiled_args) > 3 else None)
                    if timeperiod is None:
                        raise ValueError("函数 CCI 至少需要 4 个参数，或显式传入 timeperiod")
                    return self._build_function_call(
                        "CCI",
                        [compiled_args[0], compiled_args[1], compiled_args[2]],
                        {"timeperiod": timeperiod},
                    )
                elif func_name == "OBV":
                    self._require_positional_args(func_name, compiled_args, 2)
                    self._require_args(func_name, compiled_args, 2, compiled_kwargs)
                    return self._build_function_call(
                        "OBV",
                        [compiled_args[0], compiled_args[1]],
                    )
                else:
                    return self._build_function_call(func_name, compiled_args, compiled_kwargs)
            elif func_name in ["mean", "std", "max", "min"]:
                self._require_args(func_name, compiled_args, 1, compiled_kwargs)
                if len(compiled_args) >= 2:
                    return f"({compiled_args[0]}).rolling(window={compiled_args[1]}, min_periods=1).{func_name}()"
                return f"({compiled_args[0]}).{func_name}()"
            elif func_name == "median":
                self._require_args(func_name, compiled_args, 1, compiled_kwargs)
                if len(compiled_args) >= 2:
                    return f"({compiled_args[0]}).rolling(window={compiled_args[1]}, min_periods=1).median()"
                return f"({compiled_args[0]}).median()"
            elif func_name == "rank":
                self._require_args(func_name, compiled_args, 1, compiled_kwargs)
                return f"({compiled_args[0]}).rank(pct=True)"
            elif func_name == "zscore":
                self._require_args(func_name, compiled_args, 1, compiled_kwargs)
                if len(compiled_args) >= 2:
                    return (
                        f"(({compiled_args[0]}) - ({compiled_args[0]}).rolling(window={compiled_args[1]}, min_periods=1).mean()) / "
                        f"(({compiled_args[0]}).rolling(window={compiled_args[1]}, min_periods=1).std() + 1e-8)"
                    )
                return f"(({compiled_args[0]}) - ({compiled_args[0]}).mean()) / (({compiled_args[0]}).std() + 1e-8)"
            else:
                return self._build_function_call(func_name, compiled_args, compiled_kwargs)

        elif node_type == "operation":
            # 运算符
            operator = node["operator"]
            left = self._compile_node(node["left"])
            right = self._compile_node(node["right"])

            return f"({left} {operator} {right})"

        else:
            raise ValueError(f"未知的节点类型: {node_type}")

    def validate_formula(self, formula_code: str) -> Tuple[bool, str]:
        """
        验证公式代码

        Args:
            formula_code: 公式代码

        Returns:
            (是否有效, 错误消息)
        """
        try:
            resolved_type = formula_engine_manager.validate(formula_code)
            return True, f"验证通过（类型: {resolved_type}）"

        except Exception as e:
            return False, f"验证失败：{e}"

    def parse_expression(self, expression: str) -> Dict:
        """
        解析表达式为公式树

        Args:
            expression: 表达式字符串

        Returns:
            公式树
        """
        try:
            resolved_type = formula_engine_manager.infer_formula_type(expression)
            if resolved_type == "mylanguage":
                node = formula_engine_manager.parse_mylanguage_expression(expression)
                return self._mylanguage_ast_to_formula_tree(node)

            parsed = ast.parse(expression, mode="eval")
            return self._ast_to_formula_tree(parsed.body)

        except Exception as e:
            raise ValueError(f"表达式解析失败：{e}")

    def _mylanguage_ast_to_formula_tree(self, node: Any) -> Dict:
        """将麦语言 AST 节点转换为公式树。"""
        if isinstance(node, IdentifierNode):
            return {"type": "column", "value": node.name}

        if isinstance(node, LiteralNode):
            return {"type": "literal", "value": node.value}

        if isinstance(node, BinaryNode):
            return {
                "type": "operation",
                "operator": node.operator,
                "left": self._mylanguage_ast_to_formula_tree(node.left),
                "right": self._mylanguage_ast_to_formula_tree(node.right),
            }

        if isinstance(node, UnaryNode):
            operand = self._mylanguage_ast_to_formula_tree(node.operand)
            if node.operator == "-":
                return {
                    "type": "operation",
                    "operator": "*",
                    "left": {"type": "literal", "value": -1},
                    "right": operand,
                }
            raise ValueError(f"暂不支持的一元运算: {node.operator}")

        if isinstance(node, CallNode):
            if not isinstance(node.callee, IdentifierNode):
                raise ValueError("仅支持函数名直接调用")
            args = [self._mylanguage_ast_to_formula_tree(arg) for arg in node.args]
            return {
                "type": "function",
                "name": node.callee.name,
                "args": args,
                "kwargs": {
                    key: self._mylanguage_ast_to_formula_tree(value)
                    for key, value in node.kwargs.items()
                },
            }

        if isinstance(node, IndexNode):
            return {
                "type": "index",
                "target": self._mylanguage_ast_to_formula_tree(node.target),
                "index": self._mylanguage_ast_to_formula_tree(node.index),
            }

        raise ValueError(f"不支持的麦语言 AST 节点: {type(node)}")

    def _ast_to_formula_tree(self, node: ast.AST) -> Dict:
        """将AST节点转换为公式树节点"""
        if isinstance(node, ast.Name):
            # 变量名（如 close, open）
            return {"type": "column", "value": node.id}

        elif isinstance(node, ast.Constant):
            # 常量
            return {"type": "literal", "value": node.value}

        elif isinstance(node, ast.BinOp):
            # 二元运算
            return {
                "type": "operation",
                "operator": self._ast_op_to_str(node.op),
                "left": self._ast_to_formula_tree(node.left),
                "right": self._ast_to_formula_tree(node.right),
            }

        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("暂不支持链式比较表达式")
            return {
                "type": "operation",
                "operator": self._ast_op_to_str(node.ops[0]),
                "left": self._ast_to_formula_tree(node.left),
                "right": self._ast_to_formula_tree(node.comparators[0]),
            }

        elif isinstance(node, ast.UnaryOp):
            operand = self._ast_to_formula_tree(node.operand)
            if isinstance(node.op, ast.USub):
                return {
                    "type": "operation",
                    "operator": "*",
                    "left": {"type": "literal", "value": -1},
                    "right": operand,
                }
            raise ValueError(f"不支持的一元运算: {type(node.op)}")

        elif isinstance(node, ast.Call):
            # 函数调用
            kwargs = {
                keyword.arg: self._ast_to_formula_tree(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            if isinstance(node.func, ast.Attribute):
                return {
                    "type": "method",
                    "target": self._ast_to_formula_tree(node.func.value),
                    "name": node.func.attr,
                    "args": [self._ast_to_formula_tree(arg) for arg in node.args],
                    "kwargs": kwargs,
                }

            func_name = node.func.id if isinstance(node.func, ast.Name) else str(node.func)
            args = [self._ast_to_formula_tree(arg) for arg in node.args]

            return {
                "type": "function",
                "name": func_name,
                "args": args,
                "kwargs": kwargs,
            }

        elif isinstance(node, ast.Attribute):
            return {
                "type": "attribute",
                "target": self._ast_to_formula_tree(node.value),
                "name": node.attr,
            }

        elif isinstance(node, ast.Subscript):
            return {
                "type": "index",
                "target": self._ast_to_formula_tree(node.value),
                "index": self._ast_to_formula_tree(node.slice),
            }

        elif isinstance(node, ast.BoolOp):
            operator = "AND" if isinstance(node.op, ast.And) else "OR"
            current = self._ast_to_formula_tree(node.values[0])
            for value in node.values[1:]:
                current = {
                    "type": "operation",
                    "operator": operator,
                    "left": current,
                    "right": self._ast_to_formula_tree(value),
                }
            return current

        else:
            raise ValueError(f"不支持的AST节点类型: {type(node)}")

    def _ast_op_to_str(self, op: ast.AST) -> str:
        """将AST运算符转换为字符串"""
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.Pow: "**",
            ast.Gt: ">",
            ast.Lt: "<",
            ast.GtE: ">=",
            ast.LtE: "<=",
            ast.Eq: "==",
            ast.NotEq: "!=",
        }
        return op_map.get(type(op), str(op))

    def get_available_elements(self) -> Dict:
        """
        获取可用的因子元素

        Returns:
            可用元素字典
        """
        return self.AVAILABLE_ELEMENTS

    def simplify_formula(self, formula_code: str) -> str:
        """
        规范化公式代码

        Args:
            formula_code: 原始公式代码

        Returns:
            规范化后的代码
        """
        stripped = formula_code.strip()
        if not stripped:
            return ""

        if stripped.startswith("def "):
            lines = [line.rstrip() for line in stripped.splitlines()]
            normalized_lines = []
            previous_blank = False
            for line in lines:
                is_blank = not line.strip()
                if is_blank and previous_blank:
                    continue
                normalized_lines.append(line)
                previous_blank = is_blank
            return "\n".join(normalized_lines).strip()

        try:
            return ast.unparse(ast.parse(stripped, mode="eval"))
        except Exception:
            lines = [line.strip() for line in stripped.split("\n") if line.strip()]
            return " ".join(lines)

    def _build_function_call(self, func_name: str, args: List[str], kwargs: Optional[Dict[str, str]] = None) -> str:
        parts = list(args)
        if kwargs:
            parts.extend(f"{key}={value}" for key, value in kwargs.items())
        return f"{func_name}({', '.join(parts)})"

    def _require_args(
        self,
        func_name: str,
        args: List[str],
        min_args: int,
        kwargs: Optional[Dict[str, str]] = None,
    ) -> None:
        """校验函数最少参数个数。"""
        kwargs = kwargs or {}
        if len(args) + len(kwargs) < min_args:
            raise ValueError(
                f"函数 {func_name} 至少需要 {min_args} 个参数，实际收到 {len(args)} 个位置参数和 {len(kwargs)} 个关键字参数"
            )

    def _require_positional_args(self, func_name: str, args: List[str], min_args: int) -> None:
        if len(args) < min_args:
            raise ValueError(f"函数 {func_name} 至少需要 {min_args} 个位置参数，实际收到 {len(args)} 个")


# 全局公式编译器服务实例
formula_compiler_service = FormulaCompilerService()
