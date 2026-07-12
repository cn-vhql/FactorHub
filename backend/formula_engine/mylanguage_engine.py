"""
麦语言解析与执行器。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.formula_engine.runtime import (
    FormulaExecutionError,
    FormulaRuntime,
    elementwise_binary,
    elementwise_unary,
    normalize_result,
)
from backend.formula_engine.types import FormulaExecutionResult, PlotSeries


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int


@dataclass
class ProgramNode:
    statements: List[Any]


@dataclass
class AssignmentNode:
    name: str
    expression: Any
    output: bool = False


@dataclass
class ExpressionStatementNode:
    expression: Any


@dataclass
class IdentifierNode:
    name: str


@dataclass
class LiteralNode:
    value: Any


@dataclass
class UnaryNode:
    operator: str
    operand: Any


@dataclass
class BinaryNode:
    operator: str
    left: Any
    right: Any


@dataclass
class CallNode:
    callee: Any
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexNode:
    target: Any
    index: Any


class MyLanguageTokenizer:
    """麦语言词法分析器。"""

    def tokenize(self, source: str) -> List[Token]:
        tokens: List[Token] = []
        i = 0
        bracket_depth = 0
        length = len(source)

        while i < length:
            ch = source[i]

            if ch in " \t\r":
                i += 1
                continue

            if source.startswith("//", i):
                i = self._skip_line_comment(source, i)
                continue

            if ch == "#":
                i = self._skip_line_comment(source, i)
                continue

            if ch == "{":
                i = self._skip_brace_comment(source, i)
                continue

            if ch == "\n":
                if bracket_depth == 0 and tokens and tokens[-1].kind not in {"SEMI", "LPAREN", "COMMA", "COLON", "ASSIGN"}:
                    tokens.append(Token("SEMI", ";", i))
                i += 1
                continue

            if source.startswith(":=", i):
                tokens.append(Token("ASSIGN", ":=", i))
                i += 2
                continue

            if source.startswith(">=", i) or source.startswith("<=", i) or source.startswith("==", i):
                tokens.append(Token("OP", source[i : i + 2], i))
                i += 2
                continue

            if source.startswith("!=", i) or source.startswith("<>", i):
                tokens.append(Token("OP", source[i : i + 2], i))
                i += 2
                continue

            if ch in "+-*/%^><=&|":
                tokens.append(Token("OP", ch, i))
                i += 1
                continue

            if ch == ":":
                tokens.append(Token("COLON", ":", i))
                i += 1
                continue

            if ch == ";":
                tokens.append(Token("SEMI", ";", i))
                i += 1
                continue

            if ch == ",":
                tokens.append(Token("COMMA", ",", i))
                i += 1
                continue

            if ch == "(":
                bracket_depth += 1
                tokens.append(Token("LPAREN", "(", i))
                i += 1
                continue

            if ch == ")":
                bracket_depth = max(0, bracket_depth - 1)
                tokens.append(Token("RPAREN", ")", i))
                i += 1
                continue

            if ch == "[":
                bracket_depth += 1
                tokens.append(Token("LBRACK", "[", i))
                i += 1
                continue

            if ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
                tokens.append(Token("RBRACK", "]", i))
                i += 1
                continue

            if ch in "\"'":
                token, i = self._read_string(source, i)
                tokens.append(token)
                continue

            if ch.isdigit() or (ch == "." and i + 1 < length and source[i + 1].isdigit()):
                token, i = self._read_number(source, i)
                tokens.append(token)
                continue

            if ch.isalpha() or ch == "_":
                token, i = self._read_identifier(source, i)
                tokens.append(token)
                continue

            raise FormulaExecutionError(f"麦语言词法错误：位置 {i} 存在非法字符 {ch!r}")

        tokens.append(Token("EOF", "", length))
        return tokens

    def _skip_line_comment(self, source: str, start: int) -> int:
        end = source.find("\n", start)
        return len(source) if end == -1 else end

    def _skip_brace_comment(self, source: str, start: int) -> int:
        end = source.find("}", start + 1)
        if end == -1:
            raise FormulaExecutionError("麦语言注释缺少右花括号")
        return end + 1

    def _read_string(self, source: str, start: int) -> tuple[Token, int]:
        quote = source[start]
        i = start + 1
        buffer: List[str] = []
        while i < len(source):
            ch = source[i]
            if ch == "\\" and i + 1 < len(source):
                buffer.append(source[i + 1])
                i += 2
                continue
            if ch == quote:
                return Token("STRING", "".join(buffer), start), i + 1
            buffer.append(ch)
            i += 1
        raise FormulaExecutionError("字符串缺少结束引号")

    def _read_number(self, source: str, start: int) -> tuple[Token, int]:
        i = start
        has_dot = False
        while i < len(source):
            ch = source[i]
            if ch.isdigit():
                i += 1
                continue
            if ch == "." and not has_dot:
                has_dot = True
                i += 1
                continue
            break
        return Token("NUMBER", source[start:i], start), i

    def _read_identifier(self, source: str, start: int) -> tuple[Token, int]:
        i = start
        while i < len(source) and (source[i].isalnum() or source[i] == "_"):
            i += 1
        value = source[start:i]
        upper = value.upper()
        if upper in {"AND", "OR", "NOT"}:
            return Token("OP", upper, start), i
        if upper == "TRUE":
            return Token("BOOLEAN", "TRUE", start), i
        if upper == "FALSE":
            return Token("BOOLEAN", "FALSE", start), i
        return Token("IDENT", value, start), i


class MyLanguageParser:
    """麦语言递归下降语法分析器。"""

    CALL_KEYWORDS = {
        "timeperiod",
        "fastperiod",
        "slowperiod",
        "signalperiod",
        "window",
        "min_periods",
        "nbdevup",
        "nbdevdn",
        "periods",
        "decimals",
        "dtype",
        "lower",
        "upper",
        "q",
        "period",
    }

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0

    def parse_program(self) -> ProgramNode:
        statements: List[Any] = []
        while not self._match("EOF"):
            while self._match("SEMI"):
                self._advance()
            if self._match("EOF"):
                break
            statements.append(self._parse_statement())
            while self._match("SEMI"):
                self._advance()
        return ProgramNode(statements=statements)

    def parse_expression(self) -> Any:
        return self._parse_or()

    def _parse_statement(self) -> Any:
        if self._match("IDENT") and self._peek().kind in {"ASSIGN", "COLON"}:
            name = self._advance().value
            separator = self._advance().kind
            expression = self.parse_expression()
            return AssignmentNode(name=name, expression=expression, output=(separator == "COLON"))
        return ExpressionStatementNode(expression=self.parse_expression())

    def _parse_or(self) -> Any:
        node = self._parse_and()
        while self._match_op({"OR", "|"}):
            operator = self._advance().value
            node = BinaryNode(operator=operator, left=node, right=self._parse_and())
        return node

    def _parse_and(self) -> Any:
        node = self._parse_compare()
        while self._match_op({"AND", "&"}):
            operator = self._advance().value
            node = BinaryNode(operator=operator, left=node, right=self._parse_compare())
        return node

    def _parse_compare(self) -> Any:
        node = self._parse_additive()
        while self._match_op({">", ">=", "<", "<=", "=", "==", "<>", "!="}):
            operator = self._advance().value
            node = BinaryNode(operator=operator, left=node, right=self._parse_additive())
        return node

    def _parse_additive(self) -> Any:
        node = self._parse_multiplicative()
        while self._match_op({"+", "-"}):
            operator = self._advance().value
            node = BinaryNode(operator=operator, left=node, right=self._parse_multiplicative())
        return node

    def _parse_multiplicative(self) -> Any:
        node = self._parse_unary()
        while self._match_op({"*", "/", "%"}):
            operator = self._advance().value
            node = BinaryNode(operator=operator, left=node, right=self._parse_unary())
        return node

    def _parse_unary(self) -> Any:
        if self._match_op({"+", "-", "NOT"}):
            operator = self._advance().value
            return UnaryNode(operator=operator, operand=self._parse_unary())
        return self._parse_power()

    def _parse_power(self) -> Any:
        node = self._parse_postfix()
        if self._match_op({"^"}):
            operator = self._advance().value
            node = BinaryNode(operator=operator, left=node, right=self._parse_unary())
        return node

    def _parse_postfix(self) -> Any:
        node = self._parse_primary()
        while True:
            if self._match("LPAREN"):
                node = self._parse_call(node)
                continue
            if self._match("LBRACK"):
                self._advance()
                index = self.parse_expression()
                self._expect("RBRACK")
                node = IndexNode(target=node, index=index)
                continue
            break
        return node

    def _parse_call(self, callee: Any) -> Any:
        self._expect("LPAREN")
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if not self._match("RPAREN"):
            while True:
                if (
                    self._match("IDENT")
                    and self._current().value in self.CALL_KEYWORDS
                    and self._peek().kind == "OP"
                    and self._peek().value == "="
                ):
                    key = self._advance().value
                    self._advance()
                    kwargs[key] = self.parse_expression()
                else:
                    args.append(self.parse_expression())
                if self._match("COMMA"):
                    self._advance()
                    continue
                break
        self._expect("RPAREN")
        return CallNode(callee=callee, args=args, kwargs=kwargs)

    def _parse_primary(self) -> Any:
        if self._match("NUMBER"):
            token = self._advance()
            value = float(token.value) if "." in token.value else int(token.value)
            return LiteralNode(value=value)
        if self._match("STRING"):
            return LiteralNode(value=self._advance().value)
        if self._match("BOOLEAN"):
            return LiteralNode(value=self._advance().value.upper() == "TRUE")
        if self._match("IDENT"):
            return IdentifierNode(name=self._advance().value)
        if self._match("LPAREN"):
            self._advance()
            expr = self.parse_expression()
            self._expect("RPAREN")
            return expr
        current = self._current()
        raise FormulaExecutionError(f"麦语言语法错误：位置 {current.position} 附近无法解析 {current.value!r}")

    def _current(self) -> Token:
        return self.tokens[self.position]

    def _peek(self) -> Token:
        return self.tokens[self.position + 1]

    def _advance(self) -> Token:
        token = self.tokens[self.position]
        self.position += 1
        return token

    def _expect(self, kind: str) -> Token:
        if not self._match(kind):
            current = self._current()
            raise FormulaExecutionError(f"麦语言语法错误：期望 {kind}，实际为 {current.kind}({current.value!r})")
        return self._advance()

    def _match(self, kind: str) -> bool:
        return self._current().kind == kind

    def _match_op(self, operators: set[str]) -> bool:
        return self._current().kind == "OP" and self._current().value in operators


class MyLanguageEngine:
    """麦语言执行器。"""

    def __init__(self):
        self.tokenizer = MyLanguageTokenizer()

    def parse_program(self, code: str) -> ProgramNode:
        parser = MyLanguageParser(self.tokenizer.tokenize(code))
        return parser.parse_program()

    def parse_expression(self, code: str) -> Any:
        parser = MyLanguageParser(self.tokenizer.tokenize(code))
        expression = parser.parse_expression()
        parser._expect("EOF")
        return expression

    def validate(self, code: str) -> None:
        self.parse_program(code)

    def execute(self, df: pd.DataFrame, code: str) -> pd.Series:
        return self.execute_with_metadata(df, code).primary

    def execute_with_metadata(self, df: pd.DataFrame, code: str) -> FormulaExecutionResult:
        runtime = FormulaRuntime(df.copy())
        program = self.parse_program(code)
        variables: Dict[str, Any] = {}
        last_value: Any = None
        output_statements: List[tuple[str, Any]] = []

        for statement in program.statements:
            if isinstance(statement, AssignmentNode):
                value = self._evaluate(statement.expression, runtime, variables)
                variables[statement.name] = value
                variables[statement.name.upper()] = value
                last_value = value
                if statement.output:
                    output_statements.append((statement.name, value))
            else:
                last_value = self._evaluate(statement.expression, runtime, variables)

        primary_value = self._resolve_primary_value(output_statements, last_value)
        if primary_value is None:
            raise FormulaExecutionError("麦语言公式没有产生任何结果")

        plots = [
            PlotSeries(
                name=name,
                values=normalize_result(value, runtime.index),
                render_type=self._infer_render_type(name),
            )
            for name, value in output_statements
        ]
        return FormulaExecutionResult(
            primary=normalize_result(primary_value, runtime.index),
            plots=plots,
            formula_type="mylanguage",
        )

    def _resolve_primary_value(self, output_statements: List[tuple[str, Any]], last_value: Any) -> Any:
        for name, value in output_statements:
            if name.upper() == "XG":
                return value
        if output_statements:
            return output_statements[-1][1]
        return last_value

    def _infer_render_type(self, name: str) -> Optional[str]:
        upper_name = name.upper()
        if upper_name in {"MACD", "HIST", "HISTOGRAM", "BAR"} or "HIST" in upper_name:
            return "bar"
        return "line"

    def _evaluate(self, node: Any, runtime: FormulaRuntime, variables: Dict[str, Any]) -> Any:
        if isinstance(node, LiteralNode):
            return node.value

        if isinstance(node, IdentifierNode):
            return runtime.resolve_name(node.name, variables)

        if isinstance(node, UnaryNode):
            return elementwise_unary(node.operator, self._evaluate(node.operand, runtime, variables))

        if isinstance(node, BinaryNode):
            left = self._evaluate(node.left, runtime, variables)
            right = self._evaluate(node.right, runtime, variables)
            return elementwise_binary(node.operator, left, right, runtime.index)

        if isinstance(node, CallNode):
            callee = self._evaluate(node.callee, runtime, variables)
            if not callable(callee):
                raise FormulaExecutionError("麦语言函数调用目标不可执行")
            args = [self._evaluate(arg, runtime, variables) for arg in node.args]
            kwargs = {key: self._evaluate(value, runtime, variables) for key, value in node.kwargs.items()}
            return callee(*args, **kwargs)

        if isinstance(node, IndexNode):
            target = self._evaluate(node.target, runtime, variables)
            index = self._evaluate(node.index, runtime, variables)
            if isinstance(index, pd.Series):
                valid = index.dropna()
                if valid.empty:
                    raise FormulaExecutionError("下标表达式不能为空序列")
                index = valid.iloc[-1]
            idx = int(index)
            if isinstance(target, (list, tuple)):
                return target[idx]
            raise FormulaExecutionError("麦语言仅支持对列表或元组结果进行下标访问")

        raise FormulaExecutionError(f"不支持的麦语言节点类型: {type(node)}")
