"""Safe arithmetic-expression evaluator (no eval, AST-walked)."""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from tools.schema import ToolContext, ToolSpec

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UN_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "min": min, "max": max,
}
_NAMES = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"only numeric constants allowed (got {type(node.value).__name__})")
    if isinstance(node, ast.Num):  # py<3.8 backcompat
        return node.n
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        return op(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary op not allowed: {type(node.op).__name__}")
        return op(_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _NAMES:
            return _NAMES[node.id]
        raise ValueError(f"unknown name: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("only whitelisted functions allowed")
        if node.keywords:
            raise ValueError("kwargs not allowed")
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"node not allowed: {type(node).__name__}")


def safe_eval(expression: str) -> float | int:
    if len(expression) > 200:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")
    return _eval(tree)


async def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    expr = (args.get("expression") or "").strip()
    if not expr:
        return {"error": "expression required"}
    try:
        result = safe_eval(expr)
    except Exception as e:  # noqa: BLE001
        return {"error": f"calc_error: {e}"}
    return {"expression": expr, "result": result}


TOOL = ToolSpec(
    name="calculator",
    description=(
        "Evaluate a numeric/math expression. Supports + - * / // % **, "
        "parentheses, and the functions sqrt, log, log10, exp, sin, cos, tan, "
        "abs, round, floor, ceil, min, max. Constants pi, e."
    ),
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": False,
    },
    handler=_run,
)
