import ast
import math

from langchain_core.tools import tool
from simpleeval import DEFAULT_OPERATORS, InvalidExpression, SimpleEval, safe_power

from app.tools.registry import registry


_MAX_EXPRESSION_LENGTH = 1000
_MAX_AST_NODES = 128
_MAX_EXPONENT = 1000
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.UAdd,
    ast.USub,
)


def _bounded_power(base: int | float, exponent: int | float) -> int | float:
    if abs(exponent) > _MAX_EXPONENT:
        raise ValueError(f"Exponent magnitude must not exceed {_MAX_EXPONENT}")
    result = safe_power(base, exponent)
    if not isinstance(result, (int, float)):
        raise ValueError("Only real numeric results are supported")
    return result


@registry.register
@tool
def calculate_metric(expression: str) -> str:
    """Calculate numeric expressions using +, -, *, /, **, %, and parentheses."""
    try:
        if len(expression) > _MAX_EXPRESSION_LENGTH:
            raise ValueError("Expression is too long")
        parsed = ast.parse(expression.strip(), mode="eval")
        nodes = list(ast.walk(parsed))
        if len(nodes) > _MAX_AST_NODES:
            raise ValueError("Expression is too complex")
        # Validate the entire tree before evaluating even a single subexpression.
        for node in nodes:
            if not isinstance(node, _ALLOWED_NODES):
                raise ValueError(
                    "Only numeric literals and arithmetic operators are allowed"
                )
            if isinstance(node, ast.Constant):
                if type(node.value) not in (int, float):
                    raise ValueError("Only real numeric literals are allowed")
                if isinstance(node.value, float) and not math.isfinite(node.value):
                    raise ValueError("Numeric literals must be finite")

        operators = DEFAULT_OPERATORS.copy()
        operators[ast.Pow] = _bounded_power
        evaluator = SimpleEval(operators=operators, names={}, functions={})
        result = evaluator.eval(expression, previously_parsed=parsed.body)
        if not isinstance(result, (int, float)):
            raise ValueError("Only real numeric results are supported")
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Calculation result must be finite")
        return f"Calculation result: {result}"
    except (
        ArithmeticError,
        InvalidExpression,
        RecursionError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as exc:
        return f"Calculation error: {exc!s}"
