import ast
import math
import operator
from collections.abc import Callable

from openai import OpenAI

from app.config import Settings
from app.rag.retrieval import retrieve_final_chunks


Number = int | float


class CalculationError(ValueError):
    pass


BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

ALLOWED_FUNCTIONS: dict[str, Callable[..., Number]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


def document_search(query: str, settings: Settings) -> list[dict]:
    chunks = retrieve_final_chunks(query=query, settings=settings)

    return [
        {
            "source_id": chunk.source_id,
            "filename": chunk.filename,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "score": chunk.score,
            "retrieval_method": chunk.retrieval_method,
        }
        for chunk in chunks
    ]


def web_search(query: str, settings: Settings) -> dict:
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.responses.create(
        model=settings.openai_chat_model,
        tools=[
            {
                "type": "web_search",
            }
        ],
        input=(
            "Search the web for reliable information related to this query. "
            "Return concise findings and include source references when available.\n\n"
            f"Query: {query}"
        ),
    )

    return {
        "status": "ok",
        "query": query,
        "result": response.output_text,
    }


def calculator(expression: str) -> dict:
    try:
        result = _calculate_expression(expression)
        return {"expression": expression, "result": result}
    except Exception as error:
        return {"expression": expression, "error": str(error)}


def _calculate_expression(expression: str) -> Number:
    if len(expression) > 500:
        raise CalculationError("Expression is too long.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise CalculationError("Expression is not valid arithmetic.") from error

    result = _evaluate_node(tree.body)

    if not math.isfinite(float(result)):
        raise CalculationError("Expression produced a non-finite result.")

    if isinstance(result, float) and result.is_integer():
        return int(result)

    return result


def _evaluate_node(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise CalculationError("Only numeric literals are allowed.")
        return node.value

    if isinstance(node, ast.BinOp):
        operator_func = BINARY_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise CalculationError("Unsupported arithmetic operator.")

        try:
            return operator_func(
                _evaluate_node(node.left),
                _evaluate_node(node.right),
            )
        except ZeroDivisionError as error:
            raise CalculationError("Division by zero.") from error

    if isinstance(node, ast.UnaryOp):
        operator_func = UNARY_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise CalculationError("Unsupported unary operator.")
        return operator_func(_evaluate_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculationError("Unsupported function.")

        function = ALLOWED_FUNCTIONS.get(node.func.id)
        if function is None:
            raise CalculationError(f"Unsupported function '{node.func.id}'.")

        if node.keywords:
            raise CalculationError("Keyword arguments are not supported.")

        return function(*[_evaluate_node(arg) for arg in node.args])

    raise CalculationError("Unsupported expression element.")
