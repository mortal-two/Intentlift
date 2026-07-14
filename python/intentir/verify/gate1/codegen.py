"""Safe extraction of Triton replay modules from model responses.

Gate1 deliberately compiles code produced by an untrusted model.  Compiling a
``@triton.jit`` function executes the surrounding Python module and executes
parts of the function through Triton's Python frontend, so a Markdown fence is
not a security boundary.  This module therefore accepts a deliberately small
Python subset before replay code is handed to the isolated compiler process.

The accepted subset is sufficient for ordinary, self-contained Triton kernels:

* imports are limited to ``triton`` and ``triton.language as tl``;
* every top-level definition is decorated only with ``@triton.jit``;
* function calls are limited to public ``tl`` functions, ``range`` and simple
  scalar constructors (plus Triton's tensor ``.to`` conversion);
* Python facilities that can perform host I/O, reflection, dynamic execution,
  non-terminating loops, or hidden imports are rejected.

This is a policy boundary, not a general-purpose Python sandbox.  Resource
limits and process isolation are still enforced by :mod:`compiler`.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import NoReturn

__all__ = [
    "CodeExtractionError",
    "ModuleValidation",
    "UnsafeReplayCodeError",
    "extract_python_code",
    "extract_python_module",
    "validate_replay_module",
    "validate_triton_module",
]


class CodeExtractionError(ValueError):
    """The model response does not contain one unambiguous Python module."""


class UnsafeReplayCodeError(ValueError):
    """The replay module is outside Gate1's safe Triton-Python subset."""


@dataclass(frozen=True)
class ModuleValidation:
    """Result of validating one replay module."""

    function_names: tuple[str, ...]
    tree: ast.Module


_FENCE_RE = re.compile(
    r"```[ \t]*(?P<label>[A-Za-z0-9_+.-]*)[ \t]*\r?\n"
    r"(?P<body>.*?)```",
    re.DOTALL,
)
_PYTHON_FENCE_LABELS = {"", "py", "python", "python3"}

# These are the only non-Triton call targets admitted in a kernel body.  They
# are constructors/iteration helpers with no I/O or reflection capability.
_SAFE_BUILTIN_CALLS = {"bool", "float", "int", "range"}

# Public ``tl`` calls are generally dataflow-building operations.  The handful
# below are intentionally excluded because they accept executable/opaque code
# or dynamically named symbols and would defeat structural review.
_FORBIDDEN_TL_CALLS = {
    "inline_asm_elementwise",
    "extern_elementwise",
}

# A conservative statement/expression allowlist.  Context and operator marker
# nodes are included because ``ast.walk`` yields them independently.
_ALLOWED_BODY_NODES = (
    ast.Expr,
    ast.Constant,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.For,
    ast.If,
    ast.Return,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Attribute,
    ast.Call,
    ast.keyword,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Starred,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.MatMult,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
)


def _fail(node: ast.AST | None, message: str) -> NoReturn:
    if node is not None and getattr(node, "lineno", None) is not None:
        message = f"line {node.lineno}: {message}"
    raise UnsafeReplayCodeError(message)


def extract_python_code(response: str) -> str:
    """Extract one Python module from a raw or Markdown-fenced model response.

    A response with code fences must contain exactly one Python (or unlabeled)
    fence.  Refusing ambiguous multi-block answers prevents a caller from
    accidentally compiling a different block than the one it audited.
    """

    if not isinstance(response, str) or not response.strip():
        raise CodeExtractionError("model response is empty")

    matches = list(_FENCE_RE.finditer(response))
    if matches:
        candidates = [
            match.group("body")
            for match in matches
            if match.group("label").lower() in _PYTHON_FENCE_LABELS
        ]
        if not candidates:
            raise CodeExtractionError("model response has no Python code fence")
        if len(candidates) != 1:
            raise CodeExtractionError(
                f"model response has {len(candidates)} Python code fences; expected one"
            )
        code = candidates[0]
    else:
        # Raw mode is useful for providers that return the code payload without
        # presentation Markdown.  Any prose will naturally fail Python parsing.
        code = response

    code = code.lstrip("\ufeff").strip()
    if not code:
        raise CodeExtractionError("extracted Python module is empty")
    return code + "\n"


def extract_python_module(
    response: str, *, required_function: str | None = None
) -> str:
    """Extract and validate a model-generated Triton module."""

    code = extract_python_code(response)
    validate_triton_module(code, required_function=required_function)
    return code


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_attr(node: ast.AST, root: str, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and _is_name(node.value, root)
    )


def _check_identifier(identifier: str, node: ast.AST, what: str) -> None:
    if "__" in identifier:
        _fail(node, f"dunder {what} {identifier!r} is forbidden")


def _validate_import(node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "triton" and alias.asname is None:
                continue
            if alias.name == "triton.language" and alias.asname == "tl":
                continue
            _fail(
                node,
                "imports are limited to `import triton` and "
                "`import triton.language as tl`",
            )
        return

    if (
        node.module == "triton"
        and node.level == 0
        and len(node.names) == 1
        and node.names[0].name == "language"
        and node.names[0].asname == "tl"
    ):
        return
    _fail(node, "from-imports are limited to `from triton import language as tl`")


def _validate_annotation(node: ast.AST | None) -> None:
    if node is None:
        return
    if not _is_attr(node, "tl", "constexpr"):
        _fail(node, "only `tl.constexpr` parameter annotations are allowed")


def _validate_jit_decorator(node: ast.AST, arg_names: set[str]) -> None:
    if _is_attr(node, "triton", "jit"):
        return
    if not (
        isinstance(node, ast.Call) and _is_attr(node.func, "triton", "jit")
    ):
        _fail(node, "every function must be decorated only with `@triton.jit`")
    if node.args:
        _fail(node, "`@triton.jit` positional arguments are forbidden")
    seen: set[str] = set()
    for keyword in node.keywords:
        if keyword.arg != "do_not_specialize":
            _fail(keyword, "only `do_not_specialize` is allowed in `@triton.jit`")
        if keyword.arg in seen:
            _fail(keyword, "duplicate `do_not_specialize` argument")
        seen.add(keyword.arg)
        try:
            value = ast.literal_eval(keyword.value)
        except (TypeError, ValueError, SyntaxError):
            _fail(keyword.value, "`do_not_specialize` must be a literal list/tuple")
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, (str, int)) for item in value
        ):
            _fail(keyword.value, "invalid `do_not_specialize` value")
        for item in value:
            if isinstance(item, str) and item not in arg_names:
                _fail(keyword.value, f"unknown `do_not_specialize` argument {item!r}")
            if isinstance(item, int) and not 0 <= item < len(arg_names):
                _fail(keyword.value, f"invalid `do_not_specialize` index {item}")


def _stored_names(nodes: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for statement in nodes:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
    return names


def _validate_call(node: ast.Call) -> None:
    target = node.func
    if isinstance(target, ast.Name):
        if target.id not in _SAFE_BUILTIN_CALLS:
            _fail(node, f"call to Python name {target.id!r} is forbidden")
        return

    if isinstance(target, ast.Attribute):
        _check_identifier(target.attr, target, "attribute")
        if _is_name(target.value, "tl"):
            if target.attr.startswith("_") or target.attr in _FORBIDDEN_TL_CALLS:
                _fail(node, f"call to `tl.{target.attr}` is forbidden")
            return
        # Triton tensors expose ``.to(dtype)`` as a frontend conversion.  No
        # other arbitrary Python method dispatch is permitted.
        if target.attr == "to" and len(node.args) <= 1 and not node.keywords:
            return
    _fail(node, "calls must target a public `tl` function, range/scalar constructor, or `.to`")


def _validate_loop(node: ast.For) -> None:
    iterator = node.iter
    if not isinstance(iterator, ast.Call):
        _fail(iterator, "for-loops must iterate over `range`, `tl.range`, or `tl.static_range`")
    target = iterator.func
    valid = _is_name(target, "range") or (
        isinstance(target, ast.Attribute)
        and _is_name(target.value, "tl")
        and target.attr in {"range", "static_range"}
    )
    if not valid:
        _fail(iterator, "for-loops must iterate over `range`, `tl.range`, or `tl.static_range`")
    if node.orelse:
        _fail(node, "for-else is not allowed in replay kernels")


def _validate_function(node: ast.FunctionDef) -> None:
    _check_identifier(node.name, node, "function name")
    args = node.args
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg:
        _fail(node, "positional-only, variadic, and keyword-only parameters are forbidden")
    if args.defaults or args.kw_defaults:
        _fail(node, "function parameter defaults are forbidden")
    if node.returns is not None or node.type_comment is not None:
        _fail(node, "return and type-comment annotations are forbidden")

    arg_names = {arg.arg for arg in args.args}
    for arg in args.args:
        _check_identifier(arg.arg, arg, "parameter")
        _validate_annotation(arg.annotation)
    if len(node.decorator_list) != 1:
        _fail(node, "every function must have exactly one `@triton.jit` decorator")
    _validate_jit_decorator(node.decorator_list[0], arg_names)

    local_names = arg_names | _stored_names(node.body)
    allowed_load_names = local_names | _SAFE_BUILTIN_CALLS | {"tl"}

    for statement in node.body:
        for child in ast.walk(statement):
            if isinstance(child, ast.FunctionDef):
                _fail(child, "nested functions are forbidden")
            if not isinstance(child, _ALLOWED_BODY_NODES):
                _fail(child, f"Python construct {type(child).__name__} is forbidden")
            if isinstance(child, ast.Name):
                _check_identifier(child.id, child, "name")
                if isinstance(child.ctx, ast.Load) and child.id not in allowed_load_names:
                    _fail(child, f"global/free name {child.id!r} is forbidden")
            elif isinstance(child, ast.Attribute):
                _check_identifier(child.attr, child, "attribute")
                if _is_name(child.value, "tl"):
                    if child.attr.startswith("_"):
                        _fail(child, f"private Triton attribute {child.attr!r} is forbidden")
                elif child.attr != "to":
                    _fail(child, f"attribute access {child.attr!r} is forbidden")
            elif isinstance(child, ast.Call):
                _validate_call(child)
                if any(keyword.arg is None for keyword in child.keywords):
                    _fail(child, "dynamic **keyword expansion is forbidden")
            elif isinstance(child, ast.For):
                _validate_loop(child)
            elif isinstance(child, ast.Constant):
                if child.value is not None and not isinstance(
                    child.value, (bool, int, float, str)
                ):
                    _fail(child, f"literal type {type(child.value).__name__} is forbidden")
                if isinstance(child.value, str) and "__" in child.value:
                    _fail(child, "dunder strings are forbidden")


def validate_triton_module(
    code: str, *, required_function: str | None = None
) -> ModuleValidation:
    """Validate ``code`` against Gate1's strict Triton replay policy.

    The function raises :class:`UnsafeReplayCodeError` for both syntax errors
    and policy violations.  Returning the parsed tree lets callers avoid a
    second parse when they need function metadata.
    """

    if not isinstance(code, str) or not code.strip():
        raise UnsafeReplayCodeError("replay module is empty")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        where = f"line {exc.lineno}: " if exc.lineno is not None else ""
        raise UnsafeReplayCodeError(f"{where}invalid Python syntax: {exc.msg}") from exc

    functions: list[str] = []
    saw_triton = False
    saw_tl = False
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            if "__" in node.value.value:
                _fail(node, "dunder strings are forbidden")
            continue  # module docstring
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _validate_import(node)
            if isinstance(node, ast.Import):
                saw_triton |= any(alias.name == "triton" for alias in node.names)
                saw_tl |= any(alias.name == "triton.language" for alias in node.names)
            else:
                saw_tl = True
            continue
        if isinstance(node, ast.FunctionDef):
            _validate_function(node)
            if node.name in functions:
                _fail(node, f"duplicate function definition {node.name!r}")
            functions.append(node.name)
            continue
        _fail(node, "top level may contain only Triton imports and `@triton.jit` functions")

    if not saw_triton:
        raise UnsafeReplayCodeError("module must contain `import triton`")
    if not saw_tl:
        raise UnsafeReplayCodeError("module must import `triton.language as tl`")
    if not functions:
        raise UnsafeReplayCodeError("module contains no `@triton.jit` function")
    if required_function is not None and required_function not in functions:
        raise UnsafeReplayCodeError(
            f"required Triton function {required_function!r} is missing"
        )
    return ModuleValidation(tuple(functions), tree)


# Naming used by the Gate1 design document and early callers.
validate_replay_module = validate_triton_module
