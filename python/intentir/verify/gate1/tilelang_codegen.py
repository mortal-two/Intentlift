"""Extraction and fail-closed policy for model-generated TileLang Replay."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import NoReturn

from .codegen import CodeExtractionError, extract_python_code


class UnsafeTileLangReplayCodeError(ValueError):
    """The module is outside the reviewed public TileLang Python subset."""

    def __init__(
        self,
        message: str,
        *,
        rule_code: str = "tilelang_static_policy_rejected",
        line: int | None = None,
    ) -> None:
        self.rule_code = rule_code
        self.line = line
        super().__init__(message)


@dataclass(frozen=True)
class TileLangModuleValidation:
    function_name: str
    tree: ast.Module


_FORBIDDEN_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Match,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.Yield,
    ast.YieldFrom,
)
_SAFE_BUILTINS = {"bool", "float", "int", "range"}
_FORBIDDEN_ATTRIBUTES = {
    "chmod",
    "connect",
    "dump",
    "dumps",
    "exec",
    "load",
    "loads",
    "mkdir",
    "open",
    "popen",
    "read",
    "read_bytes",
    "read_text",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "send",
    "socket",
    "system",
    "unlink",
    "write",
    "write_bytes",
    "write_text",
}


def _fail(
    node: ast.AST | None,
    message: str,
    *,
    rule_code: str = "tilelang_static_policy_rejected",
) -> NoReturn:
    line = getattr(node, "lineno", None) if node is not None else None
    if line is not None:
        message = f"line {line}: {message}"
    raise UnsafeTileLangReplayCodeError(
        message,
        rule_code=rule_code,
        line=line,
    )


def _is_t_attribute(node: ast.AST, name: str | None = None) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "T"
        and (name is None or node.attr == name)
    )


def _literal_or_schedule_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (bool, int, float, str, type(None)))
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_literal_or_schedule_expr(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub)) and _literal_or_schedule_expr(node.operand)
    if isinstance(node, ast.BinOp):
        return isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod),
        ) and _literal_or_schedule_expr(node.left) and _literal_or_schedule_expr(node.right)
    if isinstance(node, ast.Call):
        return _is_t_attribute(node.func, "dynamic") and all(
            _literal_or_schedule_expr(item) for item in node.args
        )
    if isinstance(node, ast.Name):
        return node.id.isupper()
    return False


def validate_tilelang_module(
    code: str,
    *,
    required_function: str,
) -> TileLangModuleValidation:
    """Validate one direct ``@T.prim_func`` module without executing it."""

    if not isinstance(code, str) or not code.strip():
        raise UnsafeTileLangReplayCodeError("TileLang Replay module is empty")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeTileLangReplayCodeError(f"invalid Python: {exc}") from exc

    functions: list[ast.FunctionDef] = []
    import_seen = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            if len(node.names) != 1:
                _fail(node, "TileLang Replay permits one canonical import")
            alias = node.names[0]
            if alias.name != "tilelang.language" or alias.asname != "T":
                _fail(node, "use exactly `import tilelang.language as T`")
            import_seen = True
            continue
        if isinstance(node, ast.ImportFrom):
            if (
                node.level != 0
                or node.module != "tilelang"
                or len(node.names) != 1
                or node.names[0].name != "language"
                or node.names[0].asname != "T"
            ):
                _fail(node, "unsupported TileLang Replay import")
            import_seen = True
            continue
        if isinstance(node, ast.AnnAssign):
            _fail(
                node,
                "annotated top-level TileLang schedule bindings are forbidden",
                rule_code=(
                    "tilelang_module_binding_requires_uppercase_literal_or_T_dynamic"
                ),
            )
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
            if (
                len(targets) != 1
                or not isinstance(targets[0], ast.Name)
                or not targets[0].id.isupper()
                or value is None
                or not _literal_or_schedule_expr(value)
            ):
                _fail(
                    node,
                    "top-level TileLang values require an uppercase target and a "
                    "literal or T.dynamic value; T.symbolic is forbidden",
                    rule_code=(
                        "tilelang_module_binding_requires_uppercase_literal_or_T_dynamic"
                    ),
                )
            continue
        if isinstance(node, ast.FunctionDef):
            functions.append(node)
            continue
        _fail(node, "TileLang Replay top level permits only import, constants and one PrimFunc")

    if not import_seen:
        raise UnsafeTileLangReplayCodeError("TileLang Replay lacks its canonical import")
    if len(functions) != 1 or functions[0].name != required_function:
        raise UnsafeTileLangReplayCodeError(
            "TileLang Replay must define exactly the canonical PrimFunc"
        )
    function = functions[0]
    if len(function.decorator_list) != 1 or not _is_t_attribute(
        function.decorator_list[0], "prim_func"
    ):
        _fail(function, "TileLang Replay function must use only @T.prim_func")
    if function.args.vararg is not None or function.args.kwarg is not None:
        _fail(function, "variadic TileLang PrimFunc is forbidden")
    if (
        function.args.posonlyargs
        or function.args.kwonlyargs
        or function.args.defaults
        or any(item is not None for item in function.args.kw_defaults)
        or function.returns is not None
    ):
        _fail(
            function,
            "TileLang PrimFunc permits only ordinary annotated parameters without defaults",
            rule_code="tilelang_primfunc_signature_without_defaults",
        )

    for item in ast.walk(function):
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            _fail(
                item,
                "imports inside the TileLang PrimFunc are forbidden",
                rule_code="tilelang_inner_import_forbidden",
            )
        if isinstance(item, _FORBIDDEN_NODES):
            _fail(item, f"{type(item).__name__} is forbidden in TileLang Replay")
        if isinstance(item, ast.Name) and "__" in item.id:
            _fail(item, "dunder names are forbidden")
        if isinstance(item, ast.Attribute):
            if "__" in item.attr or item.attr in _FORBIDDEN_ATTRIBUTES:
                _fail(item, f"attribute {item.attr!r} is forbidden")
        if isinstance(item, ast.Call):
            target = item.func
            if isinstance(target, ast.Name):
                if target.id not in _SAFE_BUILTINS:
                    _fail(target, f"unqualified call {target.id!r} is forbidden")
            elif not _is_t_attribute(target):
                _fail(target, "TileLang calls must use the public T namespace")
    return TileLangModuleValidation(required_function, tree)


def extract_tilelang_module(response: str, *, required_function: str) -> str:
    try:
        code = extract_python_code(response)
    except CodeExtractionError as exc:
        raise UnsafeTileLangReplayCodeError(str(exc)) from exc
    validate_tilelang_module(code, required_function=required_function)
    return code


__all__ = [
    "TileLangModuleValidation",
    "UnsafeTileLangReplayCodeError",
    "extract_tilelang_module",
    "validate_tilelang_module",
]
