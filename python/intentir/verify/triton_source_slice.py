"""Deterministic fixed-C source projection for real Triton kernel modules.

The projection is Harness-side plumbing.  It selects one top-level kernel and
the module declarations required to load it, while replacing source-side
autotune/heuristic selection with the concrete schedule already bound by the
Harness.  It never invents an operator, predicate, or implementation strategy.
"""

from __future__ import annotations

import ast
import builtins
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TRITON_SOURCE_SLICE_PROTOCOL = "intentir.triton.fixed-c-source-slice.v1"


class TritonSourceSliceError(ValueError):
    """The requested real-source projection cannot be formed unambiguously."""


def _annotation_is_constexpr(annotation: ast.expr | None) -> bool:
    return annotation is not None and ast.unparse(annotation).endswith(
        "constexpr"
    )


def _decorator_name(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    try:
        return ast.unparse(target)
    except ValueError:
        return "<unknown>"


def _is_jit_decorator(node: ast.expr) -> bool:
    return _decorator_name(node) in {"triton.jit", "jit"}


def _is_schedule_decorator(node: ast.expr) -> bool:
    return _decorator_name(node) in {
        "triton.autotune",
        "triton.heuristics",
        "autotune",
        "heuristics",
    }


class _BoundNameCollector(ast.NodeVisitor):
    """Collect names bound by one module statement, without entering scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(
            alias.asname or alias.name.split(".")[0] for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)


def _bound_names(node: ast.AST) -> set[str]:
    collector = _BoundNameCollector()
    collector.visit(node)
    return collector.names


class _LoadNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


def _load_names(node: ast.AST) -> set[str]:
    collector = _LoadNameCollector()
    collector.visit(node)
    return collector.names


def _function_dependencies(function: ast.FunctionDef) -> set[str]:
    local_names = {
        function.name,
        *(
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        ),
    }
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local_names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node is not function:
                local_names.add(node.name)
    retained_decorators = [
        decorator
        for decorator in function.decorator_list
        if not _is_schedule_decorator(decorator)
    ]
    projection = copy.deepcopy(function)
    projection.decorator_list = retained_decorators
    return _load_names(projection) - local_names - set(dir(builtins))


def _statement_dependencies(statement: ast.stmt) -> set[str]:
    if isinstance(statement, ast.FunctionDef):
        return _function_dependencies(statement)
    return _load_names(statement) - _bound_names(statement) - set(dir(builtins))


@dataclass(frozen=True)
class TritonSourceSlice:
    code: str
    function: str
    parameters: tuple[str, ...]
    constexpr_parameters: tuple[str, ...]
    included_declarations: tuple[str, ...]
    included_functions: tuple[str, ...]
    removed_decorators: tuple[str, ...]
    inlined_imports: tuple[str, ...] = ()
    protocol: str = TRITON_SOURCE_SLICE_PROTOCOL

    def validate_abi(
        self,
        *,
        parameter_order: Iterable[str],
        constexpr_parameters: Iterable[str],
    ) -> None:
        expected_order = tuple(parameter_order)
        expected_constexpr = tuple(constexpr_parameters)
        if expected_order != self.parameters:
            raise TritonSourceSliceError(
                "Harness parameter order differs from the selected source ABI: "
                f"{expected_order!r} != {self.parameters!r}"
            )
        if expected_constexpr != self.constexpr_parameters:
            raise TritonSourceSliceError(
                "Harness constexpr set/order differs from the selected source ABI: "
                f"{expected_constexpr!r} != {self.constexpr_parameters!r}"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "function": self.function,
            "parameters": list(self.parameters),
            "constexpr_parameters": list(self.constexpr_parameters),
            "included_declarations": list(self.included_declarations),
            "included_functions": list(self.included_functions),
            "removed_decorators": list(self.removed_decorators),
            "inlined_imports": list(self.inlined_imports),
        }


def build_triton_source_slice(
    source: str,
    *,
    source_symbol: str,
) -> TritonSourceSlice:
    """Select one real kernel and its mechanically required module closure."""

    if not source_symbol.isidentifier():
        raise TritonSourceSliceError("source symbol must be a Python identifier")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise TritonSourceSliceError(f"source cannot be parsed: {exc}") from exc
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == source_symbol
    ]
    if len(matches) != 1:
        raise TritonSourceSliceError(
            f"source symbol {source_symbol!r} must resolve exactly once at module "
            f"top level; found {len(matches)}"
        )
    target = matches[0]
    if not any(_is_jit_decorator(item) for item in target.decorator_list):
        raise TritonSourceSliceError("selected source function is not @triton.jit")
    unexpected = [
        _decorator_name(item)
        for item in target.decorator_list
        if not _is_jit_decorator(item) and not _is_schedule_decorator(item)
    ]
    if unexpected:
        raise TritonSourceSliceError(
            "selected kernel has unsupported non-schedule decorators: "
            + ", ".join(unexpected)
        )

    bindings: dict[str, list[int]] = {}
    for index, statement in enumerate(tree.body):
        for name in _bound_names(statement):
            bindings.setdefault(name, []).append(index)
    selected: set[int] = {tree.body.index(target)}
    pending = list(_function_dependencies(target))
    unresolved: set[str] = set()
    while pending:
        name = pending.pop()
        owners = bindings.get(name, [])
        if not owners:
            unresolved.add(name)
            continue
        if len(owners) != 1:
            raise TritonSourceSliceError(
                f"module dependency {name!r} is bound by multiple top-level statements"
            )
        owner = owners[0]
        if owner in selected:
            continue
        selected.add(owner)
        pending.extend(_statement_dependencies(tree.body[owner]))
    # Names supplied by Triton's compile-time language may legitimately have no
    # Python declaration (for example function-local type properties).  Every
    # module-global call/helper/import, however, must have resolved above.
    called_names = {
        call.func.id
        for call in ast.walk(target)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    unresolved_calls = sorted(unresolved & called_names)
    if unresolved_calls:
        raise TritonSourceSliceError(
            "selected kernel has unresolved called globals: "
            + ", ".join(unresolved_calls)
        )

    projected_body: list[ast.stmt] = []
    removed: list[str] = []
    included_declarations: list[str] = []
    included_functions: list[str] = []
    for index, statement in enumerate(tree.body):
        if index not in selected:
            continue
        item = copy.deepcopy(statement)
        names = sorted(_bound_names(item))
        included_declarations.extend(names)
        if isinstance(item, ast.FunctionDef):
            included_functions.append(item.name)
            if item.name == source_symbol:
                retained: list[ast.expr] = []
                for decorator in item.decorator_list:
                    if _is_schedule_decorator(decorator):
                        removed.append(_decorator_name(decorator))
                    else:
                        retained.append(decorator)
                item.decorator_list = retained
        projected_body.append(item)
    projected = ast.Module(body=projected_body, type_ignores=[])
    ast.fix_missing_locations(projected)
    code = ast.unparse(projected) + "\n"
    parameters = tuple(
        argument.arg
        for argument in (
            *target.args.posonlyargs,
            *target.args.args,
            *target.args.kwonlyargs,
        )
    )
    constexprs = tuple(
        argument.arg
        for argument in (
            *target.args.posonlyargs,
            *target.args.args,
            *target.args.kwonlyargs,
        )
        if _annotation_is_constexpr(argument.annotation)
    )
    return TritonSourceSlice(
        code,
        source_symbol,
        parameters,
        constexprs,
        tuple(dict.fromkeys(included_declarations)),
        tuple(included_functions),
        tuple(removed),
    )


def _module_source_path(
    module: str,
    search_roots: Iterable[str | Path],
) -> Path | None:
    relative = Path(*module.split(".")).with_suffix(".py")
    for root in search_roots:
        candidate = Path(root).expanduser().resolve() / relative
        if candidate.is_file():
            return candidate
    return None


def inline_imported_triton_helpers(
    source_slice: TritonSourceSlice,
    *,
    search_roots: Iterable[str | Path],
) -> TritonSourceSlice:
    """Inline imported ``@triton.jit`` helpers without importing a package.

    Real source trees can have a Python package ``__init__`` that requires an
    optional native extension even though the selected Triton kernel only needs
    a small JIT helper module.  This routine resolves those helper definitions
    from source and forms their ordinary AST closure.  Non-Triton imports and
    unresolved helpers remain fail-closed.
    """

    roots = tuple(Path(item).expanduser().resolve() for item in search_roots)
    tree = ast.parse(source_slice.code)
    function_loads: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef):
            function_loads.update(_function_dependencies(statement))

    replacements: dict[int, list[ast.stmt]] = {}
    inlined_labels: list[str] = []
    for index, statement in enumerate(tree.body):
        if not isinstance(statement, ast.ImportFrom) or statement.level != 0:
            continue
        module = statement.module or ""
        if module == "triton" or module.startswith("triton."):
            continue
        requested = [
            alias
            for alias in statement.names
            if (alias.asname or alias.name) in function_loads
        ]
        if not requested:
            continue
        module_path = _module_source_path(module, roots)
        if module_path is None:
            raise TritonSourceSliceError(
                f"cannot resolve imported Triton helper module {module!r}"
            )
        module_source = module_path.read_text(encoding="utf-8")
        replacement: list[ast.stmt] = []
        for alias in requested:
            if alias.asname is not None and alias.asname != alias.name:
                raise TritonSourceSliceError(
                    "aliased imported Triton helpers are not supported"
                )
            helper = build_triton_source_slice(
                module_source, source_symbol=alias.name
            )
            replacement.extend(copy.deepcopy(ast.parse(helper.code).body))
            inlined_labels.append(f"{module}:{alias.name}")
        replacements[index] = replacement

    if not replacements:
        return source_slice

    expanded: list[ast.stmt] = []
    for index, statement in enumerate(tree.body):
        expanded.extend(replacements.get(index, [statement]))
    deduplicated: list[ast.stmt] = []
    seen_structures: set[str] = set()
    named_functions: dict[str, str] = {}
    for statement in expanded:
        structure = ast.dump(statement, include_attributes=False)
        if isinstance(statement, ast.FunctionDef):
            previous = named_functions.get(statement.name)
            if previous is not None and previous != structure:
                raise TritonSourceSliceError(
                    f"inlined helper {statement.name!r} has conflicting definitions"
                )
            named_functions[statement.name] = structure
        if structure in seen_structures:
            continue
        seen_structures.add(structure)
        deduplicated.append(statement)
    projected = ast.Module(body=deduplicated, type_ignores=[])
    ast.fix_missing_locations(projected)
    code = ast.unparse(projected) + "\n"
    declarations: list[str] = []
    functions: list[str] = []
    for statement in deduplicated:
        declarations.extend(sorted(_bound_names(statement)))
        if isinstance(statement, ast.FunctionDef):
            functions.append(statement.name)
    return TritonSourceSlice(
        code,
        source_slice.function,
        source_slice.parameters,
        source_slice.constexpr_parameters,
        tuple(dict.fromkeys(declarations)),
        tuple(functions),
        source_slice.removed_decorators,
        tuple(dict.fromkeys((*source_slice.inlined_imports, *inlined_labels))),
    )


__all__ = [
    "TRITON_SOURCE_SLICE_PROTOCOL",
    "TritonSourceSlice",
    "TritonSourceSliceError",
    "build_triton_source_slice",
    "inline_imported_triton_helpers",
]
