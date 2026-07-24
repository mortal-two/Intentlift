"""Deterministically decode and elaborate an LLM response into the one J.

The Lift model is allowed to choose semantic structure.  It is not required to
guess punctuation preferred by a particular evaluator or backend.  This module
is the single closed-world boundary that parses the bounded ContractExpr
language, resolves index vectors, materializes uniquely inferable value types,
and removes mechanically redundant output bindings.

The accepted :class:`SemanticArchive` is J.  ``JElaboration`` below is only its
private parsed-expression/type cache, analogous to a compiler symbol table; it
does not contain, serialize, or identify another J.
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

from .model import IntentFunction, IntentOp

if TYPE_CHECKING:  # pragma: no cover - imports are only for static checkers
    from intentir.verify.gate1.model import Gate1Artifact, SemanticArchive


class CanonicalJError(ValueError):
    """J wire syntax has no unique, closed ContractExpr elaboration."""


class ContractSort(str, Enum):
    INTEGER = "integer"
    BOOLEAN = "boolean"
    UNKNOWN_REF = "unknown_ref"


_BINARY_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "floordiv": "//",
    "mod": "%",
}
_COMPARE_SYMBOLS = {
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
}


@dataclass(frozen=True)
class ContractExpr:
    """One node in the closed integer/boolean ContractExpr v1 AST."""

    op: str
    args: tuple["ContractExpr", ...] = ()
    value: str | int | bool | None = None
    sort: ContractSort = ContractSort.UNKNOWN_REF

    @property
    def references(self) -> frozenset[str]:
        result: set[str] = set()
        if self.op == "ref":
            assert isinstance(self.value, str)
            result.add(self.value)
        for argument in self.args:
            result.update(argument.references)
        return frozenset(result)

    def to_text(self) -> str:
        if self.op == "ref":
            assert isinstance(self.value, str)
            return self.value
        if self.op == "int":
            assert isinstance(self.value, int) and not isinstance(self.value, bool)
            return str(self.value)
        if self.op == "bool":
            assert isinstance(self.value, bool)
            return "true" if self.value else "false"
        if self.op == "not":
            return f"not ({self.args[0].to_text()})"
        if self.op == "neg":
            return f"-({self.args[0].to_text()})"
        if self.op == "pos":
            return f"+({self.args[0].to_text()})"
        if self.op in _BINARY_SYMBOLS:
            left, right = self.args
            return (
                f"({left.to_text()} {_BINARY_SYMBOLS[self.op]} "
                f"{right.to_text()})"
            )
        if self.op in _COMPARE_SYMBOLS:
            left, right = self.args
            return (
                f"{left.to_text()} {_COMPARE_SYMBOLS[self.op]} "
                f"{right.to_text()}"
            )
        if self.op in {"and", "or"}:
            return f" {self.op} ".join(
                f"({argument.to_text()})" for argument in self.args
            )
        if self.op == "call":
            assert isinstance(self.value, str)
            return (
                f"{self.value}("
                + ", ".join(argument.to_text() for argument in self.args)
                + ")"
            )
        raise AssertionError(f"unknown ContractExpr opcode {self.op!r}")

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op, "sort": self.sort.value}
        if self.value is not None:
            result["value"] = self.value
        if self.args:
            result["args"] = [argument.to_json() for argument in self.args]
        return result


def _integer_like(expression: ContractExpr) -> bool:
    return expression.sort in {ContractSort.INTEGER, ContractSort.UNKNOWN_REF}


def _boolean_like(expression: ContractExpr) -> bool:
    return expression.sort in {ContractSort.BOOLEAN, ContractSort.UNKNOWN_REF}


def _contract_expr(node: ast.AST, *, context: str) -> ContractExpr:
    if isinstance(node, ast.Name):
        if node.id in {"true", "false", "True", "False"}:
            return ContractExpr(
                "bool",
                value=node.id.lower() == "true",
                sort=ContractSort.BOOLEAN,
            )
        return ContractExpr(
            "ref", value=node.id, sort=ContractSort.UNKNOWN_REF
        )
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return ContractExpr(
                "bool", value=node.value, sort=ContractSort.BOOLEAN
            )
        if isinstance(node.value, int):
            return ContractExpr(
                "int", value=node.value, sort=ContractSort.INTEGER
            )
        raise CanonicalJError(
            f"{context} permits only integer and boolean literals"
        )
    if isinstance(node, ast.UnaryOp):
        argument = _contract_expr(node.operand, context=context)
        if isinstance(node.op, ast.Not):
            if not _boolean_like(argument):
                raise CanonicalJError(f"{context} applies not to an integer")
            return ContractExpr("not", (argument,), sort=ContractSort.BOOLEAN)
        if isinstance(node.op, (ast.USub, ast.UAdd)):
            if not _integer_like(argument):
                raise CanonicalJError(f"{context} applies a sign to a boolean")
            return ContractExpr(
                "neg" if isinstance(node.op, ast.USub) else "pos",
                (argument,),
                sort=ContractSort.INTEGER,
            )
        raise CanonicalJError(f"{context} contains an unsupported unary operator")
    if isinstance(node, ast.BinOp):
        opcode = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.FloorDiv: "floordiv",
            ast.Mod: "mod",
        }.get(type(node.op))
        if opcode is None:
            raise CanonicalJError(
                f"{context} contains an unsupported arithmetic operator"
            )
        left = _contract_expr(node.left, context=context)
        right = _contract_expr(node.right, context=context)
        if not _integer_like(left) or not _integer_like(right):
            raise CanonicalJError(f"{context} uses boolean arithmetic")
        return ContractExpr(
            opcode, (left, right), sort=ContractSort.INTEGER
        )
    if isinstance(node, ast.BoolOp):
        opcode = "and" if isinstance(node.op, ast.And) else (
            "or" if isinstance(node.op, ast.Or) else None
        )
        if opcode is None:
            raise CanonicalJError(f"{context} contains an unsupported boolean op")
        arguments = tuple(
            _contract_expr(value, context=context) for value in node.values
        )
        if not all(_boolean_like(argument) for argument in arguments):
            raise CanonicalJError(f"{context} uses an integer as a predicate")
        return ContractExpr(opcode, arguments, sort=ContractSort.BOOLEAN)
    if isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
        terms: list[ContractExpr] = []
        for left_node, operator, right_node in zip(
            operands[:-1], node.ops, operands[1:], strict=True
        ):
            opcode = {
                ast.Lt: "lt",
                ast.LtE: "le",
                ast.Gt: "gt",
                ast.GtE: "ge",
                ast.Eq: "eq",
                ast.NotEq: "ne",
            }.get(type(operator))
            if opcode is None:
                raise CanonicalJError(
                    f"{context} contains an unsupported comparison"
                )
            left = _contract_expr(left_node, context=context)
            right = _contract_expr(right_node, context=context)
            if not _integer_like(left) or not _integer_like(right):
                raise CanonicalJError(
                    f"{context} compares a boolean as an integer"
                )
            terms.append(
                ContractExpr(opcode, (left, right), sort=ContractSort.BOOLEAN)
            )
        if len(terms) == 1:
            return terms[0]
        return ContractExpr("and", tuple(terms), sort=ContractSort.BOOLEAN)
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in {"ceil_div", "cdiv", "min", "max"}
            or node.keywords
            or len(node.args) != 2
        ):
            raise CanonicalJError(
                f"{context} permits only binary ceil_div/cdiv/min/max calls"
            )
        arguments = tuple(
            _contract_expr(argument, context=context) for argument in node.args
        )
        if not all(_integer_like(argument) for argument in arguments):
            raise CanonicalJError(f"{context} calls an integer function on bool")
        return ContractExpr(
            "call",
            arguments,
            value="ceil_div" if node.func.id == "cdiv" else node.func.id,
            sort=ContractSort.INTEGER,
        )
    raise CanonicalJError(
        f"{context} contains forbidden syntax {type(node).__name__}"
    )


def _parse_python_expression(expression: str, *, context: str) -> ast.AST:
    if not isinstance(expression, str) or not expression.strip():
        raise CanonicalJError(f"{context} must be a non-empty expression")
    try:
        return ast.parse(expression.strip(), mode="eval").body
    except SyntaxError as exc:
        raise CanonicalJError(f"{context} has invalid ContractExpr syntax") from exc


def parse_contract_expr(
    expression: str,
    *,
    expected: ContractSort,
    context: str = "ContractExpr",
) -> ContractExpr:
    parsed = _contract_expr(
        _parse_python_expression(expression, context=context), context=context
    )
    if expected is ContractSort.INTEGER and not _integer_like(parsed):
        raise CanonicalJError(f"{context} must have integer/index sort")
    if expected is ContractSort.BOOLEAN and not _boolean_like(parsed):
        raise CanonicalJError(f"{context} must have boolean sort")
    return parsed


@dataclass(frozen=True)
class CanonicalIndex:
    """A structured ordered index vector, never an unparsed wire string."""

    components: tuple[ContractExpr, ...]

    @property
    def direct_axes(self) -> tuple[str, ...] | None:
        axes: list[str] = []
        for component in self.components:
            if component.op == "ref":
                assert isinstance(component.value, str)
                axes.append(component.value)
            elif component.op == "int" and component.value == 0:
                axes.append("0")
            else:
                return None
        return tuple(axes)

    def to_text(self) -> str:
        return "[" + ", ".join(item.to_text() for item in self.components) + "]"

    def to_json(self) -> list[dict[str, Any]]:
        return [component.to_json() for component in self.components]


def parse_index_expression(
    expression: str, *, context: str = "index_expression"
) -> CanonicalIndex:
    node = _parse_python_expression(expression, context=context)
    if isinstance(node, (ast.List, ast.Tuple)):
        elements = node.elts
    else:
        elements = [node]
    components = tuple(
        _contract_expr(element, context=context) for element in elements
    )
    if not all(_integer_like(component) for component in components):
        raise CanonicalJError(f"{context} contains a boolean index component")
    return CanonicalIndex(components)


@dataclass(frozen=True)
class CanonicalValueType:
    dtype: str
    shape: tuple[Any, ...]

    def to_json(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class CanonicalBoundaryAccess:
    access_id: str
    index: CanonicalIndex
    predicate: ContractExpr | None


@dataclass(frozen=True)
class CanonicalOutputBinding:
    output: str
    value: str
    node_ref: str | None


@dataclass(frozen=True)
class JElaboration:
    """Private structured facts cached on the one accepted J object."""

    domains: tuple[tuple[str, ContractExpr], ...]
    accesses: tuple[CanonicalBoundaryAccess, ...]
    value_types: tuple[tuple[str, CanonicalValueType], ...]
    output_bindings: tuple[CanonicalOutputBinding, ...]
    _snapshot: str

    @property
    def access_by_id(self) -> dict[str, CanonicalBoundaryAccess]:
        return {access.access_id: access for access in self.accesses}

    @property
    def domain_by_id(self) -> dict[str, ContractExpr]:
        return dict(self.domains)

    @property
    def value_type_by_name(self) -> dict[str, CanonicalValueType]:
        return dict(self.value_types)

    def structured_boundary_json(self) -> dict[str, Any]:
        """Diagnostic typed view; it is not an additional persisted artifact."""

        return {
            "domains": {
                name: expression.to_json() for name, expression in self.domains
            },
            "accesses": [
                {
                    "access_id": access.access_id,
                    "axes": access.index.to_json(),
                    "predicate": (
                        access.predicate.to_json()
                        if access.predicate is not None else None
                    ),
                }
                for access in self.accesses
            ],
            "output_bindings": [
                {
                    "output": binding.output,
                    "value": binding.value,
                    "node_ref": binding.node_ref,
                }
                for binding in self.output_bindings
            ],
        }


def _result_names(op: IntentOp) -> tuple[str, ...]:
    if isinstance(op.result, str):
        return (op.result,)
    if isinstance(op.result, list):
        return tuple(item for item in op.result if isinstance(item, str))
    return ()


def _walk_ops(body: Sequence[IntentOp]) -> Iterable[IntentOp]:
    for op in body:
        yield op
        yield from _walk_ops(op.body)


def _broadcast_shape(shapes: Sequence[tuple[Any, ...]]) -> tuple[Any, ...] | None:
    if not shapes:
        return None
    rank = max(len(shape) for shape in shapes)
    result: list[Any] = []
    for offset in range(1, rank + 1):
        dimensions = [
            shape[-offset] if offset <= len(shape) else 1 for shape in shapes
        ]
        nonunit = [dimension for dimension in dimensions if dimension != 1]
        if not nonunit:
            result.append(1)
            continue
        first = nonunit[0]
        if any(dimension != first for dimension in nonunit[1:]):
            return None
        result.append(first)
    return tuple(reversed(result))


def _project_explicit_broadcast_shape(
    shape: tuple[Any, ...],
    axes: Sequence[int],
    target_shape: tuple[Any, ...] | None,
) -> tuple[Any, ...]:
    """Project one operand using only broadcast axes explicitly present in J."""

    from intentir.frontend.emit_mlir import _broadcast_projection

    try:
        projected, _inserted = _broadcast_projection(
            list(shape),
            axes,
            list(target_shape) if target_shape is not None else None,
        )
    except (NotImplementedError, ValueError) as exc:
        raise CanonicalJError(str(exc)) from exc
    if projected is None:  # pragma: no cover - concrete shape is supplied above
        raise CanonicalJError("explicit broadcast has no closed shape projection")
    return tuple(projected)


def _infer_shape(
    op: IntentOp,
    known: Mapping[str, CanonicalValueType],
    broadcast_axes: Mapping[str, Mapping[str, tuple[int, ...]]] | None = None,
) -> tuple[Any, ...] | None:
    operand_shapes = [known[name].shape for name in op.operands if name in known]
    if op.op in {"map_unary", "cast", "scan", "flip"} and operand_shapes:
        return operand_shapes[0]
    if op.op in {"map_binary", "select"} and len(operand_shapes) == len(op.operands):
        explicit = (
            broadcast_axes.get(op.result, {})
            if broadcast_axes is not None and isinstance(op.result, str)
            else {}
        )
        target = known.get(op.result) if isinstance(op.result, str) else None
        target_shape = target.shape if target is not None else None
        if target_shape is None and explicit:
            target_shape = next(
                (
                    shape
                    for operand, shape in zip(
                        op.operands, operand_shapes, strict=True
                    )
                    if operand not in explicit
                    and len(shape) == max(len(item) for item in operand_shapes)
                ),
                None,
            )
        projected = tuple(
            _project_explicit_broadcast_shape(
                shape,
                explicit.get(operand, ()),
                target_shape,
            )
            for operand, shape in zip(op.operands, operand_shapes, strict=True)
        )
        inferred = _broadcast_shape(projected)
        if inferred is None and op.op == "map_binary":
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} operands have incompatible "
                "NumPy broadcast shapes; a non-trailing broadcast must be "
                "declared explicitly in J.boundary.broadcasts"
            )
        return inferred
    if op.op == "transpose" and operand_shapes:
        permutation = op.attrs.get("perm")
        if (
            isinstance(permutation, list)
            and all(isinstance(axis, int) and not isinstance(axis, bool) for axis in permutation)
            and sorted(permutation) == list(range(len(operand_shapes[0])))
        ):
            return tuple(operand_shapes[0][axis] for axis in permutation)
    if op.op in {"reduce", "arg_reduce"} and operand_shapes:
        axes_value = op.attrs.get("axes", [op.attrs.get("axis")])
        if isinstance(axes_value, list) and all(
            isinstance(axis, int) and not isinstance(axis, bool)
            for axis in axes_value
        ):
            source = operand_shapes[0]
            axes = {axis if axis >= 0 else len(source) + axis for axis in axes_value}
            if any(axis < 0 or axis >= len(source) for axis in axes):
                return None
            if op.attrs.get("keepdims") is True:
                return tuple(1 if index in axes else dim for index, dim in enumerate(source))
            return tuple(dim for index, dim in enumerate(source) if index not in axes)
    if op.op == "sort" and operand_shapes:
        shape = list(operand_shapes[0])
        width = op.attrs.get("k")
        if width is not None and shape:
            shape[-1] = width
        return tuple(shape)
    if op.op in {"const", "rand"}:
        shape = op.attrs.get("shape", [])
        if isinstance(shape, list):
            return tuple(shape)
    if op.op == "iota" and "n" in op.attrs:
        return (op.attrs["n"],)
    return None


def _infer_dtypes(
    op: IntentOp,
    known: Mapping[str, CanonicalValueType],
) -> dict[str, str]:
    # Keep one source of truth with the MLIR emitter's closed operation table.
    from intentir.frontend.emit_mlir import _infer_op_dtypes

    dtypes = {name: value.dtype for name, value in known.items()}
    return _infer_op_dtypes(op, dtypes)


def _merge_type(
    known: dict[str, CanonicalValueType],
    name: str,
    value: CanonicalValueType,
    *,
    context: str,
) -> bool:
    previous = known.get(name)
    if previous is not None and previous != value:
        raise CanonicalJError(
            f"{context} infers conflicting types for {name!r}: "
            f"{previous.to_json()} versus {value.to_json()}"
        )
    if previous is None:
        known[name] = value
        return True
    return False


def _elaborate_function_types(
    fn: IntentFunction,
    broadcast_axes: Mapping[str, Mapping[str, tuple[int, ...]]] | None = None,
) -> dict[str, CanonicalValueType]:
    known: dict[str, CanonicalValueType] = {
        tensor.name: CanonicalValueType(tensor.dtype, tuple(tensor.shape))
        for tensor in [*fn.inputs, *fn.outputs]
    }
    outputs = {
        tensor.name: CanonicalValueType(tensor.dtype, tuple(tensor.shape))
        for tensor in fn.outputs
    }
    for binding in fn.returns:
        output_type = outputs.get(binding.output)
        if output_type is None:
            raise CanonicalJError(
                f"IntentFunction return names unknown output {binding.output!r}"
            )
        _merge_type(
            known,
            binding.value,
            output_type,
            context="IntentFunction return",
        )

    operations = tuple(_walk_ops(fn.body))
    for _ in range(len(operations) + 3):
        changed = False
        for op in operations:
            names = _result_names(op)
            declared: CanonicalValueType | None = None
            if op.type is not None:
                if (
                    not isinstance(op.type, dict)
                    or not isinstance(op.type.get("dtype"), str)
                    or not isinstance(op.type.get("shape"), list)
                ):
                    raise CanonicalJError(
                        f"IntentOp {op.node_id or op.op!r} has a partial result type"
                    )
                declared = CanonicalValueType(
                    op.type["dtype"], tuple(op.type["shape"])
                )
                if len(names) == 1:
                    changed |= _merge_type(
                        known,
                        names[0],
                        declared,
                        context=f"IntentOp {op.node_id or op.op}",
                    )

            inferred_dtypes = _infer_dtypes(op, known)
            inferred_shape = _infer_shape(op, known, broadcast_axes)
            for name in names:
                previous = known.get(name)
                dtype = inferred_dtypes.get(name) or (
                    previous.dtype if previous is not None else None
                )
                shape = inferred_shape if inferred_shape is not None else (
                    previous.shape if previous is not None else None
                )
                if dtype is None or shape is None:
                    continue
                changed |= _merge_type(
                    known,
                    name,
                    CanonicalValueType(dtype, tuple(shape)),
                    context=f"IntentOp {op.node_id or op.op}",
                )
            if declared is not None and len(names) == 1:
                actual = known.get(names[0])
                if actual is not None and actual != declared:
                    raise CanonicalJError(
                        f"IntentOp {op.node_id or op.op!r} explicit type conflicts "
                        "with its closed operation signature"
                    )
        if not changed:
            break

    for op in operations:
        names = _result_names(op)
        if op.type is None and len(names) == 1 and names[0] in known:
            op.type = known[names[0]].to_json()
    return known


def _node_results(j: Any) -> dict[str, tuple[str, ...]]:
    by_path: dict[str, tuple[str, ...]] = {}

    def visit(body: Sequence[IntentOp], prefix: str = "body") -> None:
        for index, op in enumerate(body):
            path = f"{prefix}[{index}]"
            by_path[path] = _result_names(op)
            visit(op.body, f"{path}.body")

    visit(j.intent.body)
    result = dict(by_path)
    for node_id, path in j.node_ids.items():
        if path in by_path:
            result[node_id] = by_path[path]
    return result


def _declared_broadcast_axes(
    j: Any,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Resolve only the broadcast facts explicitly serialized in canonical J."""

    by_path: dict[str, IntentOp] = {}

    def visit(body: Sequence[IntentOp], prefix: str = "body") -> None:
        for index, operation in enumerate(body):
            path = f"{prefix}[{index}]"
            by_path[path] = operation
            visit(operation.body, f"{path}.body")

    visit(j.intent.body)
    nodes = dict(by_path)
    for node_id, path in j.node_ids.items():
        operation = by_path.get(path)
        if operation is not None:
            nodes[node_id] = operation

    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for projection in j.boundary.broadcasts:
        operation = nodes.get(projection.node_ref)
        if operation is None or not isinstance(operation.result, str):
            raise CanonicalJError(
                "J boundary broadcast references an unknown single-result node"
            )
        if operation.op not in {"map_binary", "select"}:
            raise CanonicalJError(
                "J boundary broadcast must target map_binary or select"
            )
        if projection.operand not in operation.operands:
            raise CanonicalJError(
                "J boundary broadcast operand is not consumed by its node"
            )
        if any(
            isinstance(axis, bool) or not isinstance(axis, int)
            for axis in projection.axes
        ):
            raise CanonicalJError(
                "canonical value typing requires integer broadcast axes"
            )
        axes = tuple(int(axis) for axis in projection.axes)
        previous = result.setdefault(operation.result, {}).get(
            projection.operand
        )
        if previous is not None and previous != axes:
            raise CanonicalJError(
                "J boundary contains conflicting broadcast projections"
            )
        result[operation.result][projection.operand] = axes
    return result


def _remove_redundant_identity_mappings(j: Any, payload: dict[str, Any]) -> None:
    boundary = payload["boundary"]
    mappings = boundary.get("output_mappings", [])
    if not mappings:
        return
    outputs = {tensor.name for tensor in j.intent.outputs}
    node_results = _node_results(j)
    stores = [
        access for access in j.boundary.accesses
        if access.kind.value == "store"
    ]
    retained: list[dict[str, Any]] = []
    for mapping in mappings:
        output = mapping["output"]
        value = mapping["value"]
        node_ref = mapping.get("node_ref")
        matching_stores = [
            access for access in stores
            if access.tensor == output and access.node_ref == node_ref
        ]
        proven_identity = (
            output == value
            and output in outputs
            and (
                node_ref is None
                or (
                    output in node_results.get(node_ref, ())
                    and len(matching_stores) == 1
                )
            )
        )
        if not proven_identity:
            retained.append(mapping)
    boundary["output_mappings"] = retained


def _snapshot(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _build_output_bindings(j: Any) -> tuple[CanonicalOutputBinding, ...]:
    returns = {item.output: item.value for item in j.intent.returns}
    mappings = {item.output: item for item in j.boundary.output_mappings}
    node_results = _node_results(j)
    result_nodes: dict[str, list[str]] = {}
    for node_ref, results in node_results.items():
        if not node_ref.startswith("n"):
            continue
        for result in results:
            result_nodes.setdefault(result, []).append(node_ref)
    bindings: list[CanonicalOutputBinding] = []
    for output in j.intent.outputs:
        mapping = mappings.get(output.name)
        returned = returns.get(output.name)
        if mapping is not None and returned is not None and mapping.value != returned:
            raise CanonicalJError(
                f"output {output.name!r} has conflicting return and boundary mapping"
            )
        value = mapping.value if mapping is not None else (returned or output.name)
        node_ref = mapping.node_ref if mapping is not None else None
        if node_ref is None:
            candidates = sorted(set(result_nodes.get(value, [])))
            if len(candidates) == 1:
                node_ref = candidates[0]
        bindings.append(CanonicalOutputBinding(output.name, value, node_ref))
    return tuple(bindings)


def _elaborate_j(j: "SemanticArchive") -> tuple["SemanticArchive", JElaboration]:
    """Construct the accepted J and its private parsed cache."""

    # Local import prevents the J wire dataclasses from depending on their
    # elaborated view while they are being defined.
    from intentir.verify.gate1.model import SemanticArchive

    if not isinstance(j, SemanticArchive):
        raise TypeError("elaborate_j requires SemanticArchive J")
    source_payload = j.to_json()
    source_snapshot = _snapshot(source_payload)
    cached = getattr(j, "_intentir_j_elaboration", None)
    if isinstance(cached, JElaboration) and cached._snapshot == source_snapshot:
        return j, cached
    source_cached = getattr(j, "_intentir_elaborated_source", None)
    if (
        isinstance(source_cached, tuple)
        and len(source_cached) == 3
        and source_cached[0] == source_snapshot
        and isinstance(source_cached[1], SemanticArchive)
        and isinstance(source_cached[2], JElaboration)
    ):
        return source_cached[1], source_cached[2]

    payload = copy.deepcopy(source_payload)
    function = IntentFunction.from_json(payload["intent"])
    _elaborate_function_types(function, _declared_broadcast_axes(j))
    payload["intent"] = function.to_json()

    boundary = payload["boundary"]
    for domain_id, expression in list(boundary.get("domains", {}).items()):
        parsed = parse_contract_expr(
            expression,
            expected=ContractSort.BOOLEAN,
            context=f"J.boundary.domains.{domain_id}",
        )
        boundary["domains"][domain_id] = parsed.to_text()
    for index, access in enumerate(boundary.get("accesses", [])):
        parsed_index = parse_index_expression(
            access["index_expression"],
            context=f"J.boundary.accesses[{index}].index_expression",
        )
        access["index_expression"] = parsed_index.to_text()
        predicate = access.get("predicate")
        if predicate is not None:
            parsed_predicate = parse_contract_expr(
                predicate,
                expected=ContractSort.BOOLEAN,
                context=f"J.boundary.accesses[{index}].predicate",
            )
            access["predicate"] = parsed_predicate.to_text()

    # Inspect identity against the original node/access objects, then serialize
    # the one canonical omission form.
    _remove_redundant_identity_mappings(j, payload)
    canonical_archive = (
        j if payload == source_payload else SemanticArchive.from_json(payload)
    )
    canonical_payload = canonical_archive.to_json()
    canonical_snapshot = _snapshot(canonical_payload)

    domains = tuple(
        (
            name,
            parse_contract_expr(
                expression,
                expected=ContractSort.BOOLEAN,
                context=f"J.boundary.domains.{name}",
            ),
        )
        for name, expression in canonical_archive.boundary.domains.items()
    )
    accesses = tuple(
        CanonicalBoundaryAccess(
            access.access_id,
            parse_index_expression(
                access.index_expression,
                context=f"J.boundary.accesses.{access.access_id}.index_expression",
            ),
            (
                parse_contract_expr(
                    access.predicate,
                    expected=ContractSort.BOOLEAN,
                    context=f"J.boundary.accesses.{access.access_id}.predicate",
                )
                if access.predicate is not None else None
            ),
        )
        for access in canonical_archive.boundary.accesses
    )
    typed_function = IntentFunction.from_json(canonical_archive.intent.to_json())
    value_types = tuple(
        sorted(
            _elaborate_function_types(
                typed_function,
                _declared_broadcast_axes(canonical_archive),
            ).items()
        )
    )
    record = JElaboration(
        domains,
        accesses,
        value_types,
        _build_output_bindings(canonical_archive),
        canonical_snapshot,
    )
    setattr(canonical_archive, "_intentir_j_elaboration", record)
    if canonical_archive is not j:
        setattr(
            j,
            "_intentir_elaborated_source",
            (source_snapshot, canonical_archive, record),
        )
    return canonical_archive, record


def elaborate_j(j: "SemanticArchive") -> "SemanticArchive":
    """Decode/normalize a response-form archive into the one accepted J."""

    return _elaborate_j(j)[0]


def j_elaboration(j: "SemanticArchive") -> JElaboration:
    """Return structured facts derived from J; never a second J object."""

    accepted, record = _elaborate_j(j)
    if accepted is not j:
        raise ValueError(
            "j_elaboration requires the accepted J returned by elaborate_j"
        )
    return record


def canonicalize_semantic_archive(j: "SemanticArchive") -> "SemanticArchive":
    return elaborate_j(j)


def canonicalize_gate1_artifact(artifact: "Gate1Artifact") -> "Gate1Artifact":
    from intentir.verify.gate1.model import Gate1Artifact

    if not isinstance(artifact, Gate1Artifact):
        raise TypeError("canonicalize_gate1_artifact requires Gate1Artifact")
    archive = elaborate_j(artifact.j)
    if archive is artifact.j:
        return artifact
    return Gate1Artifact(
        archive,
        artifact.c,
        artifact.pi,
        wire_schema=artifact.wire_schema,
    )


__all__ = [
    "CanonicalBoundaryAccess",
    "CanonicalIndex",
    "CanonicalJError",
    "CanonicalOutputBinding",
    "CanonicalValueType",
    "ContractExpr",
    "ContractSort",
    "JElaboration",
    "canonicalize_gate1_artifact",
    "canonicalize_semantic_archive",
    "elaborate_j",
    "j_elaboration",
    "parse_contract_expr",
    "parse_index_expression",
]
