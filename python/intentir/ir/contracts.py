"""Strict, target-neutral Layer-B contracts for canonical IntentIR programs.

The historical in-memory model permits open ``layer_b`` dictionaries and
source-shaped index strings.  Those remain readable while migration is in
progress, but they are not a safe lowering boundary.  This module defines the
closed representation consumed by :mod:`intentir.ir.program`: predicates are a
small expression tree, logical values are distinct from write destinations,
and every correctness-relevant fact expands to an auditable backend obligation.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class ContractError(ValueError):
    """Raised when a portable Layer-B contract is malformed or incomplete."""


JsonScalar = None | bool | int | float | str
Axis = int | str


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{context} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ContractError(f"{context} keys must be strings")
    return value


def _strict(
    value: Any,
    *,
    context: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> Mapping[str, Any]:
    result = _mapping(value, context)
    missing = sorted(required - set(result))
    unknown = sorted(set(result) - required - optional)
    if missing:
        raise ContractError(f"{context} missing fields: {missing}")
    if unknown:
        raise ContractError(f"{context} has unknown fields: {unknown}")
    return result


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", value) is None:
        raise ContractError(f"{context} must be a non-empty portable identifier")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{context} must be a non-empty string")
    return value


def _scalar(value: Any, context: str) -> JsonScalar:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ContractError(f"{context} must be a finite JSON scalar")


def _axis(value: Any, context: str) -> Axis:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ContractError(f"{context} must be an integer or semantic axis name")
    if isinstance(value, int) and value < 0:
        raise ContractError(f"{context} must be non-negative")
    if isinstance(value, str):
        return _identifier(value, context)
    return value


def _array(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{context} must be an array")
    return value


def _enum(enum_type: type[Enum], value: Any, context: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise ContractError(f"{context} must be one of: {choices}") from exc


class ExprOp(str, Enum):
    CONSTANT = "constant"
    SYMBOL = "symbol"
    NEGATE = "negate"
    NOT = "not"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    FLOOR_DIVIDE = "floor_divide"
    MODULO = "modulo"
    POWER = "power"
    AND = "and"
    OR = "or"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"
    CEIL_DIVIDE = "ceil_divide"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"


_EXPR_ARITY: dict[ExprOp, int | None] = {
    ExprOp.CONSTANT: 0,
    ExprOp.SYMBOL: 0,
    ExprOp.NEGATE: 1,
    ExprOp.NOT: 1,
    ExprOp.ADD: 2,
    ExprOp.SUBTRACT: 2,
    ExprOp.MULTIPLY: 2,
    ExprOp.DIVIDE: 2,
    ExprOp.FLOOR_DIVIDE: 2,
    ExprOp.MODULO: 2,
    ExprOp.POWER: 2,
    ExprOp.AND: None,
    ExprOp.OR: None,
    ExprOp.EQUAL: 2,
    ExprOp.NOT_EQUAL: 2,
    ExprOp.LESS: 2,
    ExprOp.LESS_EQUAL: 2,
    ExprOp.GREATER: 2,
    ExprOp.GREATER_EQUAL: 2,
    ExprOp.CEIL_DIVIDE: 2,
    ExprOp.MINIMUM: 2,
    ExprOp.MAXIMUM: 2,
}


@dataclass(frozen=True)
class ContractExpr:
    """Closed symbolic expression used for domains and access predicates."""

    op: ExprOp
    arguments: tuple["ContractExpr", ...] = ()
    value: JsonScalar = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", _enum(ExprOp, self.op, "expression op"))
        args = tuple(
            item if isinstance(item, ContractExpr) else ContractExpr.from_dict(item)
            for item in self.arguments
        )
        object.__setattr__(self, "arguments", args)
        arity = _EXPR_ARITY[self.op]
        if arity is not None and len(args) != arity:
            raise ContractError(f"expression {self.op.value!r} requires {arity} arguments")
        if self.op in {ExprOp.AND, ExprOp.OR} and len(args) < 2:
            raise ContractError(f"expression {self.op.value!r} requires at least two arguments")
        if self.op is ExprOp.CONSTANT:
            object.__setattr__(self, "value", _scalar(self.value, "constant value"))
        elif self.op is ExprOp.SYMBOL:
            object.__setattr__(self, "value", _identifier(self.value, "expression symbol"))
        elif self.value is not None:
            raise ContractError(f"expression {self.op.value!r} must not carry value")

    @classmethod
    def constant(cls, value: JsonScalar) -> "ContractExpr":
        return cls(ExprOp.CONSTANT, value=value)

    @classmethod
    def symbol(cls, name: str) -> "ContractExpr":
        return cls(ExprOp.SYMBOL, value=name)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractExpr":
        data = _strict(
            value,
            context="contract expression",
            required={"op"},
            optional={"arguments", "value"},
        )
        args = _array(data.get("arguments", ()), "contract expression arguments")
        return cls(
            op=data["op"],
            arguments=tuple(cls.from_dict(item) for item in args),
            value=data.get("value"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op.value}
        if self.arguments:
            result["arguments"] = [item.to_dict() for item in self.arguments]
        if self.op in {ExprOp.CONSTANT, ExprOp.SYMBOL}:
            result["value"] = self.value
        return result

    @property
    def symbols(self) -> frozenset[str]:
        if self.op is ExprOp.SYMBOL:
            assert isinstance(self.value, str)
            return frozenset({self.value})
        return frozenset().union(*(item.symbols for item in self.arguments))


_BINARY_AST_OPS: dict[type[ast.operator], ExprOp] = {
    ast.Add: ExprOp.ADD,
    ast.Sub: ExprOp.SUBTRACT,
    ast.Mult: ExprOp.MULTIPLY,
    ast.Div: ExprOp.DIVIDE,
    ast.FloorDiv: ExprOp.FLOOR_DIVIDE,
    ast.Mod: ExprOp.MODULO,
    ast.Pow: ExprOp.POWER,
    ast.BitAnd: ExprOp.AND,
    ast.BitOr: ExprOp.OR,
}
_COMPARE_AST_OPS: dict[type[ast.cmpop], ExprOp] = {
    ast.Eq: ExprOp.EQUAL,
    ast.NotEq: ExprOp.NOT_EQUAL,
    ast.Lt: ExprOp.LESS,
    ast.LtE: ExprOp.LESS_EQUAL,
    ast.Gt: ExprOp.GREATER,
    ast.GtE: ExprOp.GREATER_EQUAL,
}
_CALL_AST_OPS = {
    "ceil_div": ExprOp.CEIL_DIVIDE,
    "cdiv": ExprOp.CEIL_DIVIDE,
    "min": ExprOp.MINIMUM,
    "max": ExprOp.MAXIMUM,
}


def _expr_from_ast(node: ast.AST) -> ContractExpr:
    if isinstance(node, ast.Constant):
        return ContractExpr.constant(_scalar(node.value, "predicate literal"))
    if isinstance(node, ast.Name):
        return ContractExpr.symbol(node.id)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return ContractExpr(ExprOp.NEGATE, (_expr_from_ast(node.operand),))
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return ContractExpr(ExprOp.NOT, (_expr_from_ast(node.operand),))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_AST_OPS:
        return ContractExpr(
            _BINARY_AST_OPS[type(node.op)],
            (_expr_from_ast(node.left), _expr_from_ast(node.right)),
        )
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        return ContractExpr(
            ExprOp.AND if isinstance(node.op, ast.And) else ExprOp.OR,
            tuple(_expr_from_ast(item) for item in node.values),
        )
    if isinstance(node, ast.Compare):
        comparisons: list[ContractExpr] = []
        left = node.left
        for operator, right in zip(node.ops, node.comparators):
            if type(operator) not in _COMPARE_AST_OPS:
                raise ContractError("predicate uses an unsupported comparison")
            comparisons.append(
                ContractExpr(
                    _COMPARE_AST_OPS[type(operator)],
                    (_expr_from_ast(left), _expr_from_ast(right)),
                )
            )
            left = right
        return comparisons[0] if len(comparisons) == 1 else ContractExpr(ExprOp.AND, tuple(comparisons))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _CALL_AST_OPS
        and not node.keywords
        and len(node.args) == 2
    ):
        return ContractExpr(
            _CALL_AST_OPS[node.func.id], tuple(_expr_from_ast(item) for item in node.args)
        )
    raise ContractError(
        f"predicate construct {type(node).__name__} is outside portable expression v1"
    )


def parse_contract_expression(value: str) -> ContractExpr:
    """Parse legacy text only when it is losslessly representable by Expr v1."""

    if not isinstance(value, str) or not value.strip():
        raise ContractError("predicate text must be a non-empty string")
    try:
        parsed = ast.parse(value, mode="eval")
    except SyntaxError as exc:
        raise ContractError(f"invalid predicate expression: {exc.msg}") from exc
    return _expr_from_ast(parsed.body)


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    ATOMIC = "atomic"
    SYNCHRONIZE = "synchronize"


class IndexRole(str, Enum):
    """Ownership of an access index, never the physical index spelling."""

    LOGICAL_NODE = "logical_node"
    REALIZATION_ONLY = "realization_only"


class AliasRelation(str, Enum):
    MUST_ALIAS = "must_alias"
    MAY_ALIAS = "may_alias"
    NO_ALIAS = "no_alias"


class EffectMode(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    ATOMIC = "atomic"
    SYNCHRONIZE = "synchronize"


class WriteCoverage(str, Enum):
    FULL = "full"
    PREDICATED = "predicated"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ConflictKind(str, Enum):
    UNIQUE_WRITER = "unique_writer"
    ATOMIC = "atomic"
    REDUCTION = "reduction"
    ORDERED = "ordered"
    UNDEFINED = "undefined"
    UNKNOWN = "unknown"


class NanPolicy(str, Enum):
    PROPAGATE = "propagate"
    PREFER_NUMBER = "prefer_number"
    IGNORE = "ignore"
    UNSPECIFIED = "unspecified"


class EmptyDomainPolicy(str, Enum):
    IDENTITY = "identity"
    ERROR = "error"
    UNDEFINED = "undefined"
    UNSPECIFIED = "unspecified"


class ReassociationPolicy(str, Enum):
    FORBIDDEN = "forbidden"
    ALLOWED = "allowed"
    EXACT_TREE = "exact_tree"
    UNSPECIFIED = "unspecified"


class ScopeKind(str, Enum):
    FUNCTION = "function"
    PROGRAM_INSTANCE = "program_instance"
    LOOP_REGION = "loop_region"
    BRANCH_REGION = "branch_region"


class StorageMutability(str, Enum):
    IMMUTABLE = "immutable"
    WRITE_ONCE = "write_once"
    MUTABLE = "mutable"


class ObligationCategory(str, Enum):
    VALUE_DRIVING = "value_driving"
    LEGALITY = "legality"
    PERMISSION = "permission"


class ObligationStatus(str, Enum):
    CONSUMED = "consumed"
    CHECKED = "checked"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ContractObligation:
    obligation_id: str
    category: ObligationCategory
    capability: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _identifier(self.obligation_id, "obligation id"))
        object.__setattr__(self, "category", _enum(ObligationCategory, self.category, "obligation category"))
        object.__setattr__(self, "capability", _identifier(self.capability, "obligation capability"))

    def to_dict(self) -> dict[str, str]:
        return {
            "obligation_id": self.obligation_id,
            "category": self.category.value,
            "capability": self.capability,
        }


@dataclass(frozen=True)
class DomainContract:
    contract_id: str
    name: str
    predicate: ContractExpr

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_id", _identifier(self.contract_id, "domain contract id"))
        object.__setattr__(self, "name", _identifier(self.name, "domain name"))
        if not isinstance(self.predicate, ContractExpr):
            object.__setattr__(self, "predicate", ContractExpr.from_dict(self.predicate))

    def to_dict(self) -> dict[str, Any]:
        return {"contract_id": self.contract_id, "name": self.name, "predicate": self.predicate.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DomainContract":
        data = _strict(value, context="domain contract", required={"contract_id", "name", "predicate"})
        return cls(data["contract_id"], data["name"], ContractExpr.from_dict(data["predicate"]))


@dataclass(frozen=True)
class AccessContract:
    contract_id: str
    mode: AccessMode
    tensor: str
    node_ref: str | None = None
    predicate: ContractExpr | None = None
    fill_value: JsonScalar = None
    index_role: IndexRole = IndexRole.REALIZATION_ONLY
    logical_index_node: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_id", _identifier(self.contract_id, "access contract id"))
        object.__setattr__(self, "mode", _enum(AccessMode, self.mode, "access mode"))
        object.__setattr__(self, "tensor", _identifier(self.tensor, "access tensor"))
        if self.node_ref is not None:
            object.__setattr__(self, "node_ref", _identifier(self.node_ref, "access node_ref"))
        if self.predicate is not None and not isinstance(self.predicate, ContractExpr):
            object.__setattr__(self, "predicate", ContractExpr.from_dict(self.predicate))
        object.__setattr__(self, "fill_value", _scalar(self.fill_value, "access fill_value"))
        object.__setattr__(self, "index_role", _enum(IndexRole, self.index_role, "access index_role"))
        if self.logical_index_node is not None:
            object.__setattr__(self, "logical_index_node", _identifier(self.logical_index_node, "logical index node"))
        if self.index_role is IndexRole.LOGICAL_NODE and self.logical_index_node is None:
            raise ContractError("logical_node access requires logical_index_node")
        if self.index_role is IndexRole.REALIZATION_ONLY and self.logical_index_node is not None:
            raise ContractError("realization_only access cannot carry logical_index_node")
        if self.mode is AccessMode.READ and self.predicate is not None and self.fill_value is None:
            raise ContractError("predicated read requires an explicit fill_value")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "contract_id": self.contract_id,
            "mode": self.mode.value,
            "tensor": self.tensor,
            "index_role": self.index_role.value,
        }
        if self.node_ref is not None:
            result["node_ref"] = self.node_ref
        if self.predicate is not None:
            result["predicate"] = self.predicate.to_dict()
        if self.fill_value is not None:
            result["fill_value"] = self.fill_value
        if self.logical_index_node is not None:
            result["logical_index_node"] = self.logical_index_node
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AccessContract":
        data = _strict(
            value,
            context="access contract",
            required={"contract_id", "mode", "tensor", "index_role"},
            optional={"node_ref", "predicate", "fill_value", "logical_index_node"},
        )
        return cls(
            contract_id=data["contract_id"], mode=data["mode"], tensor=data["tensor"],
            node_ref=data.get("node_ref"),
            predicate=(ContractExpr.from_dict(data["predicate"]) if "predicate" in data else None),
            fill_value=data.get("fill_value"), index_role=data["index_role"],
            logical_index_node=data.get("logical_index_node"),
        )


@dataclass(frozen=True)
class BroadcastProjection:
    contract_id: str
    node_ref: str
    operand: str
    source_rank: int
    consumer_rank: int
    inserted_semantic_axes: tuple[Axis, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_id", _identifier(self.contract_id, "broadcast contract id"))
        object.__setattr__(self, "node_ref", _identifier(self.node_ref, "broadcast node_ref"))
        object.__setattr__(self, "operand", _identifier(self.operand, "broadcast operand"))
        for name in ("source_rank", "consumer_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"broadcast {name} must be non-negative")
        axes = tuple(_axis(item, "broadcast semantic axis") for item in self.inserted_semantic_axes)
        object.__setattr__(self, "inserted_semantic_axes", axes)
        if self.consumer_rank - self.source_rank != len(axes):
            raise ContractError("broadcast rank delta must equal inserted semantic axis count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id, "node_ref": self.node_ref,
            "operand": self.operand, "source_rank": self.source_rank,
            "consumer_rank": self.consumer_rank,
            "inserted_semantic_axes": list(self.inserted_semantic_axes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BroadcastProjection":
        data = _strict(
            value, context="broadcast projection",
            required={"contract_id", "node_ref", "operand", "source_rank", "consumer_rank", "inserted_semantic_axes"},
        )
        return cls(
            data["contract_id"], data["node_ref"], data["operand"],
            data["source_rank"], data["consumer_rank"],
            tuple(_array(data["inserted_semantic_axes"], "broadcast inserted axes")),
        )


@dataclass(frozen=True)
class AliasContract:
    contract_id: str
    left: str
    right: str
    relation: AliasRelation

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_id", _identifier(self.contract_id, "alias contract id"))
        object.__setattr__(self, "left", _identifier(self.left, "alias left"))
        object.__setattr__(self, "right", _identifier(self.right, "alias right"))
        object.__setattr__(self, "relation", _enum(AliasRelation, self.relation, "alias relation"))
        if self.left == self.right:
            raise ContractError("alias endpoints must be distinct")

    def to_dict(self) -> dict[str, str]:
        return {"contract_id": self.contract_id, "left": self.left, "right": self.right, "relation": self.relation.value}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AliasContract":
        data = _strict(value, context="alias contract", required={"contract_id", "left", "right", "relation"})
        return cls(data["contract_id"], data["left"], data["right"], data["relation"])


@dataclass(frozen=True)
class EffectContract:
    contract_id: str
    mode: EffectMode
    target: str
    node_ref: str | None = None
    access_ref: str | None = None
    predicate: ContractExpr | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_id", _identifier(self.contract_id, "effect contract id"))
        object.__setattr__(self, "mode", _enum(EffectMode, self.mode, "effect mode"))
        object.__setattr__(self, "target", _identifier(self.target, "effect target"))
        for name in ("node_ref", "access_ref"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _identifier(getattr(self, name), f"effect {name}"))
        if self.predicate is not None and not isinstance(self.predicate, ContractExpr):
            object.__setattr__(self, "predicate", ContractExpr.from_dict(self.predicate))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"contract_id": self.contract_id, "mode": self.mode.value, "target": self.target}
        for name in ("node_ref", "access_ref"):
            if getattr(self, name) is not None:
                result[name] = getattr(self, name)
        if self.predicate is not None:
            result["predicate"] = self.predicate.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EffectContract":
        data = _strict(
            value, context="effect contract", required={"contract_id", "mode", "target"},
            optional={"node_ref", "access_ref", "predicate"},
        )
        return cls(
            data["contract_id"], data["mode"], data["target"], data.get("node_ref"),
            data.get("access_ref"),
            ContractExpr.from_dict(data["predicate"]) if "predicate" in data else None,
        )


@dataclass(frozen=True)
class DestinationBinding:
    contract_id: str
    output: str
    destination: str
    value: str
    coverage: WriteCoverage
    conflict: ConflictKind
    access_ref: str | None = None
    predicate: ContractExpr | None = None

    def __post_init__(self) -> None:
        for name in ("contract_id", "output", "destination", "value"):
            object.__setattr__(self, name, _identifier(getattr(self, name), f"destination {name}"))
        object.__setattr__(self, "coverage", _enum(WriteCoverage, self.coverage, "destination coverage"))
        object.__setattr__(self, "conflict", _enum(ConflictKind, self.conflict, "destination conflict"))
        if self.access_ref is not None:
            object.__setattr__(self, "access_ref", _identifier(self.access_ref, "destination access_ref"))
        if self.predicate is not None and not isinstance(self.predicate, ContractExpr):
            object.__setattr__(self, "predicate", ContractExpr.from_dict(self.predicate))
        if self.coverage in {WriteCoverage.PREDICATED, WriteCoverage.PARTIAL} and self.predicate is None:
            raise ContractError("conditional/partial destination requires a predicate")
        if self.coverage is WriteCoverage.FULL and self.predicate is not None:
            raise ContractError("full destination must not carry a write predicate")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "contract_id": self.contract_id, "output": self.output,
            "destination": self.destination, "value": self.value,
            "coverage": self.coverage.value, "conflict": self.conflict.value,
        }
        if self.access_ref is not None:
            result["access_ref"] = self.access_ref
        if self.predicate is not None:
            result["predicate"] = self.predicate.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DestinationBinding":
        data = _strict(
            value, context="destination binding",
            required={"contract_id", "output", "destination", "value", "coverage", "conflict"},
            optional={"access_ref", "predicate"},
        )
        return cls(
            data["contract_id"], data["output"], data["destination"], data["value"],
            data["coverage"], data["conflict"], data.get("access_ref"),
            ContractExpr.from_dict(data["predicate"]) if "predicate" in data else None,
        )


@dataclass(frozen=True)
class ReductionContract:
    contract_id: str
    node_ref: str
    semantic_axes: tuple[Axis, ...]
    accumulator_dtype: str
    result_dtype: str
    identity: JsonScalar
    masked_contribution: JsonScalar
    nan_policy: NanPolicy
    empty_domain_policy: EmptyDomainPolicy
    reassociation: ReassociationPolicy

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_id", _identifier(self.contract_id, "reduction contract id"))
        object.__setattr__(self, "node_ref", _identifier(self.node_ref, "reduction node_ref"))
        axes = tuple(_axis(item, "reduction semantic axis") for item in self.semantic_axes)
        if not axes:
            raise ContractError("reduction semantic_axes must not be empty")
        object.__setattr__(self, "semantic_axes", axes)
        object.__setattr__(self, "accumulator_dtype", _text(self.accumulator_dtype, "reduction accumulator_dtype"))
        object.__setattr__(self, "result_dtype", _text(self.result_dtype, "reduction result_dtype"))
        object.__setattr__(self, "identity", _scalar(self.identity, "reduction identity"))
        object.__setattr__(self, "masked_contribution", _scalar(self.masked_contribution, "reduction masked contribution"))
        object.__setattr__(self, "nan_policy", _enum(NanPolicy, self.nan_policy, "reduction nan_policy"))
        object.__setattr__(self, "empty_domain_policy", _enum(EmptyDomainPolicy, self.empty_domain_policy, "reduction empty policy"))
        object.__setattr__(self, "reassociation", _enum(ReassociationPolicy, self.reassociation, "reduction reassociation"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id, "node_ref": self.node_ref,
            "semantic_axes": list(self.semantic_axes),
            "accumulator_dtype": self.accumulator_dtype,
            "result_dtype": self.result_dtype, "identity": self.identity,
            "masked_contribution": self.masked_contribution,
            "nan_policy": self.nan_policy.value,
            "empty_domain_policy": self.empty_domain_policy.value,
            "reassociation": self.reassociation.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReductionContract":
        data = _strict(
            value, context="reduction contract",
            required={
                "contract_id", "node_ref", "semantic_axes", "accumulator_dtype",
                "result_dtype", "identity", "masked_contribution", "nan_policy",
                "empty_domain_policy", "reassociation",
            },
        )
        return cls(
            data["contract_id"], data["node_ref"],
            tuple(_array(data["semantic_axes"], "reduction semantic_axes")),
            data["accumulator_dtype"], data["result_dtype"], data["identity"],
            data["masked_contribution"], data["nan_policy"],
            data["empty_domain_policy"], data["reassociation"],
        )


@dataclass(frozen=True)
class ScopeContract:
    contract_id: str
    kind: ScopeKind
    parent: str | None
    captures: tuple[str, ...] = ()
    definitions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_id", _identifier(self.contract_id, "scope contract id"))
        object.__setattr__(self, "kind", _enum(ScopeKind, self.kind, "scope kind"))
        if self.parent is not None:
            object.__setattr__(self, "parent", _identifier(self.parent, "scope parent"))
        for name in ("captures", "definitions", "effects"):
            values = tuple(_identifier(item, f"scope {name}") for item in getattr(self, name))
            if len(values) != len(set(values)):
                raise ContractError(f"scope {name} must be unique")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id, "kind": self.kind.value,
            "parent": self.parent, "captures": list(self.captures),
            "definitions": list(self.definitions), "effects": list(self.effects),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScopeContract":
        data = _strict(
            value, context="scope contract",
            required={"contract_id", "kind", "parent", "captures", "definitions", "effects"},
        )
        return cls(
            data["contract_id"], data["kind"], data["parent"],
            tuple(_array(data["captures"], "scope captures")),
            tuple(_array(data["definitions"], "scope definitions")),
            tuple(_array(data["effects"], "scope effects")),
        )


@dataclass(frozen=True)
class StorageContract:
    contract_id: str
    members: tuple[str, ...]
    scope_ref: str
    mutability: StorageMutability
    init_event: str
    last_use_event: str
    escapes: bool
    crosses_iteration: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_id", _identifier(self.contract_id, "storage contract id"))
        members = tuple(_identifier(item, "storage member") for item in self.members)
        if not members or len(members) != len(set(members)):
            raise ContractError("storage members must be non-empty and unique")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "scope_ref", _identifier(self.scope_ref, "storage scope_ref"))
        object.__setattr__(self, "mutability", _enum(StorageMutability, self.mutability, "storage mutability"))
        object.__setattr__(self, "init_event", _identifier(self.init_event, "storage init_event"))
        object.__setattr__(self, "last_use_event", _identifier(self.last_use_event, "storage last_use_event"))
        if not isinstance(self.escapes, bool) or not isinstance(self.crosses_iteration, bool):
            raise ContractError("storage escape/cross-iteration flags must be booleans")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id, "members": list(self.members),
            "scope_ref": self.scope_ref, "mutability": self.mutability.value,
            "init_event": self.init_event, "last_use_event": self.last_use_event,
            "escapes": self.escapes, "crosses_iteration": self.crosses_iteration,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StorageContract":
        data = _strict(
            value, context="storage contract",
            required={
                "contract_id", "members", "scope_ref", "mutability", "init_event",
                "last_use_event", "escapes", "crosses_iteration",
            },
        )
        return cls(
            data["contract_id"], tuple(_array(data["members"], "storage members")),
            data["scope_ref"], data["mutability"], data["init_event"],
            data["last_use_event"], data["escapes"], data["crosses_iteration"],
        )


@dataclass(frozen=True)
class BoundaryContract:
    domains: tuple[DomainContract, ...] = ()
    accesses: tuple[AccessContract, ...] = ()
    broadcasts: tuple[BroadcastProjection, ...] = ()
    destinations: tuple[DestinationBinding, ...] = ()
    aliases: tuple[AliasContract, ...] = ()
    effects: tuple[EffectContract, ...] = ()
    reductions: tuple[ReductionContract, ...] = ()
    scopes: tuple[ScopeContract, ...] = ()
    storage: tuple[StorageContract, ...] = ()

    def __post_init__(self) -> None:
        specs = (
            ("domains", DomainContract), ("accesses", AccessContract),
            ("broadcasts", BroadcastProjection), ("destinations", DestinationBinding),
            ("aliases", AliasContract), ("effects", EffectContract),
            ("reductions", ReductionContract), ("scopes", ScopeContract),
            ("storage", StorageContract),
        )
        all_ids: list[str] = []
        for name, item_type in specs:
            values = tuple(
                item if isinstance(item, item_type) else item_type.from_dict(item)
                for item in getattr(self, name)
            )
            object.__setattr__(self, name, values)
            all_ids.extend(item.contract_id for item in values)
        if len(all_ids) != len(set(all_ids)):
            raise ContractError("Layer-B contract_id values must be globally unique")
        access_ids = {item.contract_id for item in self.accesses}
        for item in self.effects:
            if item.access_ref is not None and item.access_ref not in access_ids:
                raise ContractError(f"effect {item.contract_id!r} references unknown access")
        for item in self.destinations:
            if item.access_ref is not None and item.access_ref not in access_ids:
                raise ContractError(f"destination {item.contract_id!r} references unknown access")
        scope_ids = {item.contract_id for item in self.scopes}
        for item in self.scopes:
            if item.parent is not None and item.parent not in scope_ids:
                raise ContractError(f"scope {item.contract_id!r} references unknown parent")
        for item in self.storage:
            if item.scope_ref not in scope_ids:
                raise ContractError(f"storage {item.contract_id!r} references unknown scope")

    def to_dict(self) -> dict[str, Any]:
        return {
            "domains": [item.to_dict() for item in self.domains],
            "accesses": [item.to_dict() for item in self.accesses],
            "broadcasts": [item.to_dict() for item in self.broadcasts],
            "destinations": [item.to_dict() for item in self.destinations],
            "aliases": [item.to_dict() for item in self.aliases],
            "effects": [item.to_dict() for item in self.effects],
            "reductions": [item.to_dict() for item in self.reductions],
            "scopes": [item.to_dict() for item in self.scopes],
            "storage": [item.to_dict() for item in self.storage],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BoundaryContract":
        names = {
            "domains": DomainContract, "accesses": AccessContract,
            "broadcasts": BroadcastProjection, "destinations": DestinationBinding,
            "aliases": AliasContract, "effects": EffectContract,
            "reductions": ReductionContract, "scopes": ScopeContract,
            "storage": StorageContract,
        }
        data = _strict(value, context="boundary contract", required=set(names))
        kwargs: dict[str, Any] = {}
        for name, item_type in names.items():
            kwargs[name] = tuple(
                item_type.from_dict(item)
                for item in _array(data[name], f"boundary {name}")
            )
        return cls(**kwargs)

    def obligations(self) -> tuple[ContractObligation, ...]:
        """Expand every correctness fact into a stable backend checklist."""

        result: list[ContractObligation] = []

        def add(owner: str, suffix: str, category: ObligationCategory, capability: str) -> None:
            result.append(ContractObligation(f"{owner}:{suffix}", category, capability))

        for item in self.domains:
            add(item.contract_id, "predicate", ObligationCategory.LEGALITY, "boundary.domain.predicate")
        for item in self.accesses:
            category = ObligationCategory.VALUE_DRIVING if item.mode is AccessMode.READ else ObligationCategory.PERMISSION
            add(item.contract_id, "access", category, f"boundary.access.{item.mode.value}")
            add(item.contract_id, "index_role", ObligationCategory.VALUE_DRIVING, f"boundary.index.{item.index_role.value}")
            if item.predicate is not None:
                add(item.contract_id, "predicate", ObligationCategory.LEGALITY, "boundary.access.predicate")
            if item.fill_value is not None:
                add(item.contract_id, "fill", ObligationCategory.VALUE_DRIVING, "boundary.access.masked_fill")
        for item in self.broadcasts:
            add(item.contract_id, "projection", ObligationCategory.VALUE_DRIVING, "boundary.broadcast.semantic_projection")
        for item in self.destinations:
            add(item.contract_id, "binding", ObligationCategory.PERMISSION, "boundary.destination.binding")
            add(item.contract_id, "coverage", ObligationCategory.LEGALITY, f"boundary.destination.coverage.{item.coverage.value}")
            add(item.contract_id, "conflict", ObligationCategory.LEGALITY, f"boundary.destination.conflict.{item.conflict.value}")
        for item in self.aliases:
            add(item.contract_id, "relation", ObligationCategory.PERMISSION, f"boundary.alias.{item.relation.value}")
        for item in self.effects:
            add(item.contract_id, "effect", ObligationCategory.PERMISSION, f"boundary.effect.{item.mode.value}")
            if item.predicate is not None:
                add(item.contract_id, "predicate", ObligationCategory.LEGALITY, "boundary.effect.predicate")
        for item in self.reductions:
            add(item.contract_id, "axes", ObligationCategory.VALUE_DRIVING, "boundary.reduction.semantic_axes")
            add(item.contract_id, "dtype", ObligationCategory.VALUE_DRIVING, "boundary.reduction.dtype")
            add(item.contract_id, "masked", ObligationCategory.VALUE_DRIVING, "boundary.reduction.masked_contribution")
            add(item.contract_id, "nan", ObligationCategory.VALUE_DRIVING, f"boundary.reduction.nan.{item.nan_policy.value}")
            add(item.contract_id, "empty", ObligationCategory.LEGALITY, f"boundary.reduction.empty.{item.empty_domain_policy.value}")
            add(item.contract_id, "reassociation", ObligationCategory.PERMISSION, f"boundary.reduction.reassociation.{item.reassociation.value}")
        for item in self.scopes:
            add(item.contract_id, "scope", ObligationCategory.LEGALITY, "boundary.scope")
        for item in self.storage:
            add(item.contract_id, "lifetime", ObligationCategory.LEGALITY, "boundary.storage.lifetime")
            add(item.contract_id, "mutability", ObligationCategory.PERMISSION, f"boundary.storage.{item.mutability.value}")
        return tuple(result)


@dataclass(frozen=True)
class ObligationResult:
    obligation_id: str
    status: ObligationStatus
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _identifier(self.obligation_id, "obligation result id"))
        object.__setattr__(self, "status", _enum(ObligationStatus, self.status, "obligation result status"))
        if not isinstance(self.detail, str):
            raise ContractError("obligation result detail must be a string")

    def to_dict(self) -> dict[str, str]:
        return {"obligation_id": self.obligation_id, "status": self.status.value, "detail": self.detail}


@dataclass(frozen=True)
class ObligationLedger:
    """Backend acknowledgement of every required Layer-B obligation."""

    contract_digest: str
    required: tuple[ContractObligation, ...]
    results: tuple[ObligationResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contract_digest, str) or re.fullmatch(r"[0-9a-f]{64}", self.contract_digest) is None:
            raise ContractError("ledger contract_digest must be a SHA-256 hex digest")
        required = tuple(self.required)
        results = tuple(self.results)
        required_ids = [item.obligation_id for item in required]
        result_ids = [item.obligation_id for item in results]
        if len(required_ids) != len(set(required_ids)):
            raise ContractError("required obligation ids must be unique")
        if len(result_ids) != len(set(result_ids)):
            raise ContractError("obligation result ids must be unique")
        missing = sorted(set(required_ids) - set(result_ids))
        extra = sorted(set(result_ids) - set(required_ids))
        if missing or extra:
            raise ContractError(f"obligation ledger mismatch: missing={missing}, extra={extra}")
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "results", results)

    @property
    def unsupported(self) -> tuple[str, ...]:
        return tuple(
            item.obligation_id for item in self.results
            if item.status is ObligationStatus.UNSUPPORTED
        )

    def assert_supported(self) -> None:
        if self.unsupported:
            raise ContractError(
                "backend does not support required Layer-B obligations: "
                + ", ".join(self.unsupported)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_digest": self.contract_digest,
            "required": [item.to_dict() for item in self.required],
            "results": [item.to_dict() for item in self.results],
        }


__all__ = [
    "AccessContract", "AccessMode", "AliasContract", "AliasRelation", "Axis",
    "BoundaryContract", "BroadcastProjection", "ConflictKind", "ContractError",
    "ContractExpr", "ContractObligation", "DestinationBinding", "DomainContract",
    "EffectContract", "EffectMode", "EmptyDomainPolicy", "ExprOp", "IndexRole",
    "JsonScalar", "NanPolicy", "ObligationCategory", "ObligationLedger",
    "ObligationResult", "ObligationStatus", "ReassociationPolicy", "ReductionContract",
    "ScopeContract", "ScopeKind", "StorageContract", "StorageMutability", "WriteCoverage",
    "parse_contract_expression",
]
