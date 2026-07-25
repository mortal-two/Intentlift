"""Canonical in-memory model for the Layer-A value graph in ``J``.

The persisted contract is intentionally smaller than the downstream
``intent`` MLIR dialect.  Layer B lives only in the enclosing canonical-J
boundary, while scheduling and source realization live in C and pi.  This
module therefore rejects legacy per-op correctness and scheduling fields
instead of silently preserving them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _canonical_dimension(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _closed_keys(value: dict[str, Any], allowed: set[str], path: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path} has forbidden fields: {unknown}")


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: list[Any]  # entries may be ints or symbolic dim names (e.g. "Sq")

    @classmethod
    def from_json(cls, d: dict) -> "Tensor":
        return cls(
            name=d["name"],
            dtype=d["dtype"],
            shape=[_canonical_dimension(item) for item in d["shape"]],
        )

    def to_json(self) -> dict:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class IntentReturn:
    """One Layer-A logical result binding.

    ``value`` is an SSA value produced by the semantic graph.  ``output`` is
    the logical function result it defines.  Caller-owned destinations and
    alias/writeback behavior deliberately do not live here; those are Layer-B
    contracts in the canonical J boundary object.
    """

    output: str
    value: str

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "IntentReturn":
        return cls(output=d["output"], value=d["value"])

    def to_json(self) -> dict[str, str]:
        return {"output": self.output, "value": self.value}


@dataclass
class LoopValueType:
    """Portable type carried across a semantic loop iteration.

    This is deliberately a tensor/value type, not a frontend storage decision:
    register/shared/global placement belongs outside J in realization/lowering.
    """

    dtype: str
    shape: list[Any] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "LoopValueType":
        return cls(
            dtype=d["dtype"],
            shape=[_canonical_dimension(item) for item in d.get("shape", [])],
        )

    def to_json(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape)}


@dataclass
class LoopStateBinding:
    """One explicit ``init -> body arg -> next -> result`` loop binding.

    The legacy IntentIR spelling only carried ``name`` and ``init`` and relied
    on positional ``result``/``yield`` arrays.  ``from_json`` still accepts that
    shape for non-SCRC historical artifacts, while SCRC's strict schema requires
    every field below and validates the redundant positional correspondence.
    """

    state_id: str
    init_value: Any
    body_argument: str
    next_value: str
    result_value: str
    type: LoopValueType | None = None

    @classmethod
    def from_json(
        cls,
        d: dict[str, Any],
        *,
        index: int,
        positional_result: str | None = None,
        positional_yield: str | None = None,
    ) -> "LoopStateBinding":
        legacy_name = d.get("name")
        body_argument = d.get("body_argument", legacy_name)
        result_value = d.get("result_value", positional_result or legacy_name)
        next_value = d.get("next_value", positional_yield or body_argument)
        init_value = d.get("init_value", d.get("init"))
        value_type = d.get("type")
        return cls(
            state_id=d.get("state_id", f"state{index}"),
            init_value=init_value,
            body_argument=body_argument,
            next_value=next_value,
            result_value=result_value,
            type=(LoopValueType.from_json(value_type) if value_type is not None else None),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "state_id": self.state_id,
            "init_value": self.init_value,
            "body_argument": self.body_argument,
            "next_value": self.next_value,
            "result_value": self.result_value,
        }
        if self.type is not None:
            result["type"] = self.type.to_json()
        return result

    def __getitem__(self, key: str) -> Any:
        """Read-only compatibility for archived callers using dict syntax."""

        aliases = {
            "name": "body_argument",
            "init": "init_value",
        }
        return getattr(self, aliases.get(key, key))

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style compatibility without weakening the typed model."""

        aliases = {
            "name": "body_argument",
            "init": "init_value",
        }
        return getattr(self, aliases.get(key, key), default)


@dataclass
class LoopTrip:
    """Semantic loop trip description; tiling is supplied separately by C."""

    axis: str

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "LoopTrip":
        _closed_keys(d, {"axis"}, "LoopTrip")
        return cls(axis=d["axis"])

    def to_json(self) -> dict[str, Any]:
        return {"axis": self.axis}

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class LoopScanInput:
    """One input sliced along a semantic loop trip axis."""

    name: str
    source: str
    axis: int

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "LoopScanInput":
        return cls(name=d["name"], source=d["from"], axis=int(d["axis"]))

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "from": self.source, "axis": self.axis}

    def __getitem__(self, key: str) -> Any:
        return getattr(self, "source" if key == "from" else key)


@dataclass
class EffectIfArm:
    """One explicitly polarized arm of an effect-only conditional region.

    The arm binds an existing portable ``branch_region`` scope.  Its body may
    define branch-local SSA values, but the enclosing ``effect_if`` has no SSA
    result or yield.  Effect/access membership remains solely in the bound
    scope's ``effects`` list and the matching Layer-B records; it is deliberately
    not duplicated here.
    """

    scope_id: str
    body: list["IntentOp"] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "EffectIfArm":
        _closed_keys(d, {"scope_id", "body"}, "EffectIfArm")
        return cls(
            scope_id=d["scope_id"],
            body=[IntentOp.from_json(item) for item in d["body"]],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "body": [item.to_json() for item in self.body],
        }


@dataclass
class IntentOp:
    """One Layer-A semantic node, including closed structured regions."""

    op: str
    result: Any = None  # str | list[str]
    operands: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    type: dict[str, Any] | None = None
    scope_id: str | None = None  # portable semantic scope; never a source span
    # structured-loop fields (op == "loop"); see spec/ir/structured-loop.md
    carry: list[LoopStateBinding] = field(default_factory=list)
    trip: LoopTrip | None = None
    scan: list[LoopScanInput] = field(default_factory=list)  # per-iter sliced inputs
    body: list["IntentOp"] = field(default_factory=list)
    yields: list[str] = field(default_factory=list)  # JSON key "yield"
    # effect-only binary conditional fields (op == "effect_if")
    predicate: str | None = None
    positive: EffectIfArm | None = None
    negative: EffectIfArm | None = None
    node_id: str | None = None  # canonical stable Layer-A node id

    @classmethod
    def from_json(cls, d: dict) -> "IntentOp":
        _closed_keys(
            d,
            {
                "id", "op", "result", "operands", "attrs", "type",
                "scope_id", "carry", "trip", "scan", "body", "yield",
                "predicate", "positive", "negative",
            },
            "IntentOp",
        )
        result = d.get("result")
        result_names = list(result) if isinstance(result, list) else []
        yields = list(d.get("yield", []))
        return cls(
            op=d["op"],
            node_id=d.get("id"),
            result=result,
            operands=list(d.get("operands", [])),
            attrs=dict(d.get("attrs", {})),
            type=d.get("type"),
            scope_id=d.get("scope_id"),
            carry=[
                LoopStateBinding.from_json(
                    item,
                    index=index,
                    positional_result=(result_names[index] if index < len(result_names) else None),
                    positional_yield=(yields[index] if index < len(yields) else None),
                )
                for index, item in enumerate(d.get("carry", []))
            ],
            trip=(LoopTrip.from_json(d["trip"]) if d.get("trip") is not None else None),
            scan=[LoopScanInput.from_json(item) for item in d.get("scan", [])],
            body=[cls.from_json(x) for x in d.get("body", [])],
            yields=yields,
            predicate=d.get("predicate"),
            positive=(
                EffectIfArm.from_json(d["positive"])
                if d.get("positive") is not None else None
            ),
            negative=(
                EffectIfArm.from_json(d["negative"])
                if d.get("negative") is not None else None
            ),
        )

    def to_json(self) -> dict:
        d: dict[str, Any] = {"op": self.op}
        if self.node_id is not None:
            d["id"] = self.node_id
        if self.result is not None:
            d["result"] = self.result
        if self.operands:
            d["operands"] = self.operands
        if self.attrs:
            d["attrs"] = self.attrs
        if self.type is not None:
            d["type"] = self.type
        if self.scope_id is not None:
            d["scope_id"] = self.scope_id
        if self.carry:
            d["carry"] = [item.to_json() for item in self.carry]
        if self.trip is not None:
            d["trip"] = self.trip.to_json()
        # ``scan`` is required by the canonical structured-loop wire contract,
        # and an empty scan list is semantically meaningful.  Do not collapse
        # it into field absence during a strict parse/to_json round trip.
        if self.op == "loop" or self.scan:
            d["scan"] = [item.to_json() for item in self.scan]
        if self.body:
            d["body"] = [x.to_json() for x in self.body]
        if self.yields:
            d["yield"] = self.yields
        if self.op == "effect_if":
            d["predicate"] = self.predicate
            d["positive"] = (
                self.positive.to_json() if self.positive is not None else None
            )
            d["negative"] = (
                self.negative.to_json() if self.negative is not None else None
            )
        return d


@dataclass
class IntentFunction:
    name: str
    inputs: list[Tensor] = field(default_factory=list)
    outputs: list[Tensor] = field(default_factory=list)
    body: list[IntentOp] = field(default_factory=list)
    returns: list[IntentReturn] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict) -> "IntentFunction":
        fn = d.get("intent_function", d)
        if "layer_c" in fn:
            raise ValueError(
                "IntentFunction no longer embeds C; pass pure C separately"
            )
        return cls(
            name=fn["name"],
            inputs=[Tensor.from_json(t) for t in fn.get("inputs", [])],
            outputs=[Tensor.from_json(t) for t in fn.get("outputs", [])],
            returns=[IntentReturn.from_json(item) for item in fn.get("returns", [])],
            body=[IntentOp.from_json(o) for o in fn.get("body", [])],
        )

    @classmethod
    def from_json_str(cls, s: str) -> "IntentFunction":
        return cls.from_json(json.loads(s))

    def to_json(self) -> dict:
        function = {
            "name": self.name,
            "inputs": [t.to_json() for t in self.inputs],
            "outputs": [t.to_json() for t in self.outputs],
            "body": [o.to_json() for o in self.body],
        }
        if self.returns:
            function["returns"] = [item.to_json() for item in self.returns]
        return {"intent_function": function}

    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(self.to_json(), indent=indent)
