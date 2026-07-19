"""IntentIR in-memory model.

1:1 with the IntentIR JSON spec (.trellis/spec/ir/op-signatures.md) and with the
``intent`` MLIR dialect. Round-trips JSON so the LLM-produced artifact can be
inspected, validated, and lowered to the dialect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: list[Any]  # entries may be ints or symbolic dim names (e.g. "Sq")

    @classmethod
    def from_json(cls, d: dict) -> "Tensor":
        return cls(name=d["name"], dtype=d["dtype"], shape=list(d["shape"]))

    def to_json(self) -> dict:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass
class LoopValueType:
    """Portable type carried across a semantic loop iteration.

    This is deliberately a tensor/value type, not a frontend storage decision:
    register/shared/global placement belongs to C.strategy and the rule set.
    """

    dtype: str
    shape: list[Any] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "LoopValueType":
        return cls(dtype=d["dtype"], shape=list(d.get("shape", [])))

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
    """Semantic loop trip description.

    ``tile_hint`` is accepted only as a legacy non-SCRC compatibility field.
    New SCRC J stores the semantic axis here and keeps tile choices in C.
    """

    axis: str
    tile_hint: int | str | None = None

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "LoopTrip":
        return cls(axis=d["axis"], tile_hint=d.get("tile_hint"))

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"axis": self.axis}
        if self.tile_hint is not None:
            result["tile_hint"] = self.tile_hint
        return result

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
        return cls(name=d["name"], source=d["from"], axis=d["axis"])

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "from": self.source, "axis": self.axis}

    def __getitem__(self, key: str) -> Any:
        return getattr(self, "source" if key == "from" else key)


@dataclass
class IntentOp:
    """One IntentIR op. Generic enough for all atomic ops plus ``loop``."""

    op: str
    result: Any = None  # str | list[str]
    operands: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    type: dict[str, Any] | None = None
    layer_b: dict[str, Any] = field(default_factory=dict)  # portable constraints
    out: str | None = None  # destination-passing target
    scope_id: str | None = None  # portable semantic scope; never a source span
    # structured-loop fields (op == "loop"); see spec/ir/structured-loop.md
    carry: list[LoopStateBinding] = field(default_factory=list)
    trip: LoopTrip | None = None
    scan: list[LoopScanInput] = field(default_factory=list)  # per-iter sliced inputs
    body: list["IntentOp"] = field(default_factory=list)
    yields: list[str] = field(default_factory=list)  # JSON key "yield"

    @classmethod
    def from_json(cls, d: dict) -> "IntentOp":
        result = d.get("result")
        result_names = list(result) if isinstance(result, list) else []
        yields = list(d.get("yield", []))
        return cls(
            op=d["op"],
            result=result,
            operands=list(d.get("operands", [])),
            attrs=dict(d.get("attrs", {})),
            type=d.get("type"),
            layer_b=dict(d.get("layer_b", {})),
            out=d.get("out"),
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
        )

    def to_json(self) -> dict:
        d: dict[str, Any] = {"op": self.op}
        if self.result is not None:
            d["result"] = self.result
        if self.operands:
            d["operands"] = self.operands
        if self.attrs:
            d["attrs"] = self.attrs
        if self.type is not None:
            d["type"] = self.type
        if self.layer_b:
            d["layer_b"] = self.layer_b
        if self.out is not None:
            d["out"] = self.out
        if self.scope_id is not None:
            d["scope_id"] = self.scope_id
        if self.carry:
            d["carry"] = [item.to_json() for item in self.carry]
        if self.trip is not None:
            d["trip"] = self.trip.to_json()
        if self.scan:
            d["scan"] = [item.to_json() for item in self.scan]
        if self.body:
            d["body"] = [x.to_json() for x in self.body]
        if self.yields:
            d["yield"] = self.yields
        return d


@dataclass
class IntentFunction:
    name: str
    inputs: list[Tensor] = field(default_factory=list)
    outputs: list[Tensor] = field(default_factory=list)
    layer_c: dict[str, Any] = field(default_factory=dict)  # non-binding schedule
    body: list[IntentOp] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict) -> "IntentFunction":
        fn = d.get("intent_function", d)
        return cls(
            name=fn["name"],
            inputs=[Tensor.from_json(t) for t in fn.get("inputs", [])],
            outputs=[Tensor.from_json(t) for t in fn.get("outputs", [])],
            layer_c=dict(fn.get("layer_c", {})),
            body=[IntentOp.from_json(o) for o in fn.get("body", [])],
        )

    @classmethod
    def from_json_str(cls, s: str) -> "IntentFunction":
        return cls.from_json(json.loads(s))

    def to_json(self) -> dict:
        return {
            "intent_function": {
                "name": self.name,
                "inputs": [t.to_json() for t in self.inputs],
                "outputs": [t.to_json() for t in self.outputs],
                "layer_c": self.layer_c,
                "body": [o.to_json() for o in self.body],
            }
        }

    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(self.to_json(), indent=indent)
