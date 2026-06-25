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
class IntentOp:
    """One IntentIR op. Generic enough for all atomic ops plus ``loop``."""

    op: str
    result: Any = None  # str | list[str]
    operands: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    type: dict[str, Any] | None = None
    layer_b: dict[str, Any] = field(default_factory=dict)  # portable constraints
    out: str | None = None  # destination-passing target
    # structured-loop fields (op == "loop"); see spec/ir/structured-loop.md
    carry: list[dict[str, Any]] = field(default_factory=list)
    trip: dict[str, Any] | None = None
    scan: list[dict[str, Any]] = field(default_factory=list)  # per-iter sliced inputs
    body: list["IntentOp"] = field(default_factory=list)
    yields: list[str] = field(default_factory=list)  # JSON key "yield"

    @classmethod
    def from_json(cls, d: dict) -> "IntentOp":
        return cls(
            op=d["op"],
            result=d.get("result"),
            operands=list(d.get("operands", [])),
            attrs=dict(d.get("attrs", {})),
            type=d.get("type"),
            layer_b=dict(d.get("layer_b", {})),
            out=d.get("out"),
            carry=list(d.get("carry", [])),
            trip=d.get("trip"),
            scan=list(d.get("scan", [])),
            body=[cls.from_json(x) for x in d.get("body", [])],
            yields=list(d.get("yield", [])),
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
        if self.carry:
            d["carry"] = self.carry
        if self.trip is not None:
            d["trip"] = self.trip
        if self.scan:
            d["scan"] = self.scan
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
