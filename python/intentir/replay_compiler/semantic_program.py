"""Immutable parameterized semantic-program view derived from exact J."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping


SEMANTIC_PROGRAM_SCHEMA = "intentir.replay.semantic-program.v2"
SEMANTIC_MATERIALIZER_PROTOCOL = "intentir.replay.semantic-materializer.v2"
SEMANTIC_RECEIPT_PROTOCOL = "intentir.replay.semantic-receipt.v1"


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


@dataclass(frozen=True)
class SemanticProgram:
    """One immutable ``S_J``.

    Nested JSON is retained as encoded text rather than mutable containers.  A
    caller may request a fresh decoded copy, but cannot mutate the materialized
    program held by this object.
    """

    mode: str
    json_text: str = field(repr=False)
    text: str = field(repr=False)
    counts: tuple[tuple[str, int], ...] = ()
    schema: str = SEMANTIC_PROGRAM_SCHEMA

    def __post_init__(self) -> None:
        if self.mode not in {"normal", "masked"}:
            raise ValueError("SemanticProgram.mode must be normal or masked")
        decoded = json.loads(self.json_text)
        if not isinstance(decoded, dict):
            raise TypeError("SemanticProgram JSON must be an object")
        if decoded.get("schema") != self.schema:
            raise ValueError("SemanticProgram schema does not match its payload")
        if decoded.get("mode") != self.mode:
            raise ValueError("SemanticProgram mode does not match its payload")
        if not self.text.endswith("\n"):
            raise ValueError("SemanticProgram text must end with a newline")
        if len({name for name, _value in self.counts}) != len(self.counts):
            raise ValueError("SemanticProgram counts must have unique names")
        if any(value < 0 for _name, value in self.counts):
            raise ValueError("SemanticProgram counts cannot be negative")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        mode: str,
        text: str,
        counts: Mapping[str, int],
    ) -> "SemanticProgram":
        return cls(
            mode=mode,
            json_text=_json_text(dict(payload)),
            text=text if text.endswith("\n") else text + "\n",
            counts=tuple(sorted((str(key), int(value)) for key, value in counts.items())),
        )

    def to_json(self) -> dict[str, Any]:
        value = json.loads(self.json_text)
        assert isinstance(value, dict)
        return value

    def count(self, name: str) -> int:
        return dict(self.counts).get(name, 0)

    @property
    def json_bytes(self) -> int:
        return len(self.json_text.encode("utf-8"))

    @property
    def text_bytes(self) -> int:
        return len(self.text.encode("utf-8"))


__all__ = [
    "SEMANTIC_MATERIALIZER_PROTOCOL",
    "SEMANTIC_PROGRAM_SCHEMA",
    "SEMANTIC_RECEIPT_PROTOCOL",
    "SemanticProgram",
]
