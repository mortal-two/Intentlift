"""Read-only ``S_J`` mount and proof-irrelevant receipt construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from .semantic_materializer import semantic_materialize
from .semantic_program import (
    SEMANTIC_MATERIALIZER_PROTOCOL,
    SEMANTIC_RECEIPT_PROTOCOL,
    SemanticProgram,
)


SEMANTIC_PROGRAM_JSON = "semantic_program.json"
SEMANTIC_PROGRAM_TEXT = "semantic_program.txt"
SEMANTIC_RECEIPT_JSON = "SEMANTIC_RECEIPT.json"


@dataclass(frozen=True)
class SemanticProgramMount:
    program: SemanticProgram
    receipt_json: str

    def receipt(self) -> dict[str, Any]:
        value = json.loads(self.receipt_json)
        assert isinstance(value, dict)
        return value

    def files(self) -> dict[str, str]:
        return {
            SEMANTIC_PROGRAM_JSON: self.program.json_text,
            SEMANTIC_PROGRAM_TEXT: self.program.text,
            SEMANTIC_RECEIPT_JSON: self.receipt_json,
        }


def build_semantic_program_mount(
    j: Mapping[str, Any],
    *,
    masked: bool = False,
) -> SemanticProgramMount:
    """Freshly derive and describe one read-only J-only semantic mount."""

    program = semantic_materialize(j, masked=masked)
    j_bytes = len(
        json.dumps(
            j,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    receipt = {
        "protocol": SEMANTIC_RECEIPT_PROTOCOL,
        "status": "pass",
        "materializer": SEMANTIC_MATERIALIZER_PROTOCOL,
        "semantic_program_schema": program.schema,
        "mode": program.mode,
        "parameterized": True,
        "fresh_materialization": True,
        "semantic_input": "exact J only",
        "read_only_mount": True,
        "proof_relevant": False,
        "artifact_or_gate_input": False,
        "production_input": False,
        "excluded_dependencies": [
            "pi",
            "C valuation",
            "Source",
            "Gate observation",
            "frontend identity",
            "target API",
            "historical Replay",
        ],
        "bytes": {
            "j": j_bytes,
            "semantic_program_json": program.json_bytes,
            "semantic_program_text": program.text_bytes,
        },
        "counts": dict(program.counts),
        "mounts": {
            SEMANTIC_PROGRAM_JSON: "read-only",
            SEMANTIC_PROGRAM_TEXT: "read-only",
            SEMANTIC_RECEIPT_JSON: "read-only",
        },
    }
    receipt_text = json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    return SemanticProgramMount(program, receipt_text)


__all__ = [
    "SEMANTIC_PROGRAM_JSON",
    "SEMANTIC_PROGRAM_TEXT",
    "SEMANTIC_RECEIPT_JSON",
    "SemanticProgramMount",
    "build_semantic_program_mount",
]
