"""Immutable, semantics-free identity for the formal source-aware Lift wire.

The manifest stores expected hashes.  This module only recomputes the actual
schema, fixed instructions, and request-path implementation bytes so a formal
run can fail before opening a provider connection when its wire epoch drifts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from intentir.extraction.lifters.scrc import build_scrc_lift_instructions
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema


LIFT_PROTOCOL_V2 = "llm_direct_pi_source_aware_lift_v2"
LIFT_WIRE_VERSION_V2 = "intentir.lift-wire.v2"
LIFT_PROTOCOL_V3 = "llm_direct_pi_source_aware_lift_v3"
LIFT_WIRE_VERSION_V3 = "intentir.lift-wire.v3"
LIFT_PROTOCOL_V4 = "llm_direct_pi_source_aware_lift_v4"
LIFT_WIRE_VERSION_V4 = "intentir.lift-wire.v4"
LIFT_PROTOCOL_V5 = "llm_direct_pi_source_aware_lift_v5"
LIFT_WIRE_VERSION_V5 = "intentir.lift-wire.v5"
LIFT_PROTOCOL = "llm_direct_pi_source_aware_lift_v6"
LIFT_WIRE_VERSION = "intentir.lift-wire.v6"
LIFT_CLIENT_FILES_V2 = (
    "python/intentir/extraction/providers/client.py",
    "python/intentir/extraction/lifters/scrc.py",
    "python/intentir/extraction/lifters/scrc_schema.py",
)
LIFT_CLIENT_FILES_V3 = (
    *LIFT_CLIENT_FILES_V2,
    "python/intentir/ir/intent_op_contract.py",
)
LIFT_CLIENT_FILES_V4 = LIFT_CLIENT_FILES_V3
LIFT_CLIENT_FILES_V5 = (
    *LIFT_CLIENT_FILES_V4,
    "python/intentir/verify/gate1/pi.py",
)
LIFT_CLIENT_FILES = LIFT_CLIENT_FILES_V5


class LiftWireError(ValueError):
    """The checked-in wire lock differs from the code that would make a call."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_schema_bytes() -> bytes:
    """Canonical bytes of the JSON Schema placed in Responses text.format."""

    return json.dumps(
        scrc_lift_core_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def fixed_instructions_bytes() -> bytes:
    """Exact case-independent Lift instruction bytes."""

    return build_scrc_lift_instructions().encode()


def current_lift_wire_identity(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    return {
        "version": LIFT_WIRE_VERSION,
        "schema_sha256": _sha256(canonical_schema_bytes()),
        "prompt_sha256": _sha256(fixed_instructions_bytes()),
        "client_files": [
            {
                "path": path,
                "sha256": _sha256((root / path).read_bytes()),
            }
            for path in LIFT_CLIENT_FILES
        ],
    }


def verify_lift_wire_lock(
    expected: Mapping[str, Any],
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return current identity iff the manifest lock is exact."""

    if not isinstance(expected, Mapping):
        raise LiftWireError("lift_wire must be an object")
    actual = current_lift_wire_identity(repository_root)
    if dict(expected) != actual:
        raise LiftWireError("formal Lift wire differs from the frozen manifest")
    return actual


__all__ = [
    "LIFT_CLIENT_FILES",
    "LIFT_CLIENT_FILES_V2",
    "LIFT_CLIENT_FILES_V3",
    "LIFT_CLIENT_FILES_V4",
    "LIFT_CLIENT_FILES_V5",
    "LIFT_PROTOCOL",
    "LIFT_PROTOCOL_V2",
    "LIFT_PROTOCOL_V3",
    "LIFT_PROTOCOL_V4",
    "LIFT_PROTOCOL_V5",
    "LIFT_WIRE_VERSION",
    "LIFT_WIRE_VERSION_V2",
    "LIFT_WIRE_VERSION_V3",
    "LIFT_WIRE_VERSION_V4",
    "LIFT_WIRE_VERSION_V5",
    "LiftWireError",
    "canonical_schema_bytes",
    "current_lift_wire_identity",
    "fixed_instructions_bytes",
    "verify_lift_wire_lock",
]
