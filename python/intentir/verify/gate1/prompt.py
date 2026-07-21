"""Source-isolated Replay packets for the LLM-direct Gate1 protocol."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from .model import Gate1Artifact, SchemaError


REPLAY_PACKET_SCHEMA = "intentir.gate1.replay-packet.v1"
PUBLIC_FRONTEND_CONTEXT = {
    "triton": {
        "context_policy": "public_frontend_context_v1",
        "retrieval_scope": "public Triton documentation and public examples",
        "tool_access": ["public_document_lookup"],
    },
    "cute": {
        "context_policy": "public_frontend_context_v1",
        "retrieval_scope": "public NVIDIA CuTe DSL documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry": "scrc_entry",
        "compile_entry_contract": (
            "Define one @cute.jit scrc_entry after the requested @cute.kernel. "
            "Its positional arguments are J tensor inputs, inouts, then outputs, "
            "followed by stream; construct representation-only descriptors inside it."
        ),
    },
}

_FORBIDDEN_KEYS = frozenset(
    {
        "source_code",
        "source_span",
        "source_name",
        "location_map",
        "provenance",
        "response_text",
        "candidate",
        "compiler_diff",
        "selected_cards",
        "registry_digest",
    }
)


def _walk(value: Any, path: str = "$" ):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


def assert_source_isolated_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if a Replay packet contains a privileged side channel."""

    if not isinstance(payload, Mapping):
        raise SchemaError("Replay payload must be an object")
    for path, value in _walk(dict(payload)):
        if isinstance(value, dict):
            forbidden = sorted(set(value) & _FORBIDDEN_KEYS)
            if "source" in value and not (
                path.startswith("$.pi.routes[")
                or path.startswith("$.pi.sharing[")
            ):
                forbidden.append("source")
            if forbidden:
                raise SchemaError(
                    f"Replay payload {path} contains privileged key(s): {forbidden}"
                )
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    for token in (
        "Rule" + "Card",
        "Rule" + "Registry",
        "compose_" + "provenance",
        "selected_" + "rule_cards",
    ):
        if token in encoded:
            raise SchemaError("Replay payload contains retired control-plane data")


def build_rewriter_payload(
    artifact: Gate1Artifact,
    *,
    masked: bool = False,
    frontend_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build exactly ``J + pure C + pi + public frontend context``."""

    view = artifact.rewriter_view(masked=masked)
    frontend = view["interface"]["frontend"]
    context = (
        copy.deepcopy(dict(frontend_context))
        if frontend_context is not None
        else copy.deepcopy(PUBLIC_FRONTEND_CONTEXT.get(frontend, {}))
    )
    payload = {
        "schema": REPLAY_PACKET_SCHEMA,
        "mode": "masked_j" if masked else "normal",
        "j": view["j"],
        "c": view["schedule"],
        "pi": view["pi"],
        "harness": view["interface"],
        "frontend_context": context,
    }
    assert_source_isolated_payload(payload)
    return payload


def build_normal_rewriter_view(
    artifact: Gate1Artifact,
    *,
    frontend_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_rewriter_payload(
        artifact, masked=False, frontend_context=frontend_context
    )


def build_masked_rewriter_view(
    artifact: Gate1Artifact,
    *,
    frontend_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_rewriter_payload(
        artifact, masked=True, frontend_context=frontend_context
    )


def _instructions(*, masked: bool) -> str:
    masked_clause = (
        "J has intentionally been masked for a single representation-partition audit. "
        if masked else ""
    )
    return (
        "You are the source-blind Replay LLM. Generate one complete, self-contained "
        "Python module containing the requested public-frontend kernel. You never see "
        "the source implementation. Consume J, pure C and pi directly; pi is topology, "
        "not semantic authority. Do not output analysis, patches, placeholders, helper "
        "prose or markdown. Use only public frontend APIs and the positional opaque "
        "signature in harness. "
        + masked_clause
        + "Return Python source only."
    )


def build_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    masked: bool = False,
    frontend_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    payload = build_rewriter_payload(
        artifact, masked=masked, frontend_context=frontend_context
    )
    return _instructions(masked=masked), json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False
    )


def build_normal_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    frontend_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    return build_rewriter_prompt(
        artifact, masked=False, frontend_context=frontend_context
    )


def build_masked_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    frontend_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    return build_rewriter_prompt(
        artifact, masked=True, frontend_context=frontend_context
    )


def replay_packet_digest(payload: Mapping[str, Any]) -> str:
    assert_source_isolated_payload(payload)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "PUBLIC_FRONTEND_CONTEXT", "REPLAY_PACKET_SCHEMA",
    "assert_source_isolated_payload", "build_masked_rewriter_prompt",
    "build_masked_rewriter_view", "build_normal_rewriter_prompt",
    "build_normal_rewriter_view", "build_rewriter_payload",
    "build_rewriter_prompt", "replay_packet_digest",
]
