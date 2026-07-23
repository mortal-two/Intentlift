"""Source-isolated Replay packets for the LLM-direct Gate1 protocol."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .model import Frontend, Gate1Artifact, SchemaError
from .pi import (
    render_realization_mechanism_grammar,
    render_realization_witness_wire_grammar,
)


REPLAY_PACKET_SCHEMA = "intentir.scrc.replay-input.v1"
PUBLIC_FRONTEND_CONTEXT = {
    "triton": {
        "context_policy": "public_frontend_context_v2",
        "retrieval_scope": "public Triton documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry_contract": (
            "Define exactly one @triton.jit scrc_k0 with J.signature handles "
            "followed by the anonymous MaskC(C) slots. Keep every C constexpr "
            "slot annotated tl.constexpr. Triton compile-time-only APIs do not "
            "preserve constexpr through ordinary Python aliases: pass a c* handle "
            "directly to tl.arange and any other compile-time-only API, never copy "
            "it to a local first. A single realize instance anchored to c* is "
            "satisfied by the parameter binding and its direct routed use; do not "
            "emit an identity assignment merely to materialize it. J reduce.axes "
            "are whole-tensor logical axes, while each matching "
            "J.boundary.reductions.tile_axis is the physical frontend reduction "
            "axis and must be used at the public reduction call. Region "
            "placement='program' alone never authorizes "
            "tl.program_id. A domain-anchored realize stage in a program region "
            "whose scalar dependency path enters address_base is a program "
            "coordinate; map declaration-ordered stages of that form to "
            "tl.program_id(axis). A domain/C stage whose path enters address_lane "
            "is lane-varying and may use tl.arange. Do not vectorize a program "
            "coordinate or turn a lane coordinate into tl.program_id."
            " Every pi.refinements entry is mandatory realization topology. "
            "Resolve its auth_ref in J.boundary.realization_authorizations and "
            "materialize the authorized kind at that exact instance before its "
            "routed consumers. For signed_widen, unsigned_widen, numeric_cast, "
            "and accumulation_dtype, use a public Triton cast/to operation with "
            "the authorized target_dtype; preserve distinct refined stages and "
            "do not algebraically remove or move the conversion."
            " The generated module must obey the audited public import ABI: use "
            "`import triton`, `import triton.language as tl`, and, when a reviewed "
            "libdevice function is required, exactly "
            "`from triton.language.extra.cuda import libdevice` followed by "
            "`libdevice.<function>(...)`. Directly importing a libdevice function "
            "is rejected by the fixed module validator."
        ),
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
        "harness",
        "interface",
        "binding_map",
        "compiler_fingerprint",
        "gate1_diff",
        "prior_replay",
        "conversation",
    }
)

_REPLAY_COMPONENT_KEYS = {
    "j": frozenset(
        {
            "kernel_id", "intent", "node_ids", "boundary", "signature",
            "scopes", "storage",
        }
    ),
    "c": frozenset({"slots"}),
    "pi": frozenset(
        {"regions", "instances", "routes", "order", "sharing", "refinements"}
    ),
}
_U_F_KEYS = frozenset(
    {
        "protocol",
        "frontend",
        "replay_function",
        "context_policy",
        "retrieval_scope",
        "tool_access",
        "compile_entry",
        "compile_entry_contract",
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
    if set(payload) != {"j", "c", "pi", "u_f"}:
        raise SchemaError(
            "Replay payload root must contain exactly j, c, pi, and u_f"
        )
    for component, expected in _REPLAY_COMPONENT_KEYS.items():
        value = payload[component]
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SchemaError(
                f"Replay payload {component} must contain exactly "
                + ", ".join(sorted(expected))
            )
    u_f = payload["u_f"]
    if not isinstance(u_f, Mapping):
        raise SchemaError("Replay payload u_f must be an object")
    unknown_u_f = sorted(set(u_f) - _U_F_KEYS)
    if unknown_u_f:
        raise SchemaError(
            f"Replay payload u_f has unknown field(s): {unknown_u_f}"
        )
    required_u_f = {"protocol", "frontend", "replay_function"}
    missing_u_f = sorted(required_u_f - set(u_f))
    if missing_u_f:
        raise SchemaError(
            f"Replay payload u_f is missing field(s): {missing_u_f}"
        )
    if u_f["protocol"] != REPLAY_PACKET_SCHEMA:
        raise SchemaError("Replay payload u_f uses an unsupported protocol")
    try:
        Frontend(u_f["frontend"])
    except (TypeError, ValueError) as exc:
        raise SchemaError("Replay payload u_f has an unsupported frontend") from exc
    if u_f["replay_function"] != "scrc_k0":
        raise SchemaError("Replay payload u_f must use canonical replay_function")
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


@dataclass(frozen=True)
class ReplayPacket:
    """Typed source-isolated request; this is not a fourth proof artifact."""

    j: Mapping[str, Any]
    c: Mapping[str, Any]
    pi: Mapping[str, Any]
    u_f: Mapping[str, Any]

    def __post_init__(self) -> None:
        assert_source_isolated_payload(self.to_json(validate=False))

    def to_json(self, *, validate: bool = True) -> dict[str, Any]:
        payload = {
            "j": copy.deepcopy(dict(self.j)),
            "c": copy.deepcopy(dict(self.c)),
            "pi": copy.deepcopy(dict(self.pi)),
            "u_f": copy.deepcopy(dict(self.u_f)),
        }
        if validate:
            assert_source_isolated_payload(payload)
        return payload


def build_rewriter_packet(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    masked: bool = False,
) -> ReplayPacket:
    """Build the one canonical ``J + pi + MaskC(C) + U_f`` packet."""

    # Local import avoids a package initialization cycle: TypeCheck depends on
    # the Gate1 data model, while this builder is its first production consumer.
    from intentir.verify.typecheck import type_check

    report = type_check(artifact.j, artifact.c, artifact.pi)
    if not report.passed:
        summary = ", ".join(
            f"{item.code.value}@{item.path}" for item in report.diagnostics
        )
        raise SchemaError("Replay requires TypeCheck(J,C,pi)=PASS: " + summary)
    view = artifact.rewriter_view(masked=masked)
    frontend = Frontend(frontend)
    context = copy.deepcopy(PUBLIC_FRONTEND_CONTEXT.get(frontend.value, {}))
    context.update(
        {
            "protocol": REPLAY_PACKET_SCHEMA,
            "frontend": frontend.value,
            "replay_function": "scrc_k0",
        }
    )
    return ReplayPacket(view["j"], view["c"], view["pi"], context)


def build_rewriter_payload(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    masked: bool = False,
) -> dict[str, Any]:
    """Build exactly ``J + pi + MaskC(C) + U_f``."""

    return build_rewriter_packet(
        artifact, frontend=frontend, masked=masked
    ).to_json()


def build_normal_rewriter_view(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
) -> dict[str, Any]:
    return build_rewriter_payload(
        artifact,
        frontend=frontend,
        masked=False,
    )


def build_masked_rewriter_view(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
) -> dict[str, Any]:
    return build_rewriter_payload(
        artifact,
        frontend=frontend,
        masked=True,
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
        "not semantic authority. Treat every pi instance, route, order edge, sharing "
        "relation, representation refinement and region placement as required "
        "realization topology. Resolve each refinement auth_ref only through the "
        "matching J realization_authorization, materialize it at the refined "
        "instance before routed consumers, and never infer, omit, or move a "
        "sensitive conversion. Multiple "
        "instances anchored to one handle are distinct source stages: do not collapse "
        "their chain, discard J/C dependencies, duplicate a shared producer, "
        "or invent an undocumented stage. Obtain callable logical handles from "
        "J.signature and append the anonymous slots from MaskC(C). Obtain all "
        "logical operators, predicates, "
        "fills, constants, index relations and effects only from J. Do not output "
        "analysis, patches, placeholders, helper prose or markdown. Use only public "
        "frontend APIs and the canonical opaque handles in J.signature/MaskC(C). "
        "Pi graph "
        "reference contract: "
        + render_realization_witness_wire_grammar()
        + " Pi mechanism contract: "
        + render_realization_mechanism_grammar()
        + " "
        + masked_clause
        + "Return Python source only."
    )


def build_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    masked: bool = False,
) -> tuple[str, str]:
    payload = build_rewriter_payload(
        artifact,
        frontend=frontend,
        masked=masked,
    )
    return _instructions(masked=masked), json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False
    )


def build_normal_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
) -> tuple[str, str]:
    return build_rewriter_prompt(
        artifact,
        frontend=frontend,
        masked=False,
    )


def build_masked_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
) -> tuple[str, str]:
    return build_rewriter_prompt(
        artifact,
        frontend=frontend,
        masked=True,
    )


def replay_packet_digest(payload: Mapping[str, Any]) -> str:
    assert_source_isolated_payload(payload)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "PUBLIC_FRONTEND_CONTEXT", "REPLAY_PACKET_SCHEMA", "ReplayPacket",
    "assert_source_isolated_payload", "build_masked_rewriter_prompt",
    "build_masked_rewriter_view", "build_normal_rewriter_prompt",
    "build_normal_rewriter_view", "build_rewriter_payload",
    "build_rewriter_packet", "build_rewriter_prompt", "replay_packet_digest",
]
