"""Source-isolated Replay packets for the LLM-direct Gate1 protocol."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from .model import Frontend, Gate1Artifact, SchemaError


REPLAY_PACKET_SCHEMA = "intentir.scrc.replay-input.v3"
LEGACY_REPLAY_PACKET_SCHEMAS = frozenset(
    {"intentir.scrc.replay-input.v1", "intentir.scrc.replay-input.v2"}
)
PUBLIC_FRONTEND_CONTEXT = {
    "triton": {
        "retrieval_scope": "public Triton documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry_contract": (
            "Write one self-contained public Triton module with exactly one "
            "@triton.jit function named scrc_k0. Its parameters are the declared "
            "J.signature handles, followed in order by MaskC(C) slots whose kind "
            "is constexpr and which remain annotated tl.constexpr; kind=compiler "
            "slots belong only to the external invocation. J alone supplies "
            "semantics. Physical realization remains the Replay compiler's "
            "source-blind target choice and is checked only outside this packet."
        ),
    },
    "cute": {
        "retrieval_scope": "public NVIDIA CuTe DSL documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry": "scrc_entry",
        "compile_entry_contract": (
            "Define one @cute.jit scrc_entry after the requested @cute.kernel. "
            "Its positional arguments are J tensor inputs, inouts, then outputs, "
            "followed by stream; construct representation-only descriptors inside it."
        ),
    },
    "tilelang": {
        "retrieval_scope": "public TileLang documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry_contract": (
            "Return one self-contained Python module using exactly "
            "`import tilelang.language as T` and one `@T.prim_func` named "
            "`scrc_k0`. Express tensor ABI through public `T.Tensor` annotations, "
            "device launch through `T.Kernel`, and realization topology through "
            "public TileLang operations. C values are compile-time schedule "
            "constants, never additional logical runtime parameters."
        ),
    },
    "cuda": {
        "retrieval_scope": "public CUDA C++ programming documentation and examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry_contract": (
            "Return one standalone CUDA C++ translation unit containing exactly "
            "one `extern \"C\" __global__` kernel named `scrc_k0`. Its positional "
            "runtime parameters follow J.signature; C values are compiled "
            "schedule constants and are not runtime parameters. Use only public "
            "CUDA language constructs and intrinsics. Do not emit host code, "
            "launch syntax, file I/O, inline PTX, build directives, or `main`."
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
            "scopes", "storage", "dimension_constraints",
            "value_dimension_relations",
            "dimension_relations",
        }
    ),
    "c": frozenset({"slots"}),
}
_LEGACY_PI_KEYS = frozenset(
    {
        "regions",
        "instances",
        "routes",
        "program_axes",
        "program_linearizations",
        "route_extent_transforms",
        "order",
        "sharing",
        "refinements",
        "iterations",
        "loop_carriers",
        "physical_iterations",
        "physical_loop_carriers",
        "terminal_controls",
    }
)
_HISTORICAL_J_KEY_SETS = frozenset(
    {
        _REPLAY_COMPONENT_KEYS["j"] - {"dimension_relations"},
        _REPLAY_COMPONENT_KEYS["j"]
        - {"dimension_relations", "value_dimension_relations"},
    }
)
_U_F_KEYS = frozenset(
    {
        "protocol",
        "frontend",
        "replay_function",
        "retrieval_scope",
        "tool_access",
        "compile_entry",
        "compile_entry_contract",
    }
)
_U_F_PACKET_KEYS = frozenset({"protocol", "frontend", "replay_function"})


def _walk(value: Any, path: str = "$" ):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


def _validate_source_isolated_payload(
    payload: Mapping[str, Any],
    *,
    require_active_context: bool,
) -> None:
    if not isinstance(payload, Mapping):
        raise SchemaError("Replay payload must be an object")
    root = set(payload)
    active_root = {"j", "c", "u_f"}
    historical_root = {"j", "c", "pi", "u_f"}
    if root != active_root and not (
        not require_active_context and root == historical_root
    ):
        raise SchemaError(
            "Replay payload root must contain exactly j, c, and u_f"
        )
    for component, expected in _REPLAY_COMPONENT_KEYS.items():
        value = payload[component]
        actual = set(value) if isinstance(value, Mapping) else set()
        historical_j = (
            not require_active_context
            and component == "j"
            and frozenset(actual) in _HISTORICAL_J_KEY_SETS
        )
        if not isinstance(value, Mapping) or (
            actual != expected and not historical_j
        ):
            raise SchemaError(
                f"Replay payload {component} must contain exactly "
                + ", ".join(sorted(expected))
            )
    if "pi" in payload:
        pi = payload["pi"]
        actual_pi = set(pi) if isinstance(pi, Mapping) else set()
        historical_pi_sets = {
            _LEGACY_PI_KEYS,
            _LEGACY_PI_KEYS - {"program_linearizations"},
            _LEGACY_PI_KEYS - {"program_axes", "program_linearizations"},
        }
        if not isinstance(pi, Mapping) or frozenset(actual_pi) not in historical_pi_sets:
            raise SchemaError("historical Replay payload pi has an invalid closed root")
    u_f = payload["u_f"]
    if not isinstance(u_f, Mapping):
        raise SchemaError("Replay payload u_f must be an object")
    unknown_u_f = sorted(set(u_f) - _U_F_KEYS)
    if unknown_u_f:
        raise SchemaError(
            f"Replay payload u_f has unknown field(s): {unknown_u_f}"
        )
    required_u_f = set(_U_F_PACKET_KEYS)
    missing_u_f = sorted(required_u_f - set(u_f))
    if missing_u_f:
        raise SchemaError(
            f"Replay payload u_f is missing field(s): {missing_u_f}"
        )
    if u_f["protocol"] != REPLAY_PACKET_SCHEMA and not (
        not require_active_context
        and u_f["protocol"] in LEGACY_REPLAY_PACKET_SCHEMAS
    ):
        raise SchemaError("Replay payload u_f uses an unsupported protocol")
    try:
        frontend = Frontend(u_f["frontend"])
    except (TypeError, ValueError) as exc:
        raise SchemaError("Replay payload u_f has an unsupported frontend") from exc
    if u_f["replay_function"] != "scrc_k0":
        raise SchemaError("Replay payload u_f must use canonical replay_function")
    if require_active_context:
        fixed_context = PUBLIC_FRONTEND_CONTEXT.get(frontend.value, {})
        missing_context = sorted(set(fixed_context) - set(u_f))
        if missing_context:
            raise SchemaError(
                "Replay payload u_f is missing fixed context field(s): "
                f"{missing_context}"
            )
        modified_context = sorted(
            key for key, value in fixed_context.items() if u_f.get(key) != value
        )
        if modified_context:
            raise SchemaError(
                "Replay payload u_f does not match the fixed public frontend "
                f"context field(s): {modified_context}"
            )
    for path, value in _walk(dict(payload)):
        if isinstance(value, dict):
            forbidden = sorted(set(value) & _FORBIDDEN_KEYS)
            legacy_pi_source = (
                not require_active_context
                and (
                    path.startswith("$.pi.routes[")
                    or path.startswith("$.pi.route_extent_transforms[")
                    or path.startswith("$.pi.sharing[")
                )
            )
            if "source" in value and not legacy_pi_source:
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


def assert_source_isolated_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if a fresh Replay packet is not source-isolated and current."""

    _validate_source_isolated_payload(payload, require_active_context=True)


@dataclass(frozen=True)
class ReplayPacket:
    """Typed source-isolated request; this is not a fourth proof artifact."""

    j: Mapping[str, Any]
    c: Mapping[str, Any]
    u_f: Mapping[str, Any]

    def __post_init__(self) -> None:
        assert_source_isolated_payload(self.to_json(validate=False))

    def to_json(self, *, validate: bool = True) -> dict[str, Any]:
        payload = {
            "j": copy.deepcopy(dict(self.j)),
            "c": copy.deepcopy(dict(self.c)),
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
    """Build the one canonical ``J + MaskC(C) + U_f`` Replay packet."""

    # Local import avoids a package initialization cycle: TypeCheck depends on
    # the Gate1 data model, while this builder is its first production consumer.
    from intentir.verify.typecheck import type_check

    artifact = canonicalize_gate1_artifact(artifact)
    report = type_check(
        artifact.j,
        artifact.c,
        artifact.pi,
        require_program_axes=artifact.wire_schema == artifact.SCHEMA,
    )
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
    return ReplayPacket(view["j"], view["c"], context)


def build_rewriter_payload(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    masked: bool = False,
) -> dict[str, Any]:
    """Build exactly ``J + MaskC(C) + U_f``."""

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


def _concise_instructions(*, frontend: Frontend, masked: bool) -> str:
    output = "CUDA C++" if frontend is Frontend.CUDA else "Python"
    masked_clause = (
        " J is masked for this representation audit; do not reconstruct it."
        if masked
        else ""
    )
    return (
        "You are one fresh source-blind Replay coding agent. Return one complete "
        f"self-contained {output} source module and no prose. J is the sole semantic "
        "authority and C exposes only anonymous slots. Compile the whole typed J "
        "without adding value semantics. Physical layout, program mapping, loops, "
        "control and target APIs are your source-blind compiler choices, not extra "
        "semantic inputs. Use only public frontend APIs and the canonical "
        "J.signature/MaskC(C) handles. Source, Gate feedback, previous Replay and "
        "kernel templates are unavailable. The packet's U_f gives only the callable "
        "ABI."
        + masked_clause
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
    frontend = Frontend(frontend)
    return _concise_instructions(frontend=frontend, masked=masked), json.dumps(
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


__all__ = [
    "PUBLIC_FRONTEND_CONTEXT", "REPLAY_PACKET_SCHEMA",
    "ReplayPacket", "assert_source_isolated_payload",
    "build_masked_rewriter_prompt",
    "build_masked_rewriter_view", "build_normal_rewriter_prompt",
    "build_normal_rewriter_view", "build_rewriter_payload",
    "build_rewriter_packet", "build_rewriter_prompt",
]
