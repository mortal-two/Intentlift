"""Source-isolated Replay packets for the LLM-direct Gate1 protocol."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from .model import Frontend, Gate1Artifact, SchemaError
from .pi import (
    render_realization_mechanism_grammar,
    render_realization_witness_wire_grammar,
)


REPLAY_PACKET_SCHEMA = "intentir.scrc.replay-input.v1"
PUBLIC_FRONTEND_CONTEXT = {
    "triton": {
        "context_policy": "public_frontend_context_v7",
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
            "axis and must be used at the public reduction call. It is the local "
            "tile axis after program/grid axes select a program instance, not the "
            "whole-tensor logical ordinal in J reduce.attrs.axes; never copy or "
            "offset that logical ordinal to guess it. "
            "Each J.boundary access is closed: when predicate is absent, emit an "
            "unconditional load or store; never synthesize a defensive or "
            "redundant mask from a domain, grid extent, C slot, or safety guess. "
            "Region placement='program' alone never authorizes "
            "tl.program_id. A domain-anchored realize stage in a program region "
            "whose scalar dependency path enters address_base is a program "
            "coordinate. Assign axes by first declaration of each distinct J "
            "domain semantic_ref: distinct stages anchored to the same d* remain "
            "distinct materializations of the same program axis and never allocate "
            "another axis. Only a distinct domain anchor may allocate the next "
            "tl.program_id(axis). A domain/C stage whose path enters address_lane "
            "is lane-varying and may use tl.arange. Do not vectorize a program "
            "coordinate or turn a lane coordinate into tl.program_id. Preserve "
            "every address_base -> address_lane route, then determine pointer "
            "operation count only from J.signature carrier kind plus pi routes. "
            "If a pointer/tensor p* carrier directly enters address_base, the base "
            "is already a pointer/address operation; its route to address_lane "
            "requires a second pointer/address operation for the lane contribution. "
            "If that carrier bypasses address_base and enters address_lane or the "
            "final address stage, while address_base receives only integer/domain/C "
            "contributions, preserve and merge the integer base and lane stages "
            "before exactly one pointer/address operation. In both cases preserve "
            "the program_id even if its grid extent is one. Never choose from the "
            "address_base -> address_lane edge alone, and never infer from Source "
            "names, paths, or text."
            " For a canonical binary J contraction whose per-program operands and "
            "result are rank-2 floating tiles with exactly one contracted label, "
            "and whose declared operand/result dtypes are accepted by public "
            "tl.dot, one pi realize instance anchored to that contraction node "
            "maps to one direct tl.dot realization (using the declared accumulation "
            "dtype). Do not replace that single stage with broadcasted outer-product "
            "multiplication plus tl.sum: those are additional compiler-visible "
            "stages. This direct rule does not apply to outer products, batched or "
            "otherwise unsupported contractions, unsupported tile/dtype contracts, "
            "or a pi chain that explicitly declares multiple contraction-anchored "
            "realize stages; never force tl.dot by changing J or dropping pi stages."
            " Every pi.iterations entry is mandatory realization topology. "
            "Resolve loop_instance to its J structured loop and tile_instance "
            "to the matching C tile slot. For induction_basis=extent_step, "
            "materialize a zero-based loop whose upper bound is the J trip extent "
            "and whose step is the C tile. For induction_basis=tile_ordinal, "
            "materialize a zero-based unit-step loop whose upper bound is the "
            "trip extent ceiling-divided by the matching C tile. Preserve this "
            "distinction even when "
            "the induction variable itself is unused: never insert or remove the "
            "trip-count ceiling division, and never choose a basis from an op "
            "family, concrete value, name, or frontend convention."
            " Every pi.refinements entry is mandatory realization topology. "
            "Resolve its auth_ref in J.boundary.realization_authorizations and "
            "materialize the authorized kind at that exact instance before its "
            "routed consumers. For signed_widen, unsigned_widen, numeric_cast, "
            "and accumulation_dtype, use a public Triton cast/to operation with "
            "the explicitly authorized target_dtype; preserve distinct refined "
            "stages and do not algebraically remove or move the conversion. Never "
            "propagate one refinement to a sibling parameter/instance that merely "
            "shares a dtype or address expression. Signature dtypes are otherwise "
            "authoritative: do not guess a stride/index widening or emit `.to` for "
            "a p* carrier without its own routed refinement or an explicit J cast."
            " A J.signature p* with logical_ref=null has no logical denotation. It "
            "may affect the body only through an instance anchored to that p* and "
            "a declared route to its realization consumer; if no such route exists, "
            "keep the ABI parameter but do not read it in the kernel body."
            " The generated module must obey the audited public import ABI: use "
            "`import triton`, `import triton.language as tl`, and, when a reviewed "
            "libdevice function is required, exactly "
            "`from triton.language.extra.cuda import libdevice` followed by "
            "`libdevice.<function>(...)`. Directly importing a libdevice function "
            "is rejected by the fixed module validator. In pinned Triton 3.6, "
            "spell tanh/tan/atan/atan2 as libdevice calls, elementwise power as "
            "the public `x ** y` operator, and isnan as `x != x`; `tl.tanh`, "
            "`tl.tan`, `tl.atan`, `tl.atan2`, `tl.pow`, and `tl.isnan` do not "
            "exist and are rejected before compilation."
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
    "tilelang": {
        "context_policy": "public_frontend_context_v1",
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
        "context_policy": "public_frontend_context_v1",
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

# Stored packet public-context digests and historical receipt digests are
# different evidence domains.  Existing receipt policies all hash the complete
# u_f object; changing that basis requires an explicit future policy version.
PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY = MappingProxyType(
    {
        "triton": "public_frontend_context_v7",
        "cute": "public_frontend_context_v1",
        "tilelang": "public_frontend_context_v1",
        "cuda": "public_frontend_context_v1",
    }
)
PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY = MappingProxyType(
    {
        "triton": MappingProxyType(
            {
                "public_frontend_context_v6": (
                    "5e180f9389fb124735f92ab4c5728e10121fa42bb51bb239902f9676e7f63339"
                ),
                "public_frontend_context_v7": (
                    "2f71b09d4cf3d25cb413b24252a59720822dd3a42b9f8af9eb606a36bfee9613"
                ),
            }
        ),
        "cute": MappingProxyType(
            {
                "public_frontend_context_v1": (
                    "469bcb400c5f377d3b92a68a870189f0698bdd05a2154ccba16feed0502716dc"
                )
            }
        ),
        "tilelang": MappingProxyType(
            {
                "public_frontend_context_v1": (
                    "41d26224d6682b12f7761e6a4c6f61f3f9e29d5ad5bd4b3b473bdfdacbcf91f5"
                )
            }
        ),
        "cuda": MappingProxyType(
            {
                "public_frontend_context_v1": (
                    "97d17ef646dacef8b198afd70d09221a6c59aed6816e89ee19559aff3a22f1b3"
                )
            }
        ),
    }
)
PUBLIC_FRONTEND_CONTEXT_RECEIPT_HISTORY = MappingProxyType(
    {
        "triton": MappingProxyType(
            {
                "public_frontend_context_v6": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "33f1d58a4c1e762fae6b8ff27c100adfef21bcc1d8a149409f85f8b4c7a281ec"
                        ),
                    }
                ),
                "public_frontend_context_v7": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "91eb3db3fb27a166c2fecb3e02c05ac8131be3b93fa7576272814d3050e388c6"
                        ),
                    }
                ),
            }
        ),
        "cute": MappingProxyType(
            {
                "public_frontend_context_v1": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "13c3f240eb4bd19bfbbc6982e3058ebb4ecc37c139a1e483cff0cff6a811bb37"
                        ),
                    }
                )
            }
        ),
        "tilelang": MappingProxyType(
            {
                "public_frontend_context_v1": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "b54d1d1e65f5ad35a39b47aa43b7bee9eb68e4067c82d32e1b7739bc99c2660e"
                        ),
                    }
                )
            }
        ),
        "cuda": MappingProxyType(
            {
                "public_frontend_context_v1": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "83d5c95bd2c9bc28329e42987c4eb9628395990a5b2bb8c4d07aad81fb13a530"
                        ),
                    }
                )
            }
        ),
    }
)

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
        {
            "regions",
            "instances",
            "routes",
            "order",
            "sharing",
            "refinements",
            "iterations",
        }
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
_U_F_PACKET_KEYS = frozenset({"protocol", "frontend", "replay_function"})


def _canonical_mapping_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def public_frontend_context_digest(context: Mapping[str, Any]) -> str:
    """Digest public-context content independently of any receipt convention."""

    if not isinstance(context, Mapping):
        raise SchemaError("public frontend context must be an object")
    return _canonical_mapping_digest(context)


def audit_public_frontend_context_receipt(
    *,
    frontend: Frontend | str,
    context_policy: str,
    context_sha256: str,
) -> dict[str, str]:
    """Audit an exact historical receipt without upgrading its context version."""

    try:
        frontend = Frontend(frontend)
    except (TypeError, ValueError) as exc:
        raise SchemaError("public frontend context receipt has unknown frontend") from exc
    history = PUBLIC_FRONTEND_CONTEXT_RECEIPT_HISTORY.get(frontend.value)
    if history is None:
        raise SchemaError(
            "public frontend context receipt has no versioned digest ledger for "
            f"{frontend.value!r}"
        )
    policy = history.get(context_policy)
    if policy is None:
        raise SchemaError(
            "public frontend context receipt has unknown context_policy "
            f"{context_policy!r}"
        )
    expected = str(policy["context_sha256"])
    if context_sha256 != expected:
        raise SchemaError(
            "public frontend context receipt digest does not match its declared "
            f"policy {context_policy!r}"
        )
    active = PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY[frontend.value]
    return {
        "frontend": frontend.value,
        "context_policy": context_policy,
        "context_sha256": context_sha256,
        "receipt_hash_basis": str(policy["receipt_hash_basis"]),
        "status": "current" if context_policy == active else "historical",
    }


def audit_public_frontend_context_payload(
    *,
    frontend: Frontend | str,
    context_policy: str,
    public_context_sha256: str,
) -> dict[str, str]:
    """Audit public context embedded in a stored packet, not its old receipt."""

    try:
        frontend = Frontend(frontend)
    except (TypeError, ValueError) as exc:
        raise SchemaError("public frontend context payload has unknown frontend") from exc
    history = PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY.get(frontend.value)
    if history is None or context_policy not in history:
        raise SchemaError(
            "public frontend context payload has unknown context_policy "
            f"{context_policy!r}"
        )
    if public_context_sha256 != history[context_policy]:
        raise SchemaError(
            "public frontend context payload digest does not match its declared "
            f"policy {context_policy!r}"
        )
    active = PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY[frontend.value]
    return {
        "frontend": frontend.value,
        "context_policy": context_policy,
        "public_context_sha256": public_context_sha256,
        "status": "current" if context_policy == active else "historical",
    }


def _assert_active_public_frontend_context(frontend: Frontend) -> None:
    active = PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY.get(frontend.value)
    if active is None:
        return
    context = PUBLIC_FRONTEND_CONTEXT.get(frontend.value)
    if not isinstance(context, Mapping) or context.get("context_policy") != active:
        raise SchemaError(
            f"active public frontend context policy drifted for {frontend.value!r}"
        )
    audit_public_frontend_context_payload(
        frontend=frontend,
        context_policy=active,
        public_context_sha256=public_frontend_context_digest(context),
    )


def _public_context_from_u_f(u_f: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in u_f.items()
        if key not in _U_F_PACKET_KEYS
    }


def build_public_frontend_context_receipt(
    u_f: Mapping[str, Any],
) -> dict[str, str]:
    """Build a receipt using the hash basis frozen for its declared policy."""

    if not isinstance(u_f, Mapping):
        raise SchemaError("public frontend context receipt input must be an object")
    try:
        frontend = Frontend(u_f.get("frontend"))
    except (TypeError, ValueError) as exc:
        raise SchemaError("public frontend context receipt has unknown frontend") from exc
    context_policy = u_f.get("context_policy")
    if not isinstance(context_policy, str):
        raise SchemaError("public frontend context receipt requires context_policy")
    history = PUBLIC_FRONTEND_CONTEXT_RECEIPT_HISTORY.get(frontend.value)
    policy = history.get(context_policy) if history is not None else None
    if policy is None:
        raise SchemaError(
            "public frontend context receipt has unknown context_policy "
            f"{context_policy!r}"
        )
    public_context = _public_context_from_u_f(u_f)
    audit_public_frontend_context_payload(
        frontend=frontend,
        context_policy=context_policy,
        public_context_sha256=public_frontend_context_digest(public_context),
    )
    basis = str(policy["receipt_hash_basis"])
    if basis == "full_u_f":
        digest = _canonical_mapping_digest(u_f)
    elif basis == "public_context":
        digest = public_frontend_context_digest(public_context)
    else:  # pragma: no cover - frozen ledger construction invariant
        raise SchemaError(f"unknown Replay receipt hash basis {basis!r}")
    return audit_public_frontend_context_receipt(
        frontend=frontend,
        context_policy=context_policy,
        context_sha256=digest,
    )

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
    required_u_f = set(_U_F_PACKET_KEYS)
    missing_u_f = sorted(required_u_f - set(u_f))
    if missing_u_f:
        raise SchemaError(
            f"Replay payload u_f is missing field(s): {missing_u_f}"
        )
    if u_f["protocol"] != REPLAY_PACKET_SCHEMA:
        raise SchemaError("Replay payload u_f uses an unsupported protocol")
    try:
        frontend = Frontend(u_f["frontend"])
    except (TypeError, ValueError) as exc:
        raise SchemaError("Replay payload u_f has an unsupported frontend") from exc
    if u_f["replay_function"] != "scrc_k0":
        raise SchemaError("Replay payload u_f must use canonical replay_function")
    if require_active_context:
        _assert_active_public_frontend_context(frontend)
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
        if frontend.value in PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY:
            receipt = build_public_frontend_context_receipt(u_f)
            if receipt["status"] != "current":  # pragma: no cover
                raise SchemaError("fresh Replay payload uses a historical context")
    else:
        context_policy = u_f.get("context_policy")
        if not isinstance(context_policy, str):
            raise SchemaError(
                "historical Replay payload u_f requires context_policy"
            )
        audit_public_frontend_context_payload(
            frontend=frontend,
            context_policy=context_policy,
            public_context_sha256=public_frontend_context_digest(
                _public_context_from_u_f(u_f)
            ),
        )
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


def assert_source_isolated_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if a fresh Replay packet is not source-isolated and current."""

    _validate_source_isolated_payload(payload, require_active_context=True)


def _canonical_replay_packet_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit_replay_packet_digest(
    payload: Mapping[str, Any],
    *,
    expected_digest: str,
) -> dict[str, str]:
    """Audit a stored current or historical packet without accepting it as fresh."""

    _validate_source_isolated_payload(payload, require_active_context=False)
    actual = _canonical_replay_packet_digest(payload)
    if actual != expected_digest:
        raise SchemaError("stored Replay packet digest mismatch")
    u_f = payload["u_f"]
    context_audit = audit_public_frontend_context_payload(
        frontend=str(u_f["frontend"]),
        context_policy=str(u_f["context_policy"]),
        public_context_sha256=public_frontend_context_digest(
            _public_context_from_u_f(u_f)
        ),
    )
    return {**context_audit, "packet_sha256": actual}


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

    artifact = canonicalize_gate1_artifact(artifact)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    if not report.passed:
        summary = ", ".join(
            f"{item.code.value}@{item.path}" for item in report.diagnostics
        )
        raise SchemaError("Replay requires TypeCheck(J,C,pi)=PASS: " + summary)
    view = artifact.rewriter_view(masked=masked)
    frontend = Frontend(frontend)
    _assert_active_public_frontend_context(frontend)
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


def _instructions(*, frontend: Frontend, masked: bool) -> str:
    masked_clause = (
        "J has intentionally been masked for a single representation-partition audit. "
        if masked else ""
    )
    output_contract = (
        "Return CUDA C++ source only."
        if frontend is Frontend.CUDA
        else "Return Python source only."
    )
    frontend_realization_clause = (
        "Preserve every address_base -> address_lane route. If a pointer/tensor "
        "J.signature carrier directly enters address_base, materialize one base "
        "pointer/address operation and a second lane pointer/address operation. "
        "If the carrier bypasses address_base and enters address_lane or the final "
        "address stage while the base has only integer/domain/C inputs, merge the "
        "integer contributions before exactly one pointer/address operation. Decide "
        "only from J.signature kinds and pi carrier routes, never from Source names, "
        "paths, text, or the base-to-lane edge alone. Program axis identity belongs "
        "to the J domain anchor, not instance multiplicity: materialize repeated "
        "program-coordinate instances on one d* as distinct occurrences of the "
        "same axis, never as a new axis. For one pi realize instance anchored to a "
        "canonical, public-tl.dot-supported rank-2 floating J contraction with one "
        "contracted label, use one direct tl.dot with the declared accumulation "
        "dtype; do not expand it into outer-product multiply plus tl.sum. Do not "
        "apply this spelling outside those support conditions or when pi explicitly "
        "declares multiple contraction stages. "
        if frontend is Frontend.TRITON
        else ""
    )
    return (
        "You are the source-blind Replay LLM. Generate one complete, self-contained "
        f"{frontend.value} public-frontend kernel module. You never see "
        "the source implementation. Consume J, pure C and pi directly; pi is topology, "
        "not semantic authority. Treat every pi instance, route, order edge, sharing "
        "relation, iteration relation, representation refinement and region placement as required "
        "realization topology. Resolve each refinement auth_ref only through the "
        "matching J realization_authorization, materialize it at the refined "
        "instance before routed consumers, and never infer, omit, move, or copy a "
        "sensitive conversion onto a sibling instance. A logical_ref=null p* may "
        "be read only when pi contains its anchored instance and a route from that "
        "instance to a realization consumer; otherwise it remains an unused ABI "
        "parameter. Multiple "
        "instances anchored to one handle are distinct source stages: do not collapse "
        "their chain, discard J/C dependencies, duplicate a shared producer, "
        "or invent an undocumented stage. Obtain callable logical handles from "
        "J.signature and append the anonymous slots from MaskC(C). Obtain all "
        "logical operators, predicates, fills, constants, index relations and "
        "effects only from J. Preserve explicit J stages: reduce(sum) followed by "
        "division remains sum then division, never mean or a generated sum-of-ones "
        "count. A boundary access with no predicate is unconditional: "
        "never derive a redundant mask from domains, grid extents, C, or defensive "
        "safety. "
        + frontend_realization_clause
        + "Do not output "
        "analysis, patches, placeholders, helper prose or markdown. Use only public "
        "frontend APIs and the canonical opaque handles in J.signature/MaskC(C). "
        "Pi graph "
        "reference contract: "
        + render_realization_witness_wire_grammar()
        + " Pi mechanism contract: "
        + render_realization_mechanism_grammar()
        + " "
        + masked_clause
        + output_contract
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
    return _instructions(frontend=frontend, masked=masked), json.dumps(
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
    return _canonical_replay_packet_digest(payload)


__all__ = [
    "PUBLIC_FRONTEND_CONTEXT", "PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY",
    "PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY",
    "PUBLIC_FRONTEND_CONTEXT_RECEIPT_HISTORY", "REPLAY_PACKET_SCHEMA",
    "ReplayPacket", "audit_public_frontend_context_payload",
    "audit_public_frontend_context_receipt",
    "assert_source_isolated_payload", "audit_replay_packet_digest",
    "build_masked_rewriter_prompt",
    "build_masked_rewriter_view", "build_normal_rewriter_prompt",
    "build_normal_rewriter_view", "build_rewriter_payload",
    "build_public_frontend_context_receipt", "build_rewriter_packet",
    "build_rewriter_prompt",
    "public_frontend_context_digest", "replay_packet_digest",
]
