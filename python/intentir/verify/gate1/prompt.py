"""Source-isolated Replay packets for the LLM-direct Gate1 protocol."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from .model import Frontend, Gate1Artifact, SchemaError
from .pi import (
    render_realization_mechanism_grammar,
    render_realization_witness_wire_grammar,
)


REPLAY_PACKET_SCHEMA = "intentir.scrc.replay-input.v2"
LEGACY_REPLAY_PACKET_SCHEMAS = frozenset({"intentir.scrc.replay-input.v1"})
PUBLIC_FRONTEND_CONTEXT = {
    "triton": {
        "retrieval_scope": "public Triton documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry_contract": (
            "Define exactly one @triton.jit scrc_k0 with J.signature handles "
            "followed only by anonymous MaskC(C) slots whose kind is constexpr. "
            "Keep each such slot annotated tl.constexpr. A MaskC slot whose kind "
            "is compiler is an external compiler-invocation option: never add it "
            "to the callable signature, read it in the kernel body, or turn it "
            "into a constexpr carrier. Triton compile-time-only APIs do not "
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
            "Each J.boundary.broadcasts item is an explicit consumer-local rank "
            "projection. Preserve the operand's closed J type at its producer and "
            "boundary access, then materialize exactly the declared axes at the "
            "named consumer with public explicit rank-lift operations. Never "
            "pre-expand an access index or load shape, and never rely on implicit "
            "broadcasting that erases the authored consumer stage. "
            "Every J.signature p* is one already-existing callable ABI carrier. "
            "All pi instances anchored to the same p* scope or route uses of that "
            "single carrier; they never create another external value or program "
            "coordinate. Duplicate p* instances therefore never authorize duplicate "
            "carrier reads or tl.program_id calls. "
            "Region placement='program' alone never authorizes "
            "tl.program_id. A domain-anchored realize stage in a program region "
            "whose scalar dependency path enters address_base is a program "
            "coordinate. Read its exact physical axis from pi.program_axes: each "
            "domain_ref binds one distinct J domain to axis 0, 1, or 2. Distinct "
            "stages anchored to the same d* remain distinct materializations of "
            "that same bound axis. Never infer or renumber the axis from declaration "
            "order, grid extent, C, or instance multiplicity. A domain/C stage whose path enters address_lane "
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
            " A pi realize instance anchored to an explicit J transpose is a distinct "
            "compiler-visible stage: materialize it with the frontend transpose "
            "operation (tl.trans in Triton). Never fold that authored stage into a "
            "load's lane layout or silently absorb it into a contraction."
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
            " For induction_basis=rounded_extent_step, use the same zero lower "
            "and matching C tile step as extent_step, but use exactly "
            "ceil_div(the declared J trip extent, that C tile) multiplied by "
            "that C tile as the upper bound. Do not use the current tensor shape, "
            "a launch value, or an invented rounding formula."
            " Every pi.physical_iterations entry is mandatory realization "
            "topology for a Source-visible physical loop around an existing "
            "non-loop J reduce/contraction or boundary access/domain; it never "
            "creates a semantic J loop. Resolve body_instance to that authored "
            "subject and tile_instance to the required positive C slot attached "
            "to the same closed dependent axis. The "
            "axis attachment determines the logical extent, and induction_basis "
            "has the same extent_step, rounded_extent_step, or tile_ordinal "
            "denotation as pi.iterations. direction=forward visits that closed "
            "basis in ascending order; direction=reverse visits exactly the same "
            "basis points in descending order. Never infer the extent or direction "
            "from Source names, current shapes, C values, grid conventions, or "
            "frontend APIs. Every pi.physical_loop_carriers entry preserves the "
            "declared init/state/next/result temporal stages for its physical "
            "iteration region; do not collapse the loop-carried accumulator into "
            "one stateless reduction or invent an update formula absent from J. "
            "Within each region, physical_loop_carriers declaration order is the "
            "exact common carrier-tuple ordinal across init/state/next/result; "
            "preserve it in frontend loop arguments, yields, and results, and never "
            "sort by dtype, semantic ref, or mechanism. This tuple order is not "
            "event execution order; pi.order remains its only authority."
            " Every J effect_if is mandatory effect control. Emit one real "
            "runtime `if` with explicit positive and negative arms, preserve "
            "predicate polarity, lexical arm scope, nested operation/effect order, "
            "and an explicitly empty arm. Never replace effect_if with tl.where, "
            "a value select, predicated unconditional effects, or compile-time "
            "control. Read the predicate and both arms only from J."
            " When pi.terminal_controls relates the two authored arm instances of "
            "that effect_if, emit the terminal arm as the sole body of a runtime "
            "`if` followed by a bare return, then emit the continuation arm after "
            "the `if` without an `else`. Use the authored predicate unchanged when "
            "the positive arm is terminal and its logical negation when the negative "
            "arm is terminal. Resolve both roles only through the typed pi relation; "
            "never infer termination from empty arms, effects, values, names, or "
            "frontend conventions."
            " Every J value_if is mandatory pure value control. Emit one real "
            "runtime `if` with explicit positive and negative arms in authored "
            "polarity and lexical order. Each arm may compute pure local "
            "temporaries, but its final statement must assign the authored yield "
            "to the same merge local in both arms. Never replace value_if with "
            "tl.where/select, effect_if, predicated unconditional computation, or "
            "compile-time control. effect_if, value_if, and select are "
            "hard-distinct; read the predicate, arms, yields, and result only "
            "from J."
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
            " Every pi.sharing relation preserves one compiler-visible source "
            "materialization for all listed consumers. Bind that source once and "
            "reuse the same local value, predicate, address, or load result at each "
            "consumer; do not clone an algebraically equivalent expression per use."
            " A reduction_linearization refinement is the exact source-neutral "
            "full-axis materialization immediately before its uniquely routed J "
            "reduce. For Triton, emit `tl.reshape(value, (value.numel,), "
            "can_reorder=True)` at that stage, then reduce on local axis 0. Never "
            "infer it from reduction kind or shape, apply it to another consumer, "
            "or omit it because the current carrier is already rank one."
            " A J const with value '-inf', 'inf', or 'nan' is the exact typed IEEE "
            "special authored by J. Materialize it directly at its declared dtype; "
            "never approximate it with a finite literal or promote it through an "
            "undeclared wider floating type."
            " A J.signature constexpr p* with logical_ref=x* is the callable "
            "carrier of that exact semantic dimension. Use that same p* directly "
            "where the authored pi lane/iteration realization requires x*; never "
            "replace it with a current-shape literal or an arbitrary unit extent."
            " A J.signature constexpr p* with logical_ref=null is only a "
            "Harness-fixed JIT-specialization carrier: preserve its constexpr "
            "ABI, emit the already-authored active graph, and never read it, "
            "reconstruct inactive control, or invent a dummy use. Its concrete "
            "value is absent from the Replay packet."
            " A scalar J.signature p* with logical_ref=null has no logical denotation. It "
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
    "pi": frozenset(
        {
            "regions",
            "instances",
            "routes",
            "program_axes",
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
    ),
}
_HISTORICAL_J_KEY_SETS = frozenset(
    {
        _REPLAY_COMPONENT_KEYS["j"] - {"dimension_relations"},
        _REPLAY_COMPONENT_KEYS["j"]
        - {"dimension_relations", "value_dimension_relations"},
    }
)
_HISTORICAL_PI_KEY_SETS = frozenset(
    {_REPLAY_COMPONENT_KEYS["pi"] - {"program_axes"}}
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
    if set(payload) != {"j", "c", "pi", "u_f"}:
        raise SchemaError(
            "Replay payload root must contain exactly j, c, pi, and u_f"
        )
    for component, expected in _REPLAY_COMPONENT_KEYS.items():
        value = payload[component]
        actual = set(value) if isinstance(value, Mapping) else set()
        historical_j = (
            not require_active_context
            and component == "j"
            and frozenset(actual) in _HISTORICAL_J_KEY_SETS
        )
        historical_pi = (
            component == "pi"
            and frozenset(actual) in _HISTORICAL_PI_KEY_SETS
        )
        if not isinstance(value, Mapping) or (
            actual != expected and not historical_j and not historical_pi
        ):
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
            if "source" in value and not (
                path.startswith("$.pi.routes[")
                or path.startswith("$.pi.route_extent_transforms[")
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
        "declares multiple contraction stages. Distinct ordered transfer_direct "
        "instances remain distinct compiler-visible transfers even when they read "
        "the same address. If Triton would CSE such transfers, use a public closed "
        "eviction_policy spelling to retain them; the observer normalizes the exact "
        "closed eviction spelling, so it carries no Source information. Never add "
        "a transfer absent from J/pi or collapse one that is present. "
        if frontend is Frontend.TRITON
        else ""
    )
    return (
        "You are the source-blind Replay LLM. Generate one complete, self-contained "
        f"{frontend.value} public-frontend kernel module. You never see "
        "the source implementation. Consume J, pure C and pi directly; pi is topology, "
        "not semantic authority. Treat every pi instance, route, order edge, sharing "
        "relation, route extent transform, structured or physical iteration "
        "relation, physical carrier, "
        "representation refinement and region placement as required "
        "realization topology. Region placement is exact lexical scope: emit each "
        "compiler-visible instance in its declared region, keep a loop-region "
        "instance lexically inside that loop even when it is induction-invariant, "
        "and never hoist, sink, clone, or fold it across a region boundary. In "
        "every pi order item, the instance named by "
        "before must execute before the instance named by after, including across "
        "regions and observable effects. Resolve each refinement auth_ref only through the "
        "matching J realization_authorization, materialize it at the refined "
        "instance before routed consumers, and never infer, omit, move, or copy a "
        "sensitive conversion onto a sibling instance. A logical_ref=null p* may "
        "be read only when pi contains its anchored instance and a route from that "
        "instance to a realization consumer; otherwise it remains an unused ABI "
        "parameter. Multiple "
        "instances anchored to one handle are distinct source stages: do not collapse "
        "their chain, discard J/C dependencies, duplicate a shared producer, "
        "or invent an undocumented stage. Obtain callable logical handles from "
        "J.signature and append only kind=constexpr anonymous slots from MaskC(C). "
        "Keep kind=compiler slots out of the callable signature and body; the "
        "external compiler invocation consumes them. Obtain all "
        "logical operators, predicates, fills, constants, index relations and "
        "effects only from J. Preserve explicit J stages: reduce(sum) followed by "
        "division remains sum then division, never mean or a generated sum-of-ones "
        "count. Every J.boundary.broadcasts item applies only at its named consumer: "
        "keep the operand's closed type at its producer/access and materialize the "
        "declared axes with explicit public rank-lift operations before that consumer. "
        "Do not pre-expand the load/index shape or erase the stage through implicit "
        "broadcasting. Treat every J.value_dimension_relations item as an already "
        "J-certified precondition: relation=eq permits the declared integral "
        "rank-0 value_ref and dimension_ref to denote the same logical extent. "
        "Consume only the declared relation where J shapes, domains, predicates, "
        "or operators require it. Never infer, create, delete, repair, weaken, or "
        "strengthen a value/dimension relation from concrete values, parameter "
        "positions, C, pi, Harness conventions, or frontend behavior. "
        "Treat every J.dimension_relations item as an authored logical-dimension "
        "definition. Implement exactly identity/add_const/mul_const/ceil_div_const/"
        "round_up_const over its named x* handles wherever the graph, indices, "
        "predicates, or realization topology consume the derived extent. Never "
        "infer a relation from concrete shapes, C, pi, grid, or frontend rounding, "
        "and never replace an authored relation with the current valuation. A "
        "boundary access with no predicate is unconditional: "
        "never derive a redundant mask from domains, grid extents, C, or defensive "
        "safety. "
        "Treat each J effect_if as effect control, not a value expression. Ordinarily "
        "emit one real runtime if with explicit positive and negative arms in the "
        "authored polarity, keep every arm-local operation/effect in authored "
        "order, and spell an authored empty arm explicitly. If and only if one "
        "pi.terminal_controls item names the two realize instances anchored to that "
        "effect_if's sibling arm scopes, emit the named terminal arm as one runtime "
        "if ending in a bare return and emit the named continuation arm immediately "
        "after it without an else; negate the condition exactly when the negative "
        "arm is terminal. The relation supplies only arm topology: never guess, copy, "
        "or change the J predicate, polarity, arm operations, or effects. Never lower effect_if "
        "to tl.where/select or execute either arm unconditionally under a mask. "
        "Treat each J value_if as pure value control, distinct from effect_if and "
        "select: emit one real runtime if with explicit polarity-preserving arms; "
        "each arm may use pure local temporaries but must finally assign its J "
        "yield to the same merge local. Never lower value_if to tl.where/select, "
        "effect_if, predicated unconditional computation, or compile-time control. "
        "For every pi.iterations rounded_extent_step relation, resolve its "
        "loop_instance to the J loop trip extent and tile_instance to the matching "
        "C slot; use lower 0, step equal to that C tile, and upper exactly "
        "ceil_div(trip extent, C tile) * C tile. Do not infer a bound from current "
        "shapes/values or invent another rounding formula. "
        "For every pi.route_extent_transforms item, resolve source to its anonymous "
        "required positive-integer C slot and apply the declared floor/ceil division "
        "by the literal divisor only at the named existing route target slot. This "
        "is a representation extent partition: do not turn it into J shape/domain/"
        "predicate semantics, apply it to another C use, or infer another transform. "
        "For every pi.physical_iterations relation, keep J's operation or boundary "
        "traversal logical while materializing the explicit physical loop from its "
        "same-subject C-attached axis, induction_basis, and direction. Preserve every "
        "pi.physical_loop_carriers init/state/next/result temporal stage; neither "
        "relation authorizes a new semantic operator or formula. Preserve same-region "
        "physical_loop_carriers declaration order as the shared carrier-tuple ordinal "
        "across init/state/next/result; do not sort it or treat it as event order. "
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


__all__ = [
    "PUBLIC_FRONTEND_CONTEXT", "REPLAY_PACKET_SCHEMA",
    "ReplayPacket", "assert_source_isolated_payload",
    "build_masked_rewriter_prompt",
    "build_masked_rewriter_view", "build_normal_rewriter_prompt",
    "build_normal_rewriter_view", "build_rewriter_payload",
    "build_rewriter_packet", "build_rewriter_prompt",
]
