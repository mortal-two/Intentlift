"""LLM lift-agent driver for SCRC J/C/interface Lift artifacts.

Unlike the archived extraction validator, this driver does not interpret the
JSON.  It only obtains and parses the lift proposal; Gate1 and Gate2 decide
whether that proposal is faithful to the source kernel.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from intentir.extraction.providers.client import call_responses
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.gate1.composer import materialize_replay_plan
from intentir.verify.gate1.model import (
    AliasSpec,
    Gate1Artifact,
    ParameterKind,
    ProvenanceRecord,
    ProvenanceRole,
)
from intentir.verify.gate1.rules import (
    RuleRegistry,
    default_triton_rule_registry,
)

from .prompt import SCHEMA as INTENTIR_SCHEMA
from .scrc_schema import scrc_lift_core_text_format


if TYPE_CHECKING:
    from intentir.verify.gate1.orchestrate import ReliftEvidence, ReliftRequest
    from intentir.verify.gate1.ttir import TTIRFingerprint


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# Binds held-out resume/report identity to deterministic-pi and causal
# evidence contract.  This is not a Gate score; it prevents cached artifacts or
# terminal reports from silently crossing prompt/schema/intervention protocols.
SCRC_RELIFT_PROTOCOL = (
    "gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11"
)

_DTYPE_ALIASES = {
    "float16": "f16",
    "half": "f16",
    "float32": "f32",
    "float": "f32",
    "float64": "f64",
    "double": "f64",
    "bfloat16": "bf16",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "bool": "i1",
}


def _canonical_dtype(value: str | None) -> str | None:
    if value is None:
        return None
    spelling = value.casefold().removeprefix("torch.")
    return _DTYPE_ALIASES.get(spelling, spelling)


@dataclass(frozen=True)
class SCRCProvenance:
    model: str
    prompt_sha256: str
    attempts: int
    response_id: str | None = None
    repairs: tuple[str, ...] = ()


@dataclass(frozen=True)
class SCRCExtraction:
    """Completed Lift result consumed by Gate1.

    ``artifact`` already contains Lift-owned J, C, canonical pi and compose
    provenance.  Gate1 may recompute pi to verify integrity, but must never be
    responsible for completing a partial Lift result.
    """

    artifact: Gate1Artifact
    provenance: SCRCProvenance

    def __post_init__(self) -> None:
        if self.artifact.compose_provenance is None:
            raise ValueError(
                "SCRCExtraction requires a complete LiftArtifact with "
                "Lift-generated pi provenance"
            )


@dataclass(frozen=True)
class SCRCReliftContext:
    """Source-side evidence available only to the semantic lift agent."""

    relift_round: int
    source_fingerprint: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    request: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.relift_round < 1:
            raise ValueError("relift_round must be positive")
        object.__setattr__(self, "request", dict(self.request))
        # Fail before a provider call if a caller tries to smuggle a non-JSON
        # object or a non-finite value into the auditable relift prompt.
        json.dumps(self.to_json(), allow_nan=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "relift_round": self.relift_round,
            "source_fingerprint": dict(self.source_fingerprint),
            "history": [dict(item) for item in self.history],
            "request": dict(self.request),
        }


def _source_fingerprint_prompt_view(source_fingerprint: Any) -> dict[str, Any]:
    """Keep deterministic semantic coordinates, omit the full edge graph."""

    payload = source_fingerprint.to_dict()
    digest = getattr(source_fingerprint, "digest", None)
    if not isinstance(digest, str) or not digest:
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
    dataflow_edges = payload.get("dataflow_edges")
    mask_predicates = payload.get("mask_predicates")
    return {
        "schema_version": payload.get("schema_version"),
        "fingerprint_digest": digest,
        "interface": payload.get("interface"),
        "op_multiset": payload.get("op_multiset"),
        "dataflow_edge_count": (
            len(dataflow_edges) if isinstance(dataflow_edges, list) else 0
        ),
        "reductions": payload.get("reductions"),
        "memory_effects": payload.get("memory_effects"),
        "mask_predicate_count": (
            len(mask_predicates) if isinstance(mask_predicates, list) else 0
        ),
        "effect_multiset": payload.get("effect_multiset"),
        "dots": payload.get("dots"),
        "loops": payload.get("loops"),
        "schedule_view": payload.get("schedule_view"),
    }


@dataclass(frozen=True)
class SCRCReliftRecord:
    """One bounded outer-loop relift attempt for live evidence reporting."""

    relift_round: int
    extraction: SCRCExtraction | None = None
    rejected_attempts: tuple["SCRCLiftAttempt", ...] = ()
    diagnostic: str | None = None
    request: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SCRCCompilerArgument:
    """One concrete runtime specialization visible only to the lift agent."""

    name: str
    kind: str
    dtype: str
    shape: tuple[int, ...] = ()
    value: bool | int | float | None = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid compiler argument name: {self.name!r}")
        if self.kind not in {"tensor", "scalar"}:
            raise ValueError("compiler argument kind must be tensor or scalar")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("compiler argument dtype must be a non-empty string")
        shape = tuple(int(dim) for dim in self.shape)
        if any(dim < 0 for dim in shape):
            raise ValueError("compiler argument dimensions must be non-negative")
        if self.kind == "tensor":
            if self.value is not None:
                raise ValueError(
                    "tensor compiler argument requires shape and forbids value"
                )
        elif shape or self.value is None:
            raise ValueError("scalar compiler argument requires value only")
        object.__setattr__(self, "shape", shape)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
        }
        if self.kind == "tensor":
            result["shape"] = list(self.shape)
        else:
            result["value"] = self.value
        return result


@dataclass(frozen=True)
class SCRCCompilerView:
    """Trusted source-compilation facts visible to the lift agent.

    Concrete C values commonly live at a launch site rather than in the kernel
    body.  SCRC therefore gives the lift agent a compiler-view summary
    instead of asking it to invent those values from source text.
    """

    schedule_values: Mapping[str, bool | int | float]
    compiler_slots: tuple[str, ...] = ()
    aliases: tuple[AliasSpec, ...] = ()
    arguments: tuple[SCRCCompilerArgument, ...] = ()
    grid: tuple[int, ...] = ()
    grid_contract: tuple[str, ...] = ()
    parameter_order: tuple[str, ...] = ()
    constexpr_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = dict(self.schedule_values)
        for name, value in values.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"invalid schedule name: {name!r}")
            if not isinstance(value, (bool, int, float)):
                raise TypeError(f"schedule value {name!r} must be a scalar")
        slots = tuple(self.compiler_slots)
        if len(set(slots)) != len(slots) or any(
            not isinstance(name, str) or not name.isidentifier()
            for name in slots
        ):
            raise ValueError("compiler_slots must contain unique identifiers")
        aliases = tuple(
            item if isinstance(item, AliasSpec) else AliasSpec.from_json(item)
            for item in self.aliases
        )
        arguments = tuple(
            item
            if isinstance(item, SCRCCompilerArgument)
            else SCRCCompilerArgument(**item)
            for item in self.arguments
        )
        if len({item.name for item in arguments}) != len(arguments):
            raise ValueError("compiler arguments must have unique names")
        grid = tuple(int(dim) for dim in self.grid)
        if grid and (len(grid) > 3 or any(dim <= 0 for dim in grid)):
            raise ValueError("compiler grid must have one to three positive dims")
        grid_contract = tuple(self.grid_contract)
        if any(not isinstance(item, str) or not item for item in grid_contract):
            raise ValueError("grid_contract must contain non-empty strings")
        if grid and grid_contract and len(grid) != len(grid_contract):
            raise ValueError("concrete grid and grid_contract ranks differ")
        parameter_order = tuple(self.parameter_order)
        constexpr_parameters = tuple(self.constexpr_parameters)
        for label, names in (
            ("parameter_order", parameter_order),
            ("constexpr_parameters", constexpr_parameters),
        ):
            if len(set(names)) != len(names) or any(
                not isinstance(name, str) or not name.isidentifier()
                for name in names
            ):
                raise ValueError(f"{label} must contain unique identifiers")
        runtime_names = {item.name for item in arguments}
        constexpr_names = set(constexpr_parameters)
        if runtime_names & constexpr_names:
            raise ValueError("runtime and constexpr parameter names overlap")
        if parameter_order and set(parameter_order) != (
            runtime_names | constexpr_names
        ):
            raise ValueError(
                "parameter_order must contain every runtime and constexpr "
                "parameter exactly once"
            )
        unknown_constexprs = constexpr_names - set(values)
        if unknown_constexprs:
            raise ValueError(
                "constexpr parameters have no schedule value: "
                + ", ".join(sorted(unknown_constexprs))
            )
        object.__setattr__(self, "schedule_values", values)
        object.__setattr__(self, "compiler_slots", slots)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "grid_contract", grid_contract)
        object.__setattr__(self, "parameter_order", parameter_order)
        object.__setattr__(self, "constexpr_parameters", constexpr_parameters)

    def to_json(self) -> dict[str, Any]:
        return {
            "schedule_values": dict(self.schedule_values),
            "compiler_slots": list(self.compiler_slots),
            "aliases": [item.to_json() for item in self.aliases],
            "arguments": [item.to_json() for item in self.arguments],
            "grid": list(self.grid),
            "grid_contract": list(self.grid_contract),
            "parameter_order": list(self.parameter_order),
            "constexpr_parameters": list(self.constexpr_parameters),
        }


@dataclass(frozen=True)
class SCRCLiftAttempt:
    """One provider response and the local reason it was rejected.

    The complete response is retained on the exception object so a live probe
    can inspect schema drift without issuing the same expensive request again.
    It is deliberately omitted from the exception message.
    """

    attempt: int
    model: str
    diagnostic: str
    response_text: str = field(repr=False)
    response_id: str | None = None


class SCRCLiftError(RuntimeError):
    """Raised after the bounded lift/repair budget is exhausted."""

    def __init__(self, attempts: list[SCRCLiftAttempt]) -> None:
        self.attempts = tuple(attempts)
        summary = "; ".join(
            f"attempt {item.attempt}: {item.diagnostic}" for item in attempts
        )
        super().__init__(
            f"SCRC lift failed after {len(attempts)} response(s): {summary}"
        )


def _json_object(text: str) -> dict:
    match = _FENCE.search(text)
    if match:
        text = match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("lift response contains no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("lift response root must be an object")
    return value


def _rule_catalog(registry: RuleRegistry) -> str:
    cards = [
        {
            "id": card.id,
            "selection_mode": card.selection_mode.value,
            "override_family": card.override_family,
            "applies_to": list(card.applies_to),
            "reads": list(card.reads),
        }
        for card in registry.cards
    ]
    return json.dumps(cards, ensure_ascii=False, indent=2)


def build_scrc_lift_instructions(registry: RuleRegistry) -> str:
    return f"""You are the SCRC lift agent. You may read the source kernel.
Return one JSON object for LiftCore. Do not emit prose or markdown.
LiftCore is the source-aware LLM substage of one complete Lift operation.
The LiftCore response schema forbids plan/pi; after J/C validation, the same
Lift operation deterministically materializes pi and provenance and returns a
complete LiftArtifact to Gate1. Gate1 does not generate missing pi.

J.intent must follow this IntentIR vocabulary:
{INTENTIR_SCHEMA}

For SCRC, J contains A+B only. Emit J.intent.intent_function.layer_c as an
empty object; every concrete Layer-C value belongs only in schedule.values.
J.intent Layer A must be invariant under program/tile decomposition. Never put
program ids, block ids, tile bases, lane ids, warp ids, schedule-slot names or
concrete schedule values into J.intent. A source expression such as
`pid * BLOCK + lane` is one global semantic ordinal in J, not three Layer-A
values. Association chosen only to reproduce frontend TTIR belongs to reviewed
C.strategy/RuleCards, not to extra J arithmetic.

In addition, return exactly these top-level fields:
- j: {{kernel_id, intent, node_ids, boundary, scopes, storage}}, where node_ids
  is an object mapping opaque ids to JSON paths, for example
  {{"n0":"body[0]"}}; never an array
- interface: {{kernel_id, frontend, function_name, parameters, grid,
  compiler_slots}}
- schedule: {{slots, values}}; values are the source implementation's C values
- strategy: {{bindings:[{{target_ref,family,choice}}]}}; emit only reviewed
  non-canonical source implementation choices, never semantic facts or steps
- location_map: {{source_name, entries:[{{span:{{start_line,end_line}},
  node_refs,access_refs}}]}}

Boundary is Layer B and may contain only domains, accesses, broadcasts,
aliases, effects, output_mappings and reductions. Every load/store access uses
exactly these field names: access_id, kind, tensor, index_expression, optional
predicate, optional fill_value, optional node_ref. Never emit a field named
"fill". Every tiled reduction records node_ref, tile_axis and at least one of
identity or masked_contribution. Every effects entry uses exactly effect_id,
kind, target, optional node_ref, and access_ref (the matching access_id or null
for a non-access effect); access details stay in accesses, not effects.
Reduction tile_axis is the axis of the frontend-local tile value passed to the
source reduction call, never a C schedule slot name such as BLOCK_N and never a
blind copy of the global J reduce axis. For example, a global [M,N] row reduce
has J axes=[1], but when one program holds one 1-D row tile the source call uses
tile_axis=0. boundary.broadcasts records global semantic broadcast axes; it does
not by itself require an explicit frontend expand-dims operation.

Cross-field rules:
- scopes is a portable semantic scope tree, not a copy of Python/Triton AST.
  Emit exactly one parentless function scope; optional program_instance,
  loop_region and branch_region children use parent_scope. captures,
  definitions and effects contain only J/B opaque references and must match
  actual dominance/visibility. Never put source lines, source variable/helper
  names, frontend API spellings, or formatting into a scope;
- Scope.definitions contains J value handles, never node/domain/access/scope ids.
  For a straight-line function it is exactly the ordered set of J input names
  plus top-level op result names (for add: `[a,b,out]`, not
  `[n0,a0,a1,a2,d0]`). Scope.effects contains effect_id values such as e0/e1/e2.
  A loop scope defines its body_argument, scan names and body op results, while
  captures lists only outer J values actually used in that loop body;
- only a semantic `op="loop"` may carry IntentOp.scope_id, and it must name its
  matching loop_region scope. Ordinary map/reduce/cast/select/contraction ops
  omit scope_id; nesting alone determines their portable scope;
- each StorageObject is one semantic identity and lists every J tensor/value/
  loop-state handle denoting that identity in members. storage records only
  semantic lifetime for destination-passing,
  in-place/aliasing, cross-iteration state, shared storage across outputs, or
  observable effects. Ordinary SSA liveness is derived and must not be repeated.
  register/shared/global placement, physical alloc/dealloc, prefetch, pipeline
  stages and double buffering are C/rule concerns, never storage fields;
- every loop state_id has one storage record in its loop scope with
  crosses_iteration=true and escapes=true. Every destination-passing out value
  and every Boundary alias endpoint has explicit storage. init_event and
  last_use_event are {{kind,ref}} objects; kind is one of
  scope_entry/scope_exit/node/access/state/effect and ref names the matching
  opaque scope/node/access/state/effect id. must_alias endpoints are members of
  the same StorageObject; no_alias endpoints are in different objects;
- a semantic loop uses a non-empty opaque scope_id and closed carry records with
  state_id, init_value, body_argument, next_value, result_value and type
  (dtype+shape). result exactly repeats result_value order and yield exactly
  repeats next_value order. init_value is visible outside the loop,
  body_argument is visible only inside it, and every next_value is defined by
  a body op whose explicit type exactly matches the carry type. trip contains
  only the semantic axis; never put BLOCK/tile/stage
  schedule names or values in J.trip;
- every boundary.domains value is one complete string expression;
- every BoundaryAccess has exactly one matching effects entry with the same
  target and node_ref and the corresponding kind (load=read, store=write,
  atomic=atomic, synchronize=synchronize), linked uniquely by access_ref;
  Scope.effects lists effect_id values exactly, with no missing or unrelated ids;
- every access predicate copies only the condition actually executed by the
  source. Do not add redundant lower bounds or domain facts already guaranteed
  by a constructed lane range (for tl.arange lanes, copy `col < N`, not
  `0 <= col < N`);
- node_ids contains only IntentIR body-node ids mapped to paths such as
  body[0], never domain/access/effect/output-mapping ids;
- all Lift-owned reference ids are declaration-order opaque handles: node_ids
  are exactly n0,n1,...; boundary domain ids are d0,d1,...; access_id values
  are a0,a1,...; effect_id values are e0,e1,...; scope ids are q0,q1,...;
  loop state ids are s0,s1,...; storage ids are b0,b1,... . Never encode
  operation, tensor, load/store,
  reduction, source variable or source-line hints in an id.
  strategy.target_ref uses these opaque handles; only trusted interface
  parameter and schedule-slot names may retain their required ABI spelling;
- a scalar kernel parameter that only supplies a symbolic extent is not an
  IntentIR tensor input;
- a runtime scalar consumed by the computation is an IntentIR scalar input with
  shape=[] and is linked from the scalar interface parameter with tensor_ref;
- canonicalize reciprocal square root in J as sqrt followed by recip. The replay
  may recover the source-native mechanism through the selected rule card;
- when the source spells a mean as local sum divided by the runtime logical
  extent, encode J as reduce fn=mean. The divisor is the semantic extent of the
  global J axis, never a Layer-C tile size;
- re-roll a schedule-only tiled-K accumulator loop into one J contraction. For
  the probe's row-major A[M,K] and B[K,N], use subscripts mk,kn->mn. The local
  K-tile loop and accumulator are frontend realization, not a J loop, scan, or
  reduce node;
- local rank insertion used to form 2-D access tiles (for example lane vectors
  indexed with [:,None] or [None,:]) is not a global semantic broadcast and does
  not create boundary.broadcasts. A contraction also does not create
  boundary.reductions;
- if a source helper explicitly widens a frontend program coordinate before
  offset arithmetic, add program_coordinate_width=wide for that domain;
  otherwise omit the binding. J/B remain API-free in both cases;
- in FlagGems sources, an import of
  `flag_gems.utils.triton_lang_extension as tle` makes `tle.program_id(...)` a
  visible helper whose defined result is `tl.program_id(...).to(tl.int64)`;
  add program_coordinate_width=wide for that call. Keep the helper/API spelling
  out of J/B and strategy;
- in the installed FlagGems limits helper, `get_dtype_min` returns negative
  infinity for a floating dtype and `get_dtype_max` returns positive infinity.
  Encode those floating boundary contributions in J/B as the JSON strings
  `-inf` and `+inf`, respectively; do not substitute the largest finite value.
  Keep the source helper spelling out of J/B and strategy;
- if the source reduction call omits its axis and therefore flattens the local
  tile, keep the global J axes and BoundaryReduction.tile_axis semantic facts
  and add reduction_call_shape=flattened. An explicit source axis uses the
  canonical choice, so omit the binding;
- inspect contraction subscripts and encode them completely in J. A form such
  as m,n->mn has no contracted label and selects realize_outer_product; a
  contraction with an actual tiled contracted dimension selects
  realize_k_tiled_dot. The rule carries only a J node reference, not subscripts;
- if the source computes one-dimensional mask components and later rank-inserts
  and reuses them for compound access masks, add
  mask_materialization=staged_reuse for the affected access. Do not replace
  that staging by repeated two-dimensional comparisons;
- compiler_slots contains only launch/compiler options absent from the kernel
  signature, never constexpr parameters;
- interface runtime parameters copy the trusted compiler-view argument order,
  tensor/scalar kind, dtype and tensor rank exactly; interface.grid has the same
  rank as the trusted concrete grid and copies grid_contract exactly;
- the complete interface parameter order and constexpr parameter names copy the
  trusted compiler view exactly;
- compiler-view tensor shapes and scalar values describe one specialization;
  keep runtime extents symbolic in J when the source has runtime extent params;
- for iota over a runtime logical extent, attrs.n is the exact runtime symbol
  used by the output shape (for example `"size"`), never the compiler-view
  sample such as 1003. The semantic iota is the global ordinal; do not split it
  into quotient/block-base/lane arithmetic using a C tile size;
- schedule declares every C key exactly once: source signature constexprs use
  kind constexpr, compiler_slots use kind compiler, and schedule.values copies
  the trusted compiler-view values exactly;
- boundary.aliases copies the trusted compiler-view alias contract exactly;
- strategy.target_ref may reference only the target kind accepted by its
  family: program_coordinate_width targets a domain id; address_association
  and mask_materialization target an access_id; reduction_call_shape targets a
  reduction node id.

C.strategy rules (closed, override-only vocabulary):
- absence of a target/family binding always means the canonical RuleCard;
- allowed family/choice pairs are program_coordinate_width=wide,
  address_association=pointer_staged, mask_materialization=staged_reuse and
  reduction_call_shape=flattened;
- at most one binding exists for each target/family pair;
- strategy contains no algorithm semantics, expressions, constants, operands,
  predicates, fills, frontend API names, rule ids, construction order or refs
  beyond target_ref. Never emit plan/pi in LiftCore, even as an empty object;
  the enclosing Lift finalizer adds canonical pi to the completed LiftArtifact;
- add address_association=pointer_staged only when the source keeps pointer + scalar
  base + lane tile as staged pointer additions. In particular, an inline source
  access such as `load(ptr + row*N + lane)` or
  `store(ptr + row*N + lane, ...)` uses pointer_staged. When the
  source first assigns the complete integer expression (for example
  `offset = row*N + lane`) and later uses `ptr + offset`, omit the binding for
  that access. Inspect source association; the
  semantic index_expression string alone does not distinguish the mechanisms.
- for a tiled contraction, precomputed m/n/k coordinate components are not a
  complete per-access integer offset. If the source later forms
  `pointer + row_term + column_term`, use pointer_staged and retain that
  association; do not regroup it into base-pointer plus a combined offset;
- copy compound vector predicates with the source conjunction operator `&`, not
  scalar-language `and` or `&&`;
- contraction realization is derived entirely from J subscripts and C; do not
  encode a contraction rule choice in strategy;
- direct predicates omit mask_materialization; staged and reused predicate
  components use staged_reuse.

Do not emit source or provenance fields; the driver attaches them itself.
"""


def _validate_opaque_lift_references(artifact: Gate1Artifact) -> None:
    """Reject source-semantic spellings in Lift-owned J/C reference ids.

    Materialized pi is composer-owned and may deterministically reference a J
    tensor handle; the source-blind writer still receives its opaque rendering.
    """

    def require_declaration_order(
        actual: list[str], prefix: str, label: str
    ) -> None:
        expected = [f"{prefix}{index}" for index in range(len(actual))]
        if actual != expected:
            raise ValueError(
                f"{label} must use declaration-order opaque ids: "
                f"{actual!r} != {expected!r}"
            )

    def walk_ops(op):
        yield op
        for child in op.body:
            yield from walk_ops(child)

    node_ids = list(artifact.j.node_ids)
    domain_ids = list(artifact.j.boundary.domains)
    access_ids = [access.access_id for access in artifact.j.boundary.accesses]
    effect_ids = [
        effect.effect_id for effect in artifact.j.boundary.effects
    ]
    scope_ids = [scope.scope_id for scope in artifact.j.scopes]
    storage_ids = [storage.storage_id for storage in artifact.j.storage]
    state_ids = [
        state.state_id
        for top in artifact.j.intent.body
        for op in walk_ops(top)
        for state in op.carry
    ]
    require_declaration_order(node_ids, "n", "j.node_ids")
    require_declaration_order(domain_ids, "d", "j.boundary.domains")
    require_declaration_order(access_ids, "a", "j.boundary.accesses.access_id")
    require_declaration_order(effect_ids, "e", "j.boundary.effects.effect_id")
    require_declaration_order(scope_ids, "q", "j.scopes.scope_id")
    require_declaration_order(state_ids, "s", "j.intent.loop.carry.state_id")
    require_declaration_order(storage_ids, "b", "j.storage.storage_id")

    owned_groups = (
        node_ids, domain_ids, access_ids, effect_ids, scope_ids, state_ids,
        storage_ids,
    )
    all_owned = [value for group in owned_groups for value in group]
    if len(all_owned) != len(set(all_owned)):
        raise ValueError("Lift-owned opaque ids must be globally collision-free")
    opaque_targets = set(all_owned)
    abi_targets = {
        parameter.name for parameter in artifact.interface.parameters
    } | {slot.name for slot in artifact.schedule.slots}
    tensor_targets = {
        tensor.name
        for tensor in [
            *artifact.j.intent.inputs,
            *artifact.j.intent.outputs,
        ]
    }
    for index, step in enumerate(artifact.plan.steps):
        references = [
            *([step.node_ref] if step.node_ref is not None else []),
            *step.refs,
        ]
        for ref in references:
            if ref in opaque_targets or ref in abi_targets:
                continue
            if ref in tensor_targets:
                continue
            raise ValueError(
                f"plan.steps[{index}] reference {ref!r} is not an opaque "
                "Lift-owned id or a trusted ABI/C name"
            )

    for index, binding in enumerate(artifact.strategy.bindings):
        if (
            binding.target_ref not in opaque_targets
            and binding.target_ref not in abi_targets
        ):
            raise ValueError(
                f"strategy.bindings[{index}].target_ref is not an opaque "
                "Lift-owned id or trusted ABI/C name"
            )


def _scrc_lift_prompt(
    source: str,
    name_hint: str,
    registry: RuleRegistry,
    compiler_view: SCRCCompilerView | None,
) -> tuple[str, str]:
    instructions = build_scrc_lift_instructions(registry)
    user = (
        f'Lift the following kernel. Use kernel_id "{name_hint}" and preserve '
        "its actual callable entry name in interface.function_name."
    )
    if compiler_view is not None:
        user += (
            "\n\nTRUSTED_COMPILER_VIEW (copy these C/interface facts exactly):\n"
            + json.dumps(
                compiler_view.to_json(), ensure_ascii=False, sort_keys=True
            )
        )
    return instructions, user + "\n\nSOURCE:\n" + source


def _scrc_relift_prompt(
    source: str,
    name_hint: str,
    registry: RuleRegistry,
    compiler_view: SCRCCompilerView | None,
    trusted_artifact: Gate1Artifact,
    context: SCRCReliftContext,
) -> tuple[str, str]:
    instructions = (
        build_scrc_lift_instructions(registry)
        + """

This invocation is an outer-loop semantic relift, not a replay-code repair.
Re-read SOURCE and use the source fingerprint plus prior deterministic Gate1
reports to correct the semantic archive J and C.strategy. The deterministic
composer will rebuild pi after this response is accepted.

RELIFT_EVIDENCE.request records why the outer loop was entered.  The current
authorized triggers are replay_generation_failed, replay_compile_failed,
replay_evaluation_failed and normal_no_closure.  No replay candidate is ever
patched in place.  Use the sanitized failure diagnosis or deterministic TTIR
comparison to decide whether J omitted/misstated source semantics or
C.strategy selected the wrong reviewed implementation mechanism. Scores are
ranking evidence only; every remaining hard difference must be addressed.

When RELIFT_EVIDENCE.request.semantic_guidance is non-null, it was generated
deterministically from the first completed Gate1 hard comparison. Treat it as
the explicit worklist for this Relift:
- hard_difference_digest identifies the exact, score-free failure state;
- failed_hard_categories and grouped_differences say what remains different;
- structural_guidance compares compact loop/reduction/memory-effect/mask/dot
  coordinates from the source and replay fingerprints.  A loop mismatch is a
  direct instruction to re-check semantic loop carry/streaming state in J,
  while still distinguishing algorithmic state from lowering-only topology;
- each loop entry is a bounded scoped-SSA witness: block-argument count,
  induction/carried-argument use counts, exact init-to-yield pairs, values
  captured from outside the region, body op multiset and body effects.  Use
  these facts to reconstruct dependencies from SOURCE; do not translate their
  TTIR shape one-for-one into J;
- inspect_j_surfaces names the J semantic contracts to re-check against SOURCE;
- reviewed_strategy_families lists only already registered C.strategy choices
  that may be relevant; absence means do not invent an implementation knob;
- confirmed_hard_categories are preservation constraints, not edit targets;
- current_j_summary is a compact description of the artifact that failed.

RELIFT_EVIDENCE.source_fingerprint and history are bounded prompt views.  The
complete immutable comparator evidence remains in the Gate1 audit report;
semantic_guidance carries the exact digest-bound worklist, so omitted raw edge
lists are not permission to guess or use a fuzzy score.

For a fixed-C specialization family, semantic_guidance.policy is
deterministic_fixed_c_family_hard_diff_guidance_v2. In that form:
- each category entry lists affected_profiles and confirmed_profiles;
  consensus_differences stores each exact difference once when it occurs in
  every affected profile, while profile_specific_differences retains only the
  residual shape-specific groups and exact per-profile counts;
- structural_guidance groups profiles with identical compact source/replay
  loop/reduction/effect structure, so family repetition cannot bury a shared
  semantic signal;
- current_j_summary.fixed_c_family identifies profiles that already closed and
  profiles that failed;
- one J/C/pi artifact and one replay implementation must remain shared by the
  whole family. Do not create profile-specific J, strategy, pi or source;
- already hard-closed profiles are non-regression constraints for the next
  source-aware Lift, not evidence that the failed profiles may be ignored.

This is not an attempt to improve generic first-response Lift fluency. The
only objective is to make the next source-aware Lift decision better grounded
in an observed Gate1 semantic difference. For each failed category, decide
whether SOURCE proves a J semantic gap, an existing reviewed C.strategy gap,
or no change expressible by the current schema/rule library. If no semantic
change is justified, return the editable fields unchanged and allow Gate1's
deterministic stagnation policy to stop; never modify J merely to imitate a
TTIR op count or to obtain another random writer sample.

Apply this decision order before editing:
1. Re-derive the whole-tensor function from SOURCE while ignoring program,
   block, tile and lane decomposition.
2. If CURRENT_EDITABLE_FIELDS.j already computes that function, preserve its
   Layer-A op graph. A TTIR op-count/association difference alone is not proof
   that J is missing semantics.
2a. If structural_guidance shows source loops with carried/yielded values but
    replay has no equivalent loop, determine from SOURCE whether the recurrence
    is required to preserve semantics for arbitrary runtime extents (for
    example streaming state across multiple tiles).  If it is, encode the
    semantic loop/carry/yield in J even when a whole-tensor reduce formula is
    denotationally equivalent.  If it is only schedule tiling, leave J unchanged.
3. Use only an already reviewed C.strategy family when it exactly describes
   the source mechanism. Never invent a new family, choice or RuleCard here.
4. If the existing vocabulary cannot express an implementation association,
   return the semantically correct J unchanged. Gate1 may legitimately end in
   NO_CLOSURE; do not force closure by copying schedule topology into J.

In particular, never copy a schedule value into a J const/iota/shape, and never
rewrite global `iota(n)` as `block = iota / BLOCK`, `base = block * BLOCK`,
`lane = iota - base`, even when distributing later arithmetic would be
algebraically equivalent.

You may change exactly these fields:
- j
- strategy
- location_map

You must copy interface and schedule byte-for-byte as JSON values from
TRUSTED_IMMUTABLE_FIELDS.  Do not infer, improve, reorder or otherwise edit
them.  The driver rejects any drift.  Replay source is deliberately not
provided; do not invent or discuss a previous replay implementation. Emit the
complete corrected LiftArtifact JSON only, without plan/pi.
"""
    )
    immutable = {
        "interface": trusted_artifact.interface.to_json(),
        "schedule": trusted_artifact.schedule.to_json(),
    }
    editable = {
        "j": trusted_artifact.j.to_json(),
        "strategy": trusted_artifact.strategy.to_json(),
        "location_map": trusted_artifact.location_map.to_json(),
    }
    user = (
        f'Relift the kernel with kernel_id "{name_hint}".\n\n'
        "TRUSTED_IMMUTABLE_FIELDS:\n"
        + json.dumps(immutable, ensure_ascii=False, sort_keys=True)
        + "\n\nCURRENT_EDITABLE_FIELDS:\n"
        + json.dumps(editable, ensure_ascii=False, sort_keys=True)
        + "\n\nRELIFT_EVIDENCE:\n"
        + json.dumps(context.to_json(), ensure_ascii=False, sort_keys=True)
    )
    if compiler_view is not None:
        user += (
            "\n\nTRUSTED_COMPILER_VIEW (a second immutable cross-check):\n"
            + json.dumps(
                compiler_view.to_json(), ensure_ascii=False, sort_keys=True
            )
        )
    return instructions, user + "\n\nSOURCE:\n" + source


def scrc_lift_prompt_sha256(
    source: str,
    name_hint: str,
    *,
    registry: RuleRegistry | None = None,
    compiler_view: SCRCCompilerView | None = None,
) -> str:
    """Hash the exact initial Lift prompt used for cache invalidation."""

    active_registry = registry or default_triton_rule_registry()
    instructions, user = _scrc_lift_prompt(
        source, name_hint, active_registry, compiler_view
    )
    return hashlib.sha256(
        (instructions + "\x00" + user).encode("utf-8")
    ).hexdigest()


def scrc_relift_prompt_sha256(
    source: str,
    name_hint: str,
    trusted_artifact: Gate1Artifact,
    context: SCRCReliftContext,
    *,
    registry: RuleRegistry | None = None,
    compiler_view: SCRCCompilerView | None = None,
) -> str:
    """Hash the exact semantic relift prompt used for live evidence."""

    active_registry = registry or default_triton_rule_registry()
    instructions, user = _scrc_relift_prompt(
        source,
        name_hint,
        active_registry,
        compiler_view,
        trusted_artifact,
        context,
    )
    return hashlib.sha256(
        (instructions + "\x00" + user).encode("utf-8")
    ).hexdigest()


def _shape_preserving_iotas(
    value: str,
    producers: Mapping[str, Any],
    *,
    seen: frozenset[str] = frozenset(),
) -> tuple[Any, ...]:
    """Find iota ancestors through rank-preserving elementwise operations."""

    if value in seen:
        return ()
    producer = producers.get(value)
    if producer is None:
        return ()
    if producer.op == "iota":
        return (producer,)
    if producer.op not in {"cast", "map_unary", "map_binary", "select"}:
        return ()
    next_seen = seen | {value}
    return tuple(
        ancestor
        for operand in producer.operands
        for ancestor in _shape_preserving_iotas(
            operand, producers, seen=next_seen
        )
    )


def _validate_scrc_iota_extents(artifact: Gate1Artifact) -> None:
    """Require shape-preserving iota chains to retain semantic extents.

    This closes a real J-schema gap: a runtime output extent such as ``size``
    must not silently become the sampled compiler-view value ``1003``. Physical
    block/lane association remains governed by the Lift/Relift layer-authority
    contract rather than by a pattern-matching schedule verifier.
    """

    intent = artifact.j.intent
    top_level_ops = tuple(intent.body)
    producers = {
        op.result: op
        for op in top_level_ops
        if isinstance(op.result, str)
    }
    output_values = {
        mapping.output: mapping.value
        for mapping in artifact.j.boundary.output_mappings
    }

    for output in intent.outputs:
        if len(output.shape) != 1:
            continue
        declared_extent = output.shape[0]
        if not isinstance(declared_extent, (int, str)) or isinstance(
            declared_extent, bool
        ):
            continue
        value = output_values.get(output.name, output.name)
        for iota in _shape_preserving_iotas(value, producers):
            iota_extent = iota.attrs.get("n")
            if iota_extent != declared_extent:
                raise ValueError(
                    "J.intent iota extent must equal its rank-1 semantic "
                    f"output extent: attrs.n={iota_extent!r}, "
                    f"output {output.name!r} shape={output.shape!r}; keep "
                    "runtime extents symbolic instead of copying the sampled "
                    "compiler-view value"
                )


def extract_scrc_artifact(
    source: str,
    name_hint: str,
    *,
    registry: RuleRegistry | None = None,
    provider: ProviderConfig | None = None,
    compiler_view: SCRCCompilerView | None = None,
    max_repairs: int = 3,
    effort: str | None = None,
    relift_context: SCRCReliftContext | None = None,
    trusted_artifact: Gate1Artifact | None = None,
) -> SCRCExtraction:
    """Run the complete Lift and return a full J/C/pi SCRC artifact.

    The source-aware LLM produces LiftCore (J, interface, C and location).  This
    function then performs Lift's deterministic pi finalization before the
    artifact crosses the Lift/Gate1 boundary.
    """

    registry = registry or default_triton_rule_registry()
    if (relift_context is None) != (trusted_artifact is None):
        raise ValueError(
            "relift_context and trusted_artifact must be supplied together"
        )
    if relift_context is None:
        instructions, user = _scrc_lift_prompt(
            source, name_hint, registry, compiler_view
        )
    else:
        assert trusted_artifact is not None
        instructions, user = _scrc_relift_prompt(
            source,
            name_hint,
            registry,
            compiler_view,
            trusted_artifact,
            relift_context,
        )
    prompt_sha = hashlib.sha256(
        (instructions + "\x00" + user).encode("utf-8")
    ).hexdigest()
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    repairs: list[str] = []
    rejected_attempts: list[SCRCLiftAttempt] = []
    conversation = user

    for attempt in range(max_repairs + 1):
        reply = call_responses(
            instructions,
            conversation,
            provider=provider,
            effort=effort,
            text_format=scrc_lift_core_text_format(),
        )
        try:
            payload = _json_object(reply.text)
            if set(payload) == {"gate1_artifact"}:
                payload = payload["gate1_artifact"]
            if not isinstance(payload, dict):
                raise TypeError("gate1_artifact must be an object")
            payload = dict(payload)
            if "plan" in payload or "compose_provenance" in payload:
                raise ValueError(
                    "LiftCore response must not contain plan/pi or "
                    "compose_provenance; Lift's deterministic pi finalizer "
                    "owns them"
                )
            expected_lift_core_fields = {
                "j", "interface", "schedule", "strategy", "location_map",
            }
            if set(payload) != expected_lift_core_fields:
                raise ValueError(
                    "LiftCore response fields must be exactly "
                    f"{sorted(expected_lift_core_fields)!r}, "
                    f"got {sorted(payload)!r}"
                )
            j_payload = payload.get("j")
            if not isinstance(j_payload, dict):
                raise TypeError("j must be an object")
            if not isinstance(j_payload.get("scopes"), list) or not j_payload["scopes"]:
                raise ValueError(
                    "fresh Lift/relift J requires a non-empty portable scopes tree"
                )
            if not isinstance(j_payload.get("storage"), list):
                raise ValueError("fresh Lift/relift J requires storage array")
            node_ids = payload.get("j", {}).get("node_ids")
            if isinstance(node_ids, list):
                normalized_ids: dict[str, str] = {}
                for index, item in enumerate(node_ids):
                    if not isinstance(item, dict):
                        raise TypeError("node_ids array entries must be objects")
                    node_id = item.get("id", item.get("node_id"))
                    path = item.get("path", item.get("json_path"))
                    if not isinstance(node_id, str) or not isinstance(path, str):
                        raise TypeError(
                            f"node_ids[{index}] requires string id/node_id and path/json_path"
                        )
                    normalized_ids[node_id] = path
                payload["j"]["node_ids"] = normalized_ids
            # One observed provider spelling uses the source prose term
            # ``fill`` for the persisted ``fill_value`` field. Normalize only
            # this enumerated wire alias before the strict artifact parser.
            boundary = payload.get("j", {}).get("boundary", {})
            for access in boundary.get("accesses", []):
                if "fill" in access and "fill_value" not in access:
                    access["fill_value"] = access.pop("fill")
            for effect_index, effect in enumerate(boundary.get("effects", [])):
                if "target" not in effect and isinstance(effect.get("tensor"), str):
                    effect["target"] = effect["tensor"]
                effect.setdefault("effect_id", f"e{effect_index}")
                for extra in (
                    "index_expression",
                    "predicate",
                    "tensor",
                    "value",
                ):
                    effect.pop(extra, None)
            payload["source"] = source
            # Gate1Artifact is the persisted post-compose object.  The Lift
            # wire object deliberately has no plan, so use an empty temporary
            # value only to run the strict J/interface/C parser.
            payload["plan"] = {"steps": []}
            metadata: dict[str, Any] = {
                "prompt_sha256": prompt_sha,
                "response_id": reply.raw.get("id"),
            }
            previous_provenance: list[dict[str, Any]] = []
            if trusted_artifact is not None:
                assert relift_context is not None
                metadata.update(
                    {
                        "kind": "semantic_relift",
                        "relift_round": relift_context.relift_round,
                    }
                )
                previous_provenance = [
                    item.to_json() for item in trusted_artifact.provenance
                ]
            payload["provenance"] = [
                *previous_provenance,
                ProvenanceRecord(
                    ProvenanceRole.LIFTER,
                    "intentir.extraction.lifters.scrc",
                    model=reply.model,
                    source_digest=source_digest,
                    metadata=metadata,
                ).to_json(),
            ]
            artifact = Gate1Artifact.from_json(payload)
            if artifact.j.intent.layer_c:
                raise ValueError(
                    "J.intent.layer_c must be empty; concrete C belongs only "
                    "in schedule.values"
                )
            _validate_scrc_iota_extents(artifact)
            _validate_opaque_lift_references(artifact)
            if trusted_artifact is not None:
                if artifact.interface.to_json() != (
                    trusted_artifact.interface.to_json()
                ):
                    raise ValueError(
                        "semantic relift changed trusted interface"
                    )
                if artifact.schedule.to_json() != (
                    trusted_artifact.schedule.to_json()
                ):
                    raise ValueError(
                        "semantic relift changed trusted schedule C"
                    )
            if compiler_view is not None:
                if compiler_view.parameter_order:
                    actual_order = [
                        item.name for item in artifact.interface.parameters
                    ]
                    if actual_order != list(compiler_view.parameter_order):
                        raise ValueError(
                            "interface parameter order differs from trusted "
                            f"compiler view: {actual_order!r} != "
                            f"{list(compiler_view.parameter_order)!r}"
                        )
                if compiler_view.constexpr_parameters:
                    actual_constexprs = [
                        item.name
                        for item in artifact.interface.parameters
                        if item.kind is ParameterKind.CONSTEXPR
                    ]
                    if actual_constexprs != list(
                        compiler_view.constexpr_parameters
                    ):
                        raise ValueError(
                            "interface constexpr parameters differ from "
                            "trusted compiler view"
                        )
                if compiler_view.arguments:
                    runtime_parameters = [
                        item
                        for item in artifact.interface.parameters
                        if item.kind is not ParameterKind.CONSTEXPR
                    ]
                    expected_names = [
                        item.name for item in compiler_view.arguments
                    ]
                    actual_names = [item.name for item in runtime_parameters]
                    if actual_names != expected_names:
                        raise ValueError(
                            "interface runtime parameter order differs from "
                            f"trusted compiler view: {actual_names!r} != "
                            f"{expected_names!r}"
                        )
                    tensors = {
                        tensor.name: tensor
                        for tensor in [
                            *artifact.j.intent.inputs,
                            *artifact.j.intent.outputs,
                        ]
                    }
                    scalar_extents = {
                        item.name: int(item.value)
                        for item in compiler_view.arguments
                        if item.kind == "scalar"
                        and isinstance(item.value, int)
                        and not isinstance(item.value, bool)
                    }
                    for parameter, argument in zip(
                        runtime_parameters,
                        compiler_view.arguments,
                        strict=True,
                    ):
                        pointer = parameter.kind in {
                            ParameterKind.POINTER,
                            ParameterKind.TENSOR,
                        }
                        if pointer != (argument.kind == "tensor"):
                            raise ValueError(
                                f"interface parameter {parameter.name!r} kind "
                                "differs from trusted compiler view"
                            )
                        if _canonical_dtype(parameter.dtype) != _canonical_dtype(
                            argument.dtype
                        ):
                            raise ValueError(
                                f"interface parameter {parameter.name!r} dtype "
                                "differs from trusted compiler view"
                            )
                        if argument.kind == "tensor":
                            expected_rank = len(argument.shape)
                            if parameter.rank != expected_rank:
                                raise ValueError(
                                    f"interface parameter {parameter.name!r} rank "
                                    f"{parameter.rank!r} != {expected_rank}"
                                )
                            tensor = tensors.get(parameter.tensor_ref or "")
                            if tensor is None or len(tensor.shape) != expected_rank:
                                raise ValueError(
                                    f"interface parameter {parameter.name!r} "
                                    "tensor_ref rank differs from compiler view"
                                )
                            if _canonical_dtype(tensor.dtype) != _canonical_dtype(
                                argument.dtype
                            ):
                                raise ValueError(
                                    f"interface parameter {parameter.name!r} "
                                    "J tensor dtype differs from compiler view"
                                )
                            for declared, concrete in zip(
                                tensor.shape, argument.shape, strict=True
                            ):
                                if isinstance(declared, str):
                                    if declared not in scalar_extents:
                                        raise ValueError(
                                            f"J tensor {tensor.name!r} uses "
                                            f"shape symbol {declared!r} absent "
                                            "from compiler scalar arguments"
                                        )
                                    if scalar_extents[declared] != concrete:
                                        raise ValueError(
                                            f"J tensor {tensor.name!r} shape "
                                            f"symbol {declared!r} binds to "
                                            f"{scalar_extents[declared]}, not "
                                            f"concrete extent {concrete}"
                                        )
                                elif int(declared) != concrete:
                                    raise ValueError(
                                        f"J tensor {tensor.name!r} static extent "
                                        f"{declared} != {concrete}"
                                    )
                        else:
                            if parameter.rank not in (None, 0):
                                raise ValueError(
                                    f"scalar parameter {parameter.name!r} rank must "
                                    "be absent or zero"
                                )
                            if parameter.tensor_ref is not None:
                                tensor = tensors.get(parameter.tensor_ref)
                                input_names = {
                                    item.name for item in artifact.j.intent.inputs
                                }
                                if tensor is None or tensor.name not in input_names:
                                    raise ValueError(
                                        f"scalar parameter {parameter.name!r} "
                                        "tensor_ref must name a J input"
                                    )
                                if tensor.shape:
                                    raise ValueError(
                                        f"scalar parameter {parameter.name!r} "
                                        "tensor_ref must have shape=[]"
                                    )
                                if _canonical_dtype(tensor.dtype) != _canonical_dtype(
                                    argument.dtype
                                ):
                                    raise ValueError(
                                        f"scalar parameter {parameter.name!r} "
                                        "tensor_ref dtype differs from compiler view"
                                    )
                if compiler_view.grid and len(artifact.interface.grid) != len(
                    compiler_view.grid
                ):
                    raise ValueError(
                        "interface.grid rank differs from trusted compiler view"
                    )
                if compiler_view.grid_contract and tuple(
                    artifact.interface.grid
                ) != tuple(compiler_view.grid_contract):
                    raise ValueError(
                        "interface.grid differs from trusted grid contract: "
                        f"{artifact.interface.grid!r} != "
                        f"{list(compiler_view.grid_contract)!r}"
                    )
                if dict(artifact.schedule.values) != dict(
                    compiler_view.schedule_values
                ):
                    raise ValueError(
                        "schedule.values differs from trusted compiler view: "
                        f"{artifact.schedule.values!r} != "
                        f"{dict(compiler_view.schedule_values)!r}"
                    )
                if tuple(artifact.interface.compiler_slots) != tuple(
                    compiler_view.compiler_slots
                ):
                    raise ValueError(
                        "interface.compiler_slots differs from trusted compiler view"
                    )
                if [item.to_json() for item in artifact.j.boundary.aliases] != [
                    item.to_json() for item in compiler_view.aliases
                ]:
                    raise ValueError(
                        "boundary.aliases differs from trusted compiler view"
                    )
            # Pi finalization belongs to Lift.  Gate1 receives only a complete
            # artifact and independently recomputes the expected pi for an
            # integrity check; it never fills in a missing replay plan.
            artifact, _ = materialize_replay_plan(artifact, registry)
            _validate_opaque_lift_references(artifact)
            registry.validate_strict_plan(
                artifact.plan,
                frontend=artifact.interface.frontend,
                artifact=artifact,
            )
            return SCRCExtraction(
                artifact,
                SCRCProvenance(
                    reply.model,
                    prompt_sha,
                    attempt + 1,
                    response_id=reply.raw.get("id"),
                    repairs=tuple(repairs),
                ),
            )
        except Exception as exc:
            diagnostic = f"{type(exc).__name__}: {exc}"
            repairs.append(diagnostic)
            rejected_attempts.append(
                SCRCLiftAttempt(
                    attempt=attempt + 1,
                    model=reply.model,
                    diagnostic=diagnostic,
                    response_text=reply.text,
                    response_id=reply.raw.get("id"),
                )
            )
            if attempt == max_repairs:
                raise SCRCLiftError(rejected_attempts) from exc
            conversation = (
                user
                + "\n\nYour previous artifact could not be parsed:\n"
                + diagnostic
                + "\nRe-emit the complete corrected LiftArtifact JSON only, "
                "without plan/pi."
            )

    raise AssertionError("unreachable")


@dataclass
class ResponsesSCRCRelifter:
    """Real source-aware outer-loop relifter for :class:`Gate1Runner`.

    The rewriter remains source-blind.  This class is the separate lift-agent
    channel described by SCRC section 7: it sees source-side fingerprints and
    deterministic comparison history, but never receives replay source.
    """

    registry: RuleRegistry = field(default_factory=default_triton_rule_registry)
    provider: ProviderConfig | None = None
    compiler_view: SCRCCompilerView | None = None
    max_repairs: int = 2
    effort: str | None = None
    _records: list[SCRCReliftRecord] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")

    @property
    def records(self) -> tuple[SCRCReliftRecord, ...]:
        return tuple(self._records)

    def relift(
        self,
        artifact: Gate1Artifact,
        *,
        source_fingerprint: "TTIRFingerprint",
        history: tuple["ReliftEvidence", ...],
        request: "ReliftRequest",
    ) -> Gate1Artifact:
        relift_round = len(self._records) + 1
        request_json = request.to_json()
        if request_json.get("trigger") not in {
            "replay_generation_failed",
            "replay_compile_failed",
            "replay_evaluation_failed",
            "normal_no_closure",
        }:
            raise ValueError(
                "unsupported semantic relift trigger: "
                f"{request_json.get('trigger')!r}"
            )
        try:
            from intentir.verify.gate1.orchestrate import ReliftEvidence

            if not all(isinstance(item, ReliftEvidence) for item in history):
                raise TypeError(
                    "semantic relift accepts only source-free ReliftEvidence"
                )
            if artifact.source is None:
                raise ValueError("semantic relift requires retained source text")
            context = SCRCReliftContext(
                relift_round=relift_round,
                source_fingerprint=_source_fingerprint_prompt_view(
                    source_fingerprint
                ),
                history=tuple(
                    observation.to_prompt_json()
                    for observation in history
                ),
                request=request_json,
            )
            extraction = extract_scrc_artifact(
                artifact.source,
                artifact.j.kernel_id,
                registry=self.registry,
                provider=self.provider,
                compiler_view=self.compiler_view,
                max_repairs=self.max_repairs,
                effort=self.effort,
                relift_context=context,
                trusted_artifact=artifact,
            )
        except SCRCLiftError as exc:
            self._records.append(
                SCRCReliftRecord(
                    relift_round,
                    rejected_attempts=exc.attempts,
                    diagnostic=str(exc),
                    request=request_json,
                )
            )
            raise
        except Exception as exc:
            self._records.append(
                SCRCReliftRecord(
                    relift_round,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    request=request_json,
                )
            )
            raise
        self._records.append(
            SCRCReliftRecord(
                relift_round,
                extraction=extraction,
                request=request_json,
            )
        )
        return extraction.artifact


__all__ = [
    "ResponsesSCRCRelifter",
    "SCRC_RELIFT_PROTOCOL",
    "SCRCCompilerArgument",
    "SCRCCompilerView",
    "SCRCExtraction",
    "SCRCLiftAttempt",
    "SCRCLiftError",
    "SCRCProvenance",
    "SCRCReliftContext",
    "SCRCReliftRecord",
    "build_scrc_lift_instructions",
    "extract_scrc_artifact",
    "scrc_lift_prompt_sha256",
    "scrc_relift_prompt_sha256",
]
