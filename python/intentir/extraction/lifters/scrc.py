"""LLM lift-agent driver for SCRC J/pi/C/interface artifacts.

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
from intentir.verify.gate1.model import (
    AliasSpec,
    Gate1Artifact,
    ParameterKind,
    ProvenanceRecord,
    ProvenanceRole,
)
from intentir.verify.gate1.rules import RuleRegistry, default_triton_rule_registry

from .prompt import SCHEMA as INTENTIR_SCHEMA
from .scrc_schema import scrc_artifact_text_format


if TYPE_CHECKING:
    from intentir.verify.gate1.orchestrate import CandidateRecord, ReliftRequest
    from intentir.verify.gate1.ttir import TTIRFingerprint


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# Binds held-out resume/report identity to the relift feedback contract.  This
# is not a Gate score or acceptance rule; it only prevents replaying cached
# evidence produced by an older outer-loop prompt/schema.
SCRC_RELIFT_PROTOCOL = "normal_no_closure_v1"

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
    artifact: Gate1Artifact
    provenance: SCRCProvenance


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
    body.  The SCRC plan therefore gives the lift agent a compiler-view summary
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
            "applies_to": list(card.applies_to),
            "reads": list(card.reads),
        }
        for card in registry.cards
    ]
    return json.dumps(cards, ensure_ascii=False, indent=2)


def build_scrc_lift_instructions(registry: RuleRegistry) -> str:
    return f"""You are the SCRC lift agent. You may read the source kernel.
Return one JSON object for Gate1Artifact. Do not emit prose or markdown.

J.intent must follow this IntentIR vocabulary:
{INTENTIR_SCHEMA}

For SCRC, J contains A+B only. Emit J.intent.intent_function.layer_c as an
empty object; every concrete Layer-C value belongs only in schedule.values.

In addition, return exactly these top-level fields:
- j: {{kernel_id, intent, node_ids, boundary}}, where node_ids is an object
  mapping opaque ids to JSON paths, for example {{"n0":"body[0]"}}; never an array
- interface: {{kernel_id, frontend, function_name, parameters, grid,
  compiler_slots}}
- schedule: {{slots, values}}; values are the source implementation's C values
- plan: {{steps:[{{rule,node_ref?,refs?}}]}}
- location_map: {{source_name, entries:[{{span:{{start_line,end_line}},
  node_refs,access_refs}}]}}

Boundary is Layer B and may contain only domains, accesses, broadcasts,
aliases, effects, output_mappings and reductions. Every load/store access uses
exactly these field names: access_id, kind, tensor, index_expression, optional
predicate, optional fill_value, optional node_ref. Never emit a field named
"fill". Every tiled reduction records node_ref, tile_axis and at least one of
identity or masked_contribution. Every effects entry uses exactly kind, target
and optional node_ref; access details stay in accesses, not effects.
Reduction tile_axis is the axis of the frontend-local tile value passed to the
source reduction call, never a C schedule slot name such as BLOCK_N and never a
blind copy of the global J reduce axis. For example, a global [M,N] row reduce
has J axes=[1], but when one program holds one 1-D row tile the source call uses
tile_axis=0. boundary.broadcasts records global semantic broadcast axes; it does
not by itself require an explicit frontend expand-dims operation.

Cross-field rules:
- every boundary.domains value is one complete string expression;
- every access predicate copies only the condition actually executed by the
  source. Do not add redundant lower bounds or domain facts already guaranteed
  by a constructed lane range (for tl.arange lanes, copy `col < N`, not
  `0 <= col < N`);
- node_ids contains only IntentIR body-node ids mapped to paths such as
  body[0], never domain/access/effect/output-mapping ids;
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
  offset arithmetic, preserve that as pi mechanism by selecting
  bind_wide_program_domain; otherwise select bind_program_domain. J/B remain
  API-free in both cases;
- in FlagGems sources, an import of
  `flag_gems.utils.triton_lang_extension as tle` makes `tle.program_id(...)` a
  visible helper whose defined result is `tl.program_id(...).to(tl.int64)`;
  select bind_wide_program_domain for that call. Keep the helper/API spelling
  out of J/B and pi free text;
- in the installed FlagGems limits helper, `get_dtype_min` returns negative
  infinity for a floating dtype and `get_dtype_max` returns positive infinity.
  Encode those floating boundary contributions in J/B as the JSON strings
  `-inf` and `+inf`, respectively; do not substitute the largest finite value.
  Keep the source helper spelling out of J/B and pi free text;
- if the source reduction call omits its axis and therefore flattens the local
  tile, keep the global J axes and BoundaryReduction.tile_axis semantic facts
  but select realize_flattened_reduce. An explicit source axis selects
  realize_reduce;
- inspect contraction subscripts before selecting a realization card. A form
  such as m,n->mn has no contracted label and selects realize_outer_product;
  only a contraction with an actual tiled contracted dimension selects
  realize_k_tiled_dot;
- if the source computes one-dimensional mask components and later rank-inserts
  and reuses them for compound access masks, select materialize_mask_tile with
  refs to both component and compound accesses. Do not replace that staging by
  repeated two-dimensional comparisons;
- compiler_slots contains only launch/compiler options absent from the kernel
  signature, never constexpr parameters;
- interface runtime parameters copy the trusted compiler-view argument order,
  tensor/scalar kind, dtype and tensor rank exactly; interface.grid has the same
  rank as the trusted concrete grid and copies grid_contract exactly;
- the complete interface parameter order and constexpr parameter names copy the
  trusted compiler view exactly;
- compiler-view tensor shapes and scalar values describe one specialization;
  keep runtime extents symbolic in J when the source has runtime extent params;
- schedule declares every C key exactly once: source signature constexprs use
  kind constexpr, compiler_slots use kind compiler, and schedule.values copies
  the trusted compiler-view values exactly;
- boundary.aliases copies the trusted compiler-view alias contract exactly;
- plan node_ref/refs may reference only node_ids, J tensor names, domain ids,
  or access_id values. Effects and output mappings have no standalone ids.

pi/plan rules:
- every step selects one registered rule id below;
- node_ref/refs point only to identifiers declared in J;
- never copy source text, expressions or frontend API names into plan;
- semantics live in J, implementation mechanism selection lives in plan.
- select materialize_index_tile when the source computes a complete integer tile
  offset before adding it to a pointer; select materialize_pointer_tile when the
  source keeps pointer + scalar base + lane tile as staged pointer additions.
  In particular, an inline source access such as `load(ptr + row*N + lane)` or
  `store(ptr + row*N + lane, ...)` selects materialize_pointer_tile. Select
  materialize_index_tile only when the source first assigns the complete integer
  expression (for example `offset = row*N + lane`) and later uses `ptr + offset`.
  Inspect the source association; the semantic index_expression string alone
  does not distinguish these two frontend mechanisms.
- for a tiled contraction, precomputed m/n/k coordinate components are not a
  complete per-access integer offset. If the source later forms
  `pointer + row_term + column_term`, select materialize_pointer_tile and retain
  that association; do not regroup it into base-pointer plus a combined offset;
- copy compound vector predicates with the source conjunction operator `&`, not
  scalar-language `and` or `&&`;
- a contraction plan selects realize_k_tiled_dot for the contraction node and
  its K domain/accesses. Do not also select realize_reduce, realize_loop_carry,
  or realize_broadcast for schedule-only K tiling and local rank insertion.
- select exactly one program-domain card for each source coordinate mechanism,
  and exactly one of realize_reduce/realize_flattened_reduce for each source
  reduction mechanism; do not emit both variants for the same operation.
- select exactly one contraction realization that matches the subscripts;
  realize_outer_product and realize_k_tiled_dot are mutually exclusive for one
  J node. Select materialize_mask_tile only when the source actually stages and
  reuses predicate components.

Registered rule ids (API spellings intentionally omitted):
{_rule_catalog(registry)}

Do not emit source or provenance fields; the driver attaches them itself.
"""


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
reports to correct the semantic archive J and the dependent replay plan pi.

RELIFT_EVIDENCE.request records why the outer loop was entered.  The current
authorized trigger is normal_no_closure: bounded source-blind replay repair
still left one or more hard source/replay TTIR differences.  Use the complete
deterministic comparison records to decide whether J omitted/misstated source
semantics or pi selected the wrong realization mechanism.  Scores are ranking
evidence only; every remaining hard difference must be addressed.

You may change exactly these fields:
- j
- plan
- location_map

You must copy interface and schedule byte-for-byte as JSON values from
TRUSTED_IMMUTABLE_FIELDS.  Do not infer, improve, reorder or otherwise edit
them.  The driver rejects any drift.  Replay source is deliberately not
provided; do not invent or discuss a previous replay implementation.  Emit the
complete corrected Gate1Artifact JSON only.
"""
    )
    immutable = {
        "interface": trusted_artifact.interface.to_json(),
        "schedule": trusted_artifact.schedule.to_json(),
    }
    editable = {
        "j": trusted_artifact.j.to_json(),
        "plan": trusted_artifact.plan.to_json(),
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
    """Lift a real source kernel to the full SCRC artifact through the LLM."""

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
            text_format=scrc_artifact_text_format(),
        )
        try:
            payload = _json_object(reply.text)
            if set(payload) == {"gate1_artifact"}:
                payload = payload["gate1_artifact"]
            if not isinstance(payload, dict):
                raise TypeError("gate1_artifact must be an object")
            payload = dict(payload)
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
            # One observed provider spelling uses the plan's prose term
            # ``fill`` for the persisted ``fill_value`` field. Normalize only
            # this enumerated wire alias before the strict artifact parser.
            boundary = payload.get("j", {}).get("boundary", {})
            for access in boundary.get("accesses", []):
                if "fill" in access and "fill_value" not in access:
                    access["fill_value"] = access.pop("fill")
            for effect in boundary.get("effects", []):
                if "target" not in effect and isinstance(effect.get("tensor"), str):
                    effect["target"] = effect["tensor"]
                for extra in (
                    "effect_id",
                    "index_expression",
                    "predicate",
                    "tensor",
                    "value",
                ):
                    effect.pop(extra, None)
            payload["source"] = source
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
            registry.resolve(
                artifact.plan, frontend=artifact.interface.frontend
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
                + "\nRe-emit the complete corrected Gate1Artifact JSON only."
            )

    raise AssertionError("unreachable")


def _candidate_relift_evidence(candidate: "CandidateRecord") -> dict[str, Any]:
    """Return semantic evidence without replay source or compiler source lines."""

    diagnosis = None
    if candidate.diagnosis is not None:
        diagnosis = candidate.diagnosis.to_json()
        if diagnosis.get("cause") == "replay_compile":
            # Compile diagnostics may quote model-generated source lines.  They
            # belong to the inner replay repair loop, not semantic relifting.
            diagnosis.pop("mismatches", None)
    return {
        "relift_round": candidate.relift_round,
        "generation_round": candidate.generation_round,
        "compiled": candidate.compiled,
        "comparison": (
            None
            if candidate.comparison is None
            else candidate.comparison.to_dict()
        ),
        "diagnosis": diagnosis,
        "adopted": candidate.adopted,
        "regressed": candidate.regressed,
    }


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
        history: tuple["CandidateRecord", ...],
        request: "ReliftRequest",
    ) -> Gate1Artifact:
        relift_round = len(self._records) + 1
        request_json = request.to_json()
        if request_json.get("trigger") != "normal_no_closure":
            raise ValueError(
                "unsupported semantic relift trigger: "
                f"{request_json.get('trigger')!r}"
            )
        try:
            if artifact.source is None:
                raise ValueError("semantic relift requires retained source text")
            context = SCRCReliftContext(
                relift_round=relift_round,
                source_fingerprint=source_fingerprint.to_dict(),
                history=tuple(
                    _candidate_relift_evidence(candidate)
                    for candidate in history
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
