"""Source-isolated replay prompts for normal and masked-J Gate1 runs."""

from __future__ import annotations

import json
import re
from typing import Any

from .model import Gate1Artifact, ReplayPlan, SchemaError
from .rules import RuleRegistry


_PROMPT_PREAMBLE = """You are the replay agent in the SCRC Gate1 pipeline.
Generate one parameterized kernel for the declared frontend using only the JSON
packet below. You have no source-kernel access. Treat J as the sole authority
for semantics and boundary behavior; treat pi as the complete ordered list of
replay-rule instances for this source implementation. rule_cards are the sole
authority for concrete frontend API spellings. Execute every pi step in order,
read its semantic values from referenced J/B/C/interface fields, and do not
invent hidden default steps.

Hard constraints:
1. Preserve function_name, parameter order/modes/dtypes/ranks, constexpr slots,
   and the declared grid contract.
2. Implement every J node and every declared boundary access/effect/output
   mapping by following the complete pi construction plan and the referenced J
   DAG. pi rule ids select mechanisms but never duplicate J semantic values.
3. Keep every schedule slot symbolic. Never invent or bake in a C value. Only
   interface parameters with kind=constexpr belong in the kernel signature or
   body; schedule kind=compiler and interface.compiler_slots are launch-only
   options and must never be referenced as kernel variables.
4. Every J-local identifier and every pi node_ref/refs entry is an opaque,
   declaration-order handle. Resolve it only through the J/B object that owns
   it or through reference_bindings. When a handle is bound to an interface
   parameter or schedule slot, emit the concrete name stored at that path in
   the generated ABI/body. Never infer semantics from handle spelling.
5. J reduction axes are global tensor axes; frontend reduction calls use the
   matching boundary.reductions.tile_axis for the local tile value.
6. When pi selects reduction_tiling=multi_tile_accumulate, keep one semantic J
   reduction but emit the reviewed runtime tile loop from zero to the J semantic
   extent with C.strategy.tile_slot as its exact step. Initialize the carry once
   from BoundaryReduction.identity in C.strategy.accumulator_dtype, convert each
   local partial to that carry dtype before combining, and finalize mean division
   only after the loop. Keep iteration-dependent tile offsets, final tile
   addresses, masks, loads, local reductions and carry updates inside the loop.
   If and only if pi also selects
   reduction_preheader_staging=stage_invariant_access_scaffolding, materialize
   row/outer-index adjusted base pointers and the shared lane basis before the
   loop and capture them into it; do not rebuild raw_pointer + outer_index *
   stride in the loop. Never move a tile-dependent address, mask or load to the
   preheader, and never collapse the loop for a one-tile specialization.
   If pi additionally selects
   reduction_tile_fusion=shared_producer_outputs, realize the maximal per-tile
   J producer closure shared by the reduction and the selected non-reduced
   outputs exactly once inside each runtime-loop iteration. Keep those outputs'
   computations and declared stores in the same iteration with their exact
   Boundary predicates/effects, and feed the same produced values to the local
   partial reduction. Do not emit an eager copy of the shared loads/producers
   before the loop or a second side-output loop. Only the associative reduction
   carry crosses iterations; realize consumers of the final reduced value after
   the loop.
7. boundary.broadcasts.axes are global semantic axes, never local Triton axis
   numbers. A semantic broadcast may be implicit on a local tile; do not emit an
   explicit rank change unless the selected rule's global-to-local projection
   proves that a materialized consumer-local axis is absent from the operand.
8. Reproduce each declared access predicate exactly; do not strengthen it with
   redundant domain bounds.
9. Apply each pi rule only to its declared target and in listed construction
   order. An empty pi is valid only when the artifact has no replay work.
10. Emit one deterministic syntactic normal form. Artifacts that differ only in
   one J/B literal must produce token-identical declarations, operation order,
   access order, pointer/mask construction and broadcast placement everywhere
   outside that literal's direct semantic use. Never refactor, hoist, sink,
   fold or reorder code based on a literal value.
11. Keep one-dimensional domain coordinates one-dimensional. Before realizing a
    broadcast, establish the ordered operand-local axes, ordered consumer-local
    axes, and global axes fixed/elided by the program domain. Rank-0 operands or
    an empty missing-local-axis sequence are direct uses. Otherwise emit exactly
    one tl.expand_dims per missing local axis at its final consumer-local position,
    in ascending position order; the insertion count must equal the local rank
    delta and the final local shape must match the consumer. Reuse an identical
    projected view instead of expanding per BroadcastSpec/use. If this projection
    is ambiguous, do not guess or use the global axis number as a local index.
    Never pre-expand a coordinate globally.
12. Emit only the validator-supported Triton module subset. Never emit raise,
    assert, try/except/finally, exit calls, or host-side unsupported-input
    guards. The declared interface, grid and semantic domain are the replay
    contract. Realize runtime tails with the declared masks, fills or tl.where;
    never replace missing realization evidence with an aborting Python branch.
13. Return kernel source only, with no Markdown fence or explanatory prose.
"""

_MASKED_NOTE = """
ABLATION RUN: J's semantic and boundary contents were deliberately masked and
all J-local and ABI/C identifiers were consistently alpha-renamed. Preserve the
opaque function/parameter order, grid contract, and slot declarations exactly;
the harness will restore source ABI spellings deterministically before using the
same compile case. Do not reverse or guess masked names. Produce the best replay
possible from the remaining packet.
"""

_PLAN_ABLATION_NOTES = {
    "j_only": """
PI ABLATION RUN (J-only): the complete pi was deliberately removed after the
original artifact passed strict-plan validation. The full frontend rule-card
library is supplied only for this diagnostic. Reconstruct from J/B, interface
and symbolic C slots without source access. Do not assume an omitted source
implementation choice. The interface itself is not ablated: reproduce the
declared function signature exactly, including every constexpr parameter even
when the generated body does not otherwise use it.
""",
    "plan_step_dropout": """
PI ABLATION RUN (step dropout): one validated pi rule instance was deliberately
removed. Execute only the remaining ordered plan. Do not recreate an omitted
construction step from source expectations or identifier spelling. J remains
the semantic authority, but this diagnostic measures whether the removed pi
step had observable replay value.
""",
    "plan_mechanism_substitution": """
PI ABLATION RUN (mechanism substitution): one validated pi rule instance was
replaced by another registered mechanism while the plan remained complete.
Apply the substituted mechanism to its declared target, preserve every other pi
step and J/B fact, and do not refactor unrelated code. This is a source-blind
causal test of pi mechanism control.
""",
}

_OPAQUE_ABLATION_NOTE = """
OPAQUE REFERENCE DISCIPLINE: handles such as t2, n0, a1, p3 and c0 are JSON
references, not Python identifiers. Never emit a handle directly unless the
generated body explicitly defines that local. Resolve tensor handles by finding
the interface parameter whose tensor_ref owns the handle; resolve p/c handles
through reference_bindings and then use the concrete name stored in the target
interface parameter or schedule slot. Preserve every declared interface
parameter, including unused constexpr positions.
"""


def build_rewriter_payload(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    *,
    masked: bool = False,
    plan_override: ReplayPlan | None = None,
    ablation_kind: str | None = None,
) -> dict[str, Any]:
    """Build the only payload that may cross the lift/replay isolation boundary."""

    if not isinstance(artifact, Gate1Artifact):
        artifact = Gate1Artifact.from_json(artifact)
    if not isinstance(registry, RuleRegistry):
        registry = RuleRegistry.from_json(registry)
    # Validate the unmasked artifact first.  The masked packet intentionally
    # replaces J/B identifiers and contents, so coverage cannot be reconstructed
    # from that ablation view alone.
    registry.validate_strict_plan(
        artifact.plan,
        frontend=artifact.interface.frontend,
        artifact=artifact,
    )
    if ablation_kind not in {None, *_PLAN_ABLATION_NOTES}:
        raise ValueError(f"unknown replay ablation kind: {ablation_kind!r}")
    if masked and plan_override is not None:
        raise ValueError("masked-J and pi-plan ablations must run separately")
    if (plan_override is None) != (ablation_kind is None):
        raise ValueError(
            "plan_override and ablation_kind must be supplied together"
        )
    view = artifact.rewriter_view(masked=masked)
    selected_plan = (
        plan_override
        if plan_override is not None
        else ReplayPlan.from_json(view["plan"])
        if masked
        else artifact.plan
    )
    retrieval = registry.rewriter_view(
        selected_plan,
        frontend=artifact.interface.frontend,
        artifact=None if masked or plan_override is not None else artifact,
    )
    if ablation_kind == "j_only":
        retrieval["cards"] = [
            card.to_json()
            for card in registry.cards
            if card.frontend is artifact.interface.frontend
        ]
    if masked:
        view["plan"] = retrieval["plan"]
        view["rule_selection"] = retrieval["selection"]
    else:
        view["plan"] = artifact.opaque_plan_view(selected_plan)
        view["rule_selection"] = artifact.opaque_selection_view(
            retrieval["selection"]
        )
    view["rule_cards"] = retrieval["cards"]
    view["rule_registry_digest"] = registry.digest
    if ablation_kind is not None:
        view["ablation"] = {"kind": ablation_kind}
    assert_source_isolated_payload(view)
    return view


def build_normal_rewriter_view(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
) -> dict[str, Any]:
    """Return the normal source-isolated JSON view without rendering a prompt."""

    return build_rewriter_payload(artifact, registry, masked=False)


def build_masked_rewriter_view(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
) -> dict[str, Any]:
    """Return the opaque masked-J ablation view without rendering a prompt."""

    return build_rewriter_payload(artifact, registry, masked=True)


def assert_source_isolated_payload(payload: dict[str, Any]) -> None:
    """Fail closed if privileged artifact fields enter a replay packet."""

    forbidden_paths: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if key in {"source", "source_text", "location_map", "provenance"}:
                    forbidden_paths.append(child_path)
                if key == "values" and path.endswith("schedule"):
                    forbidden_paths.append(child_path)
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload, "")
    if forbidden_paths:
        raise SchemaError(
            "rewriter payload contains privileged field(s): " + ", ".join(forbidden_paths)
        )

    if payload.get("reference_naming") == "opaque_replay_refs_v1":
        bindings = payload.get("reference_bindings")
        if not isinstance(bindings, list):
            raise SchemaError(
                "opaque replay payload requires reference_bindings array"
            )
        opaque_pattern = re.compile(r"^[tndavpcexqsb]\d+(?:_\d+)?$")
        known: set[str] = set()
        for index, binding in enumerate(bindings):
            if not isinstance(binding, dict) or set(binding) != {"ref", "owners"}:
                raise SchemaError(
                    f"reference_bindings[{index}] must contain ref/owners"
                )
            ref = binding["ref"]
            owners = binding["owners"]
            if (
                not isinstance(ref, str)
                or opaque_pattern.fullmatch(ref) is None
                or ref in known
            ):
                raise SchemaError(
                    f"reference_bindings[{index}].ref is not a unique opaque id"
                )
            if not isinstance(owners, list) or not owners:
                raise SchemaError(
                    f"reference_bindings[{index}].owners must be non-empty"
                )
            for owner in owners:
                if (
                    not isinstance(owner, dict)
                    or set(owner) != {"kind", "path"}
                    or not all(isinstance(value, str) and value for value in owner.values())
                ):
                    raise SchemaError(
                        f"reference_bindings[{index}] has an invalid owner"
                    )
            known.add(ref)

        def check_reference(value: Any, path: str) -> None:
            if not isinstance(value, str) or value not in known:
                raise SchemaError(
                    f"{path} must use a declared opaque replay reference"
                )

        plan = payload.get("plan", {})
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        if not isinstance(steps, list):
            raise SchemaError("opaque replay plan.steps must be an array")
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise SchemaError(f"plan.steps[{index}] must be an object")
            if "node_ref" in step:
                check_reference(step["node_ref"], f"plan.steps[{index}].node_ref")
            for ref_index, ref in enumerate(step.get("refs", [])):
                check_reference(ref, f"plan.steps[{index}].refs[{ref_index}]")

        selection = payload.get("rule_selection", [])
        if not isinstance(selection, list):
            raise SchemaError("opaque rule_selection must be an array")
        for index, item in enumerate(selection):
            if not isinstance(item, dict):
                raise SchemaError(f"rule_selection[{index}] must be an object")
            for key in ("target_ref", "node_ref"):
                if key in item:
                    check_reference(item[key], f"rule_selection[{index}].{key}")
            for ref_index, ref in enumerate(item.get("refs", [])):
                check_reference(
                    ref, f"rule_selection[{index}].refs[{ref_index}]"
                )

        strategy = payload.get("strategy", {})
        strategy_bindings = (
            strategy.get("bindings", []) if isinstance(strategy, dict) else None
        )
        if not isinstance(strategy_bindings, list):
            raise SchemaError("opaque strategy.bindings must be an array")
        for index, binding in enumerate(strategy_bindings):
            if not isinstance(binding, dict):
                raise SchemaError(f"strategy.bindings[{index}] must be an object")
            check_reference(
                binding.get("target_ref"),
                f"strategy.bindings[{index}].target_ref",
            )
            if "tile_slot" in binding:
                check_reference(
                    binding["tile_slot"],
                    f"strategy.bindings[{index}].tile_slot",
                )


def build_rewriter_prompt(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    *,
    masked: bool = False,
    plan_override: ReplayPlan | None = None,
    ablation_kind: str | None = None,
    indent: int | None = 2,
) -> str:
    payload = build_rewriter_payload(
        artifact,
        registry,
        masked=masked,
        plan_override=plan_override,
        ablation_kind=ablation_kind,
    )
    preamble = _PROMPT_PREAMBLE + (_MASKED_NOTE if masked else "")
    if ablation_kind is not None:
        preamble += _PLAN_ABLATION_NOTES[ablation_kind] + _OPAQUE_ABLATION_NOTE
    return preamble + "\nREPLAY_PACKET_JSON:\n" + json.dumps(
        payload,
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )


def build_normal_rewriter_prompt(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    *,
    indent: int | None = 2,
) -> str:
    return build_rewriter_prompt(artifact, registry, masked=False, indent=indent)


def build_masked_rewriter_prompt(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    *,
    indent: int | None = 2,
) -> str:
    return build_rewriter_prompt(artifact, registry, masked=True, indent=indent)


__all__ = [
    "assert_source_isolated_payload", "build_masked_rewriter_prompt",
    "build_masked_rewriter_view", "build_normal_rewriter_prompt",
    "build_normal_rewriter_view", "build_rewriter_payload", "build_rewriter_prompt",
]
