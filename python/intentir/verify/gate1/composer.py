"""Deterministic production composer for strict Gate1 replay plans.

It derives a :class:`ReplayPlan` exclusively from the validated J,
interface, schedule, strategy, and registry views of a ``Gate1Artifact``.  The
source program, location map, any pre-existing plan, and kernel name are never
consulted while selecting or ordering rules.  Production materialization is a
Lift-internal finalization step; Gate1 reuses the same pure function only to
verify the complete LiftArtifact's pi integrity.

RuleCard executable metadata drives unique selection and construction order.
Unknown work items and zero/multiple matches fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .model import (
    AccessKind,
    Gate1Artifact,
    ParameterKind,
    ReplayPlan,
    ReplayStep,
    SchemaError,
)
from .rules import RuleCard, RuleRegistry, RuleSelectionMode


COMPOSER_VERSION = "deterministic_composer_v3"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _json_leaves(value: Any, pointer: str = "") -> set[str]:
    if isinstance(value, dict):
        result: set[str] = set()
        for key, child in value.items():
            result.update(
                _json_leaves(child, f"{pointer}/{_json_pointer_token(key)}")
            )
        return result
    if isinstance(value, list):
        result = set()
        for index, child in enumerate(value):
            result.update(_json_leaves(child, f"{pointer}/{index}"))
        return result
    return {pointer or "/"}


class ComposeError(SchemaError):
    """Base class for fail-closed deterministic composition errors."""

    code = "compose_error"

    def __init__(
        self,
        message: str,
        *,
        work_item: str | None = None,
        candidates: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.work_item = work_item
        self.candidates = tuple(candidates)


class ZeroRuleMatchError(ComposeError):
    code = "zero_rule_match"


class AmbiguousRuleMatchError(ComposeError):
    code = "ambiguous_rule_match"


class ComposeDependencyError(ComposeError):
    code = "compose_dependency_error"


class ComposeMismatchError(ComposeError):
    code = "compose_mismatch"

    def __init__(self, expected_digest: str, actual_digest: str) -> None:
        super().__init__(
            "materialized replay plan does not equal deterministic composition "
            f"(expected {expected_digest}, got {actual_digest})"
        )
        self.expected_digest = expected_digest
        self.actual_digest = actual_digest


@dataclass(frozen=True)
class ProvenanceSource:
    """One legal authority for a materialized pi JSON leaf."""

    owner: str
    pointer: str

    def to_json(self) -> dict[str, str]:
        return {"owner": self.owner, "pointer": self.pointer}


@dataclass(frozen=True)
class CanonicalReferenceBinding:
    """Shadow canonical name for one materialized artifact reference."""

    canonical_ref: str
    materialized_ref: str
    kind: str
    source_pointer: str

    def to_json(self) -> dict[str, str]:
        return {
            "canonical_ref": self.canonical_ref,
            "materialized_ref": self.materialized_ref,
            "kind": self.kind,
            "source_pointer": self.source_pointer,
        }


@dataclass
class ComposeProvenance:
    """Whole-input identities and complete per-leaf derivation evidence."""

    composer_version: str
    j_digest: str
    interface_digest: str
    schedule_digest: str
    strategy_digest: str
    registry_digest: str
    plan_digest: str
    leaf_sources: dict[str, tuple[ProvenanceSource, ...]] = field(
        default_factory=dict
    )
    reference_bindings: tuple[CanonicalReferenceBinding, ...] = ()

    def validate(self, plan: ReplayPlan) -> None:
        leaves = _json_leaves(plan.to_json())
        covered = set(self.leaf_sources)
        missing = sorted(leaves - covered)
        orphan = sorted(covered - leaves)
        empty = sorted(
            pointer for pointer, sources in self.leaf_sources.items() if not sources
        )
        if missing or orphan or empty:
            details: list[str] = []
            if missing:
                details.append(f"missing={missing!r}")
            if orphan:
                details.append(f"orphan={orphan!r}")
            if empty:
                details.append(f"empty={empty!r}")
            raise ComposeDependencyError(
                "invalid replay-plan leaf provenance: " + ", ".join(details)
            )
        actual_digest = _digest(
            _canonical_plan_json(plan, self.reference_bindings)
        )
        if actual_digest != self.plan_digest:
            raise ComposeDependencyError(
                "replay-plan provenance digest does not match the materialized plan"
            )

    @property
    def coverage(self) -> float:
        return 1.0 if self.leaf_sources else 0.0

    @property
    def orphan_leaf_count(self) -> int:
        return 0

    def to_json(self) -> dict[str, Any]:
        return {
            "composer_version": self.composer_version,
            "j_digest": self.j_digest,
            "interface_digest": self.interface_digest,
            "schedule_digest": self.schedule_digest,
            "strategy_digest": self.strategy_digest,
            "registry_digest": self.registry_digest,
            "plan_digest": self.plan_digest,
            "leaf_sources": {
                pointer: [source.to_json() for source in sources]
                for pointer, sources in sorted(self.leaf_sources.items())
            },
            "reference_bindings": [
                binding.to_json() for binding in self.reference_bindings
            ],
        }


@dataclass(frozen=True)
class ComposeResult:
    plan: ReplayPlan
    provenance: ComposeProvenance

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(
            _canonical_plan_json(
                self.plan, self.provenance.reference_bindings
            )
        )

    @property
    def plan_digest(self) -> str:
        return self.provenance.plan_digest

    def to_json(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_json(),
            "provenance": self.provenance.to_json(),
        }


@dataclass(frozen=True)
class _StrategyBindingView:
    index: int
    target_ref: str
    family: str
    choice: str

    @property
    def pointer(self) -> str:
        return f"/strategy/bindings/{self.index}"


@dataclass
class _WorkItem:
    key: str
    kind: str
    role: str
    target_ref: str
    target_pointer: str
    facts: dict[str, Any]
    semantic_key: str
    aliases: tuple[str, ...] = ()
    source_pointers: tuple[str, ...] = ()
    dependencies: set[str] = field(default_factory=set)
    selected_card: RuleCard | None = None
    strategy_binding: _StrategyBindingView | None = None


# Closed compatibility metadata for the reviewed pre-executable-metadata cards.
_LEGACY_TARGET_KINDS: dict[str, tuple[str, ...]] = {
    "bind_program_domain": ("boundary.domain",),
    "bind_wide_program_domain": ("boundary.domain",),
    "materialize_index_tile": ("boundary.access.address",),
    "materialize_pointer_tile": ("boundary.access.address",),
    "realize_masked_access": ("boundary.access.effect",),
    "realize_unmasked_access": ("boundary.access.effect",),
    "materialize_mask_tile": ("boundary.access.effect",),
    "realize_elementwise": ("intent_op:map_unary", "intent_op:map_binary"),
    "realize_const": ("intent_op:const",),
    "realize_iota": ("intent_op:iota",),
    "realize_reduce": ("intent_op:reduce",),
    "realize_flattened_reduce": ("intent_op:reduce",),
    "realize_broadcast": ("boundary.broadcast",),
    "realize_k_tiled_dot": ("intent_op:contraction",),
    "realize_outer_product": ("intent_op:contraction",),
    "realize_indexed_access": ("intent_op:gather",),
    "realize_transpose": ("intent_op:transpose",),
    "bind_runtime_scalar": ("interface.runtime_scalar",),
    "declare_schedule_slot": ("schedule.slot",),
    "realize_cast": ("intent_op:cast",),
    "realize_select": ("intent_op:select",),
    "realize_loop_carry": ("intent_op:loop", "intent_op:scan"),
    "realize_output_mapping": ("boundary.output_mapping",),
}

_LEGACY_STRATEGY: dict[str, tuple[str, str]] = {
    "bind_wide_program_domain": ("program_coordinate_width", "wide"),
    "materialize_pointer_tile": ("address_association", "pointer_staged"),
    "materialize_mask_tile": ("mask_materialization", "staged_reuse"),
    "realize_flattened_reduce": ("reduction_call_shape", "flattened"),
}

_SUPPLEMENTAL_STRATEGY_FAMILIES = frozenset({"mask_materialization"})

_LEGACY_PHASE: dict[str, int] = {
    "bind_program_domain": 10,
    "bind_wide_program_domain": 10,
    "bind_runtime_scalar": 20,
    "declare_schedule_slot": 30,
    "materialize_index_tile": 40,
    "materialize_pointer_tile": 40,
    "materialize_mask_tile": 45,
    "realize_masked_access": 50,
    "realize_unmasked_access": 50,
    "realize_broadcast": 55,
    "realize_elementwise": 60,
    "realize_const": 60,
    "realize_iota": 60,
    "realize_reduce": 60,
    "realize_flattened_reduce": 60,
    "realize_k_tiled_dot": 60,
    "realize_outer_product": 60,
    "realize_indexed_access": 60,
    "realize_transpose": 60,
    "realize_cast": 60,
    "realize_select": 60,
    "realize_loop_carry": 60,
    "realize_output_mapping": 80,
}


def _selection_mode(card: RuleCard) -> str:
    value = getattr(card, "selection_mode", RuleSelectionMode.DERIVED)
    return getattr(value, "value", value)


def _card_strategy(card: RuleCard) -> tuple[str | None, str | None]:
    family = getattr(card, "strategy_family", None)
    choice = getattr(card, "strategy_choice", None)
    if family is None:
        legacy = _LEGACY_STRATEGY.get(card.id)
        if legacy is not None:
            family, legacy_choice = legacy
            if choice is None:
                choice = legacy_choice
        elif _selection_mode(card) == RuleSelectionMode.OVERRIDE.value:
            family = getattr(card, "override_family", None)
    return family, choice


def _card_target_kinds(card: RuleCard) -> tuple[str, ...]:
    explicit = tuple(getattr(card, "target_kinds", ()) or ())
    if explicit:
        return explicit
    known = _LEGACY_TARGET_KINDS.get(card.id)
    if known is not None:
        return known

    # Compatibility for test/private cards authored against the old schema.
    mapped: list[str] = []
    for tag in getattr(card, "applies_to", ()):
        if tag.startswith("intent_op:"):
            mapped.append(tag)
        elif tag in {
            "map_unary", "map_binary", "const", "reduce", "contraction",
            "cast", "select", "loop", "scan", "gather", "iota",
        }:
            mapped.append(f"intent_op:{tag}")
        elif tag in {"boundary.domains", "interface.grid"}:
            mapped.append("boundary.domain")
        elif tag == "boundary.accesses":
            mapped.extend(("boundary.access.address", "boundary.access.effect"))
        elif tag == "boundary.broadcasts":
            mapped.append("boundary.broadcast")
        elif tag == "boundary.output_mappings":
            mapped.append("boundary.output_mapping")
        elif tag in {"interface.parameters", "intent.inputs"}:
            mapped.append("interface.runtime_scalar")
        elif tag == "schedule.slots":
            mapped.append("schedule.slot")
    return tuple(dict.fromkeys(mapped))


def _lookup_field(value: Any, path: str) -> tuple[bool, Any]:
    current = value
    for component in path.split("."):
        if isinstance(current, Mapping) and component in current:
            current = current[component]
        else:
            return False, None
    return True, current


def _required_fields_match(card: RuleCard, item: _WorkItem) -> bool:
    fact_views: tuple[Mapping[str, Any], ...] = (item.facts,)
    grouped_broadcasts = item.facts.get("broadcasts")
    if item.kind == "boundary.broadcast" and isinstance(
        grouped_broadcasts, list
    ):
        if not grouped_broadcasts or not all(
            isinstance(value, Mapping) for value in grouped_broadcasts
        ):
            return False
        fact_views = tuple(grouped_broadcasts)
    for requirement in tuple(getattr(card, "required_fields", ()) or ()):
        if not isinstance(requirement, str) or not requirement:
            return False
        require_absent = requirement.startswith("!")
        field_name = requirement[1:] if require_absent else requirement
        for facts in fact_views:
            present, value = _lookup_field(facts, field_name)
            materially_present = present and value is not None
            if require_absent == materially_present:
                return False
    return True


def _contraction_has_contracted_axis(item: _WorkItem) -> bool | None:
    attrs = item.facts.get("attrs")
    if not isinstance(attrs, Mapping):
        return None
    subscripts = attrs.get("subscripts")
    if not isinstance(subscripts, str) or "->" not in subscripts:
        return None
    inputs, output = subscripts.split("->", 1)
    input_labels = {
        char for char in inputs if char.isalpha()
    }
    output_labels = {char for char in output if char.isalpha()}
    return bool(input_labels - output_labels)


def _legacy_semantic_match(card: RuleCard, item: _WorkItem) -> bool:
    if card.id in {"materialize_index_tile", "materialize_pointer_tile"}:
        return item.facts.get("kind") in {"load", "store", "atomic"}
    if card.id == "realize_masked_access":
        return (
            item.role == "effect"
            and item.facts.get("kind") in {"load", "store", "atomic"}
            and item.facts.get("predicate") is not None
        )
    if card.id == "realize_unmasked_access":
        return (
            item.role == "effect"
            and item.facts.get("kind") in {"load", "store", "atomic"}
            and item.facts.get("predicate") is None
        )
    if card.id == "materialize_mask_tile":
        return item.role == "strategy:mask_materialization"
    if card.id == "realize_k_tiled_dot":
        return _contraction_has_contracted_axis(item) is True
    if card.id == "realize_outer_product":
        return _contraction_has_contracted_axis(item) is False
    return True


def _phase_key(card: RuleCard) -> tuple[int, str]:
    raw = getattr(card, "construction_phase", None)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw, ""
    if isinstance(raw, str) and raw:
        if raw.isdigit():
            return int(raw), ""
        names = {
            "domain": 10,
            "interface": 20,
            "schedule": 30,
            "address": 40,
            "mask": 45,
            "access": 50,
            "compute": 60,
            "output": 80,
        }
        return names.get(raw, 70), raw
    return _LEGACY_PHASE.get(card.id, 70), ""


def _registry_json(registry: RuleRegistry) -> dict[str, Any]:
    cards = sorted((card.to_json() for card in registry.cards), key=lambda x: x["id"])
    return {"cards": cards}


def _strategy_views(artifact: Gate1Artifact) -> tuple[list[_StrategyBindingView], dict[str, Any]]:
    strategy = getattr(artifact, "strategy", None)
    if strategy is None:
        return [], {"bindings": []}
    bindings = getattr(strategy, "bindings", None)
    if bindings is None and isinstance(strategy, Mapping):
        bindings = strategy.get("bindings", [])
    if bindings is None:
        bindings = []
    result: list[_StrategyBindingView] = []
    json_bindings: list[dict[str, str]] = []
    for index, binding in enumerate(bindings):
        if isinstance(binding, Mapping):
            target_ref = binding.get("target_ref")
            family = binding.get("family")
            choice = binding.get("choice")
        else:
            target_ref = getattr(binding, "target_ref", None)
            family = getattr(binding, "family", None)
            choice = getattr(binding, "choice", None)
        if not all(isinstance(value, str) and value for value in (target_ref, family, choice)):
            raise ComposeError(
                f"strategy binding {index} is not a closed target/family/choice triple"
            )
        result.append(_StrategyBindingView(index, target_ref, family, choice))
        json_bindings.append(
            {"target_ref": target_ref, "family": family, "choice": choice}
        )
    return result, {
        "bindings": sorted(
            json_bindings,
            key=lambda value: (value["target_ref"], value["family"], value["choice"]),
        )
    }


def _normalize_node_path(path: str) -> str:
    for prefix in ("j.intent.intent_function.", "intent_function.", "j.intent."):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def _ops_by_path(artifact: Gate1Artifact) -> dict[str, tuple[dict[str, Any], str]]:
    root = artifact.j.intent.to_json()["intent_function"]
    result: dict[str, tuple[dict[str, Any], str]] = {}

    def visit(ops: list[dict[str, Any]], path_prefix: str, pointer_prefix: str) -> None:
        for index, op in enumerate(ops):
            path = f"{path_prefix}[{index}]"
            pointer = f"{pointer_prefix}/{index}"
            result[path] = (op, pointer)
            body = op.get("body", [])
            if isinstance(body, list):
                visit(body, f"{path}.body", f"{pointer}/body")

    visit(root.get("body", []), "body", "/j/intent/intent_function/body")
    return result


def _semantic_key(kind: str, facts: Any) -> str:
    return f"{kind}:{_digest(facts)}"


def _reference_classes(artifact: Gate1Artifact) -> dict[str, str]:
    """Return alpha-neutral classes for every declared semantic handle.

    These values are used only for graph canonicalization.  They deliberately
    preserve ABI declaration positions and semantic kinds, while erasing
    Lift-chosen opaque spellings such as ``n7`` or ``temporary_load``.
    """

    classes: dict[str, str] = {}
    for index, tensor in enumerate(artifact.j.intent.inputs):
        classes[tensor.name] = f"@input:{index}"
    for index, tensor in enumerate(artifact.j.intent.outputs):
        classes.setdefault(tensor.name, f"@output:{index}")
    for index, parameter in enumerate(artifact.interface.parameters):
        classes.setdefault(
            parameter.name,
            f"@parameter:{_enum_value(parameter.kind)}:{index}",
        )
    for slot in artifact.schedule.slots:
        classes.setdefault(slot.name, f"@slot:{_enum_value(slot.kind)}")
    for node_id in artifact.j.node_ids:
        classes[node_id] = "@node"
    for domain_id in artifact.j.boundary.domains:
        classes[domain_id] = "@domain"
    for access in artifact.j.boundary.accesses:
        classes[access.access_id] = f"@access:{_enum_value(access.kind)}"
    for effect in artifact.j.boundary.effects:
        classes[effect.effect_id] = f"@effect:{_enum_value(effect.kind)}"
    for scope in getattr(artifact.j, "scopes", ()):
        classes[scope.scope_id] = f"@scope:{_enum_value(scope.kind)}"
    for storage in getattr(artifact.j, "storage", ()):
        classes[storage.storage_id] = "@storage"

    def visit(ops: Sequence[Any]) -> None:
        for op in ops:
            payload = op.to_json() if hasattr(op, "to_json") else op
            if not isinstance(payload, Mapping):
                continue
            results = payload.get("result")
            if isinstance(results, str):
                classes.setdefault(results, "@value")
            elif isinstance(results, list):
                for result in results:
                    if isinstance(result, str):
                        classes.setdefault(result, "@value")
            for state in payload.get("carry", ()):
                if not isinstance(state, Mapping):
                    continue
                for field, marker in (
                    ("state_id", "@state"),
                    ("body_argument", "@value"),
                    ("next_value", "@value"),
                    ("result_value", "@value"),
                ):
                    value = state.get(field)
                    if isinstance(value, str):
                        classes.setdefault(value, marker)
            for scan in payload.get("scan", ()):
                if isinstance(scan, Mapping) and isinstance(scan.get("name"), str):
                    classes.setdefault(scan["name"], "@scan")
            body = payload.get("body")
            if isinstance(body, list):
                visit(body)

    visit(artifact.j.intent.body)
    return classes


def _alpha_sanitize(value: Any, classes: Mapping[str, str]) -> Any:
    """Erase declared-handle spelling while retaining semantic structure."""

    if isinstance(value, str):
        exact = classes.get(value)
        if exact is not None:
            return exact
        return re.sub(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            lambda match: classes.get(match.group(0), match.group(0)),
            value,
        )
    if isinstance(value, list):
        return [_alpha_sanitize(item, classes) for item in value]
    if isinstance(value, tuple):
        return tuple(_alpha_sanitize(item, classes) for item in value)
    if isinstance(value, Mapping):
        return {
            classes.get(str(key), str(key)): _alpha_sanitize(item, classes)
            for key, item in value.items()
        }
    return value


def _refine_structural_keys(
    items: Sequence[_WorkItem], artifact: Gate1Artifact
) -> None:
    """Compute graph/strategy-aware keys independent of opaque id spelling."""

    classes = _reference_classes(artifact)
    by_key = {item.key: item for item in items}
    consumers: dict[str, set[str]] = {item.key: set() for item in items}
    for item in items:
        for dependency in item.dependencies:
            if dependency in consumers:
                consumers[dependency].add(item.key)
    base: dict[str, str] = {}
    for item in items:
        assert item.selected_card is not None
        strategy = None
        if item.strategy_binding is not None:
            strategy = {
                "family": item.strategy_binding.family,
                "choice": item.strategy_binding.choice,
            }
        base[item.key] = _digest(
            {
                "kind": item.kind,
                "role": (
                    "strategy"
                    if item.role.startswith("strategy:")
                    else item.role
                ),
                "facts": _alpha_sanitize(item.facts, classes),
                "rule": item.selected_card.id,
                "strategy": strategy,
            }
        )
    signatures = dict(base)
    # A fixed number of Weisfeiler-Lehman-style rounds is deterministic and
    # propagates both producer and consumer context through the whole graph.
    for _ in range(max(1, len(items))):
        signatures = {
            item.key: _digest(
                {
                    "base": base[item.key],
                    "dependencies": sorted(
                        signatures[key]
                        for key in item.dependencies
                        if key in by_key
                    ),
                    "consumers": sorted(
                        signatures[key] for key in consumers[item.key]
                    ),
                }
            )
            for item in items
        }
    for item in items:
        item.semantic_key = signatures[item.key]


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _effect_matches_access(effect: Any, access: Any) -> bool:
    access_ref = getattr(effect, "access_ref", None)
    if access_ref is not None and access_ref != getattr(access, "access_id", None):
        return False
    access_kind = _enum_value(getattr(access, "kind", None))
    effect_kind = _enum_value(getattr(effect, "kind", None))
    expected_access_kind = {
        "read": "load",
        "write": "store",
        "atomic": "atomic",
        "synchronize": "synchronize",
    }.get(effect_kind)
    if expected_access_kind is None or access_kind != expected_access_kind:
        return False
    if getattr(effect, "target", None) not in {
        getattr(access, "tensor", None),
        getattr(access, "access_id", None),
    }:
        return False
    effect_node = getattr(effect, "node_ref", None)
    access_node = getattr(access, "node_ref", None)
    return effect_node is None or access_node is None or effect_node == access_node


def _enumerate_work_items(artifact: Gate1Artifact) -> list[_WorkItem]:
    items: list[_WorkItem] = []
    ops = _ops_by_path(artifact)
    path_to_node: dict[str, str] = {}
    node_to_item: dict[str, str] = {}
    result_to_item: dict[str, str] = {}

    for node_id, raw_path in artifact.j.node_ids.items():
        path = _normalize_node_path(raw_path)
        if path not in ops:
            raise ComposeDependencyError(
                f"J node {node_id!r} references unsupported or missing path {raw_path!r}"
            )
        op, op_pointer = ops[path]
        op_name = op.get("op")
        if not isinstance(op_name, str) or not op_name:
            raise ComposeDependencyError(f"J node {node_id!r} has no operation kind")
        key = f"intent:{node_id}"
        item = _WorkItem(
            key=key,
            kind=f"intent_op:{op_name}",
            role="intent",
            target_ref=node_id,
            target_pointer=f"/j/node_ids/{_json_pointer_token(node_id)}",
            facts=copy.deepcopy(op),
            semantic_key=_semantic_key(f"intent_op:{op_name}", op),
            aliases=(node_id, raw_path, path),
            source_pointers=(op_pointer,),
        )
        items.append(item)
        path_to_node[path] = node_id
        node_to_item[node_id] = key
        node_to_item[raw_path] = key
        node_to_item[path] = key
        raw_result = op.get("result")
        results = [raw_result] if isinstance(raw_result, str) else raw_result or []
        for result in results:
            if isinstance(result, str):
                if result in result_to_item:
                    raise ComposeDependencyError(
                        f"J value {result!r} has multiple defining nodes"
                    )
                result_to_item[result] = key

    # A semantic loop is one closed work item: its card consumes the typed
    # carry/trip/scan/body/yield record, so body ops need not each have an
    # external node id.  Any other unaddressed operation would be silently
    # omitted and therefore fails closed.
    missing_paths: list[str] = []
    for path in sorted(set(ops) - set(path_to_node)):
        ancestor = path
        covered_by_loop = False
        while ".body[" in ancestor:
            ancestor = ancestor.rsplit(".body[", 1)[0]
            owner = ops.get(ancestor)
            if owner is not None and owner[0].get("op") == "loop":
                if ancestor in path_to_node:
                    covered_by_loop = True
                break
        if not covered_by_loop:
            missing_paths.append(path)
    if missing_paths:
        raise ComposeDependencyError(
            "J contains operation(s) without opaque node_ids: " + ", ".join(missing_paths)
        )

    # Def-use and nested-region dependencies.
    item_by_key = {item.key: item for item in items}
    for item in items:
        for operand in item.facts.get("operands", []):
            dependency = result_to_item.get(operand)
            if dependency is not None and dependency != item.key:
                item.dependencies.add(dependency)
        path = next(
            (alias for alias in item.aliases if alias.startswith("body")), None
        )
        if path and ".body[" in path:
            parent_path = path.rsplit(".body[", 1)[0]
            parent = node_to_item.get(parent_path)
            if parent is not None and parent != item.key:
                item.dependencies.add(parent)

    for index, (domain_id, expression) in enumerate(
        sorted(artifact.j.boundary.domains.items())
    ):
        items.append(
            _WorkItem(
                key=f"domain:{domain_id}",
                kind="boundary.domain",
                role="domain",
                target_ref=domain_id,
                target_pointer=f"/j/boundary/domains/{_json_pointer_token(domain_id)}",
                facts={"name": domain_id, "expression": expression},
                semantic_key=_semantic_key("boundary.domain", {"expression": expression}),
                aliases=(domain_id,),
                source_pointers=(
                    f"/j/boundary/domains/{_json_pointer_token(domain_id)}",
                ),
            )
        )

    for index, parameter in enumerate(artifact.interface.parameters):
        if parameter.kind is not ParameterKind.SCALAR:
            continue
        payload = parameter.to_json()
        items.append(
            _WorkItem(
                key=f"scalar:{parameter.name}",
                kind="interface.runtime_scalar",
                role="runtime_scalar",
                target_ref=parameter.name,
                target_pointer=f"/interface/parameters/{index}/name",
                facts=payload,
                semantic_key=_semantic_key("interface.runtime_scalar", payload),
                aliases=tuple(
                    value
                    for value in (parameter.name, parameter.tensor_ref)
                    if value is not None
                ),
                source_pointers=(f"/interface/parameters/{index}",),
            )
        )

    for index, slot in enumerate(artifact.schedule.slots):
        payload = slot.to_json()
        items.append(
            _WorkItem(
                key=f"slot:{slot.name}",
                kind="schedule.slot",
                role="schedule_slot",
                target_ref=slot.name,
                target_pointer=f"/schedule/slots/{index}/name",
                facts=payload,
                semantic_key=_semantic_key("schedule.slot", payload),
                aliases=(slot.name,),
                source_pointers=(f"/schedule/slots/{index}",),
            )
        )

    access_items: dict[str, tuple[_WorkItem, _WorkItem]] = {}
    matched_effects: set[int] = set()
    for index, access in enumerate(artifact.j.boundary.accesses):
        payload = access.to_json()
        pointer = f"/j/boundary/accesses/{index}"
        address = _WorkItem(
            key=f"access-address:{access.access_id}",
            kind="boundary.access.address",
            role="address",
            target_ref=access.access_id,
            target_pointer=f"{pointer}/access_id",
            facts=payload,
            semantic_key=_semantic_key("boundary.access.address", payload),
            aliases=(access.access_id,),
            source_pointers=(pointer,),
        )
        effect_sources = [pointer]
        for effect_index, effect in enumerate(artifact.j.boundary.effects):
            if _effect_matches_access(effect, access):
                effect_sources.append(f"/j/boundary/effects/{effect_index}")
                matched_effects.add(effect_index)
        effect = _WorkItem(
            key=f"access-effect:{access.access_id}",
            kind="boundary.access.effect",
            role="effect",
            target_ref=access.access_id,
            target_pointer=f"{pointer}/access_id",
            facts=payload,
            semantic_key=_semantic_key("boundary.access.effect", payload),
            aliases=(access.access_id,),
            source_pointers=tuple(effect_sources),
            dependencies={address.key},
        )
        node_dependency = node_to_item.get(access.node_ref or "")
        if node_dependency is not None:
            if access.kind is AccessKind.LOAD:
                item_by_key[node_dependency].dependencies.add(effect.key)
            else:
                address.dependencies.add(node_dependency)
        items.extend((address, effect))
        access_items[access.access_id] = (address, effect)

    unmatched_effects = sorted(
        set(range(len(artifact.j.boundary.effects))) - matched_effects
    )
    if unmatched_effects:
        raise ZeroRuleMatchError(
            "boundary effect(s) cannot be associated with a declared access: "
            + ", ".join(str(index) for index in unmatched_effects),
            work_item="boundary.effect",
        )

    # A BroadcastSpec describes one operand/axes relation, while strict pi has
    # one node-level ``realize_broadcast`` construction step.  Multi-input
    # operations (for example an eye-style row/column comparison) can therefore
    # carry several distinct broadcast declarations for the same semantic
    # node.  Coalesce those declarations into one work item without dropping
    # any facts or provenance.  Exact duplicate declarations are rejected as
    # Lift noise rather than silently changing the meaning of J as a set.
    broadcasts_by_node: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, broadcast in enumerate(artifact.j.boundary.broadcasts):
        broadcasts_by_node.setdefault(broadcast.node_ref, []).append(
            (index, broadcast.to_json())
        )

    for node_ref, declarations in broadcasts_by_node.items():
        payloads_by_bytes: dict[bytes, tuple[int, dict[str, Any]]] = {}
        for index, payload in declarations:
            canonical = _canonical_bytes(payload)
            if canonical in payloads_by_bytes:
                previous, _ = payloads_by_bytes[canonical]
                raise ComposeDependencyError(
                    "J contains duplicate broadcast declarations for node "
                    f"{node_ref!r} at indices {previous} and {index}",
                    work_item=f"broadcast:{node_ref}",
                )
            payloads_by_bytes[canonical] = (index, payload)

        ordered_declarations = sorted(
            declarations,
            key=lambda pair: _canonical_bytes(pair[1]),
        )
        payloads = [copy.deepcopy(payload) for _, payload in ordered_declarations]
        source_pointers = tuple(
            sorted(
                f"/j/boundary/broadcasts/{index}"
                for index, _ in declarations
            )
        )
        facts = {
            "node_ref": node_ref,
            "broadcasts": payloads,
        }
        key = f"broadcast:{node_ref}"
        item = _WorkItem(
            key=key,
            kind="boundary.broadcast",
            role="broadcast",
            target_ref=node_ref,
            target_pointer=f"{source_pointers[0]}/node_ref",
            facts=facts,
            semantic_key=_semantic_key("boundary.broadcast", facts),
            aliases=(node_ref,),
            source_pointers=source_pointers,
        )
        node = node_to_item.get(node_ref)
        for payload in payloads:
            operand_dependency = result_to_item.get(payload["operand"])
            if operand_dependency is not None and operand_dependency != node:
                item.dependencies.add(operand_dependency)
        if node is not None:
            item_by_key[node].dependencies.add(key)
        items.append(item)

    for index, mapping in enumerate(artifact.j.boundary.output_mappings):
        payload = mapping.to_json()
        item = _WorkItem(
            key=f"output:{index}:{mapping.output}",
            kind="boundary.output_mapping",
            role="output_mapping",
            target_ref=mapping.output,
            target_pointer=f"/j/boundary/output_mappings/{index}/output",
            facts=payload,
            semantic_key=_semantic_key("boundary.output_mapping", payload),
            aliases=tuple(
                dict.fromkeys(
                    value
                    for value in (mapping.output, mapping.value, mapping.node_ref)
                    if value is not None
                )
            ),
            source_pointers=(f"/j/boundary/output_mappings/{index}",),
        )
        dependency = node_to_item.get(mapping.node_ref or "")
        if dependency is None:
            dependency = result_to_item.get(mapping.value)
        if dependency is not None:
            item.dependencies.add(dependency)
        items.append(item)

    # Reduction boundary facts are context for the referenced semantic op; a
    # boundary reduction without such an op would otherwise be silently lost.
    for index, reduction in enumerate(artifact.j.boundary.reductions):
        dependency = node_to_item.get(reduction.node_ref)
        if dependency is None:
            raise ZeroRuleMatchError(
                f"reduction boundary {index} has no addressable semantic node",
                work_item=f"boundary.reduction:{index}",
            )
        target = item_by_key[dependency]
        target.facts.setdefault("boundary_reductions", []).append(reduction.to_json())
        target.source_pointers = (
            *target.source_pointers,
            f"/j/boundary/reductions/{index}",
        )

    return items


def _binding_matches_item(binding: _StrategyBindingView, item: _WorkItem) -> bool:
    return binding.target_ref == item.target_ref or binding.target_ref in item.aliases


def _select_card(
    item: _WorkItem,
    registry: RuleRegistry,
    bindings: Sequence[_StrategyBindingView],
    consumed_bindings: set[int],
) -> RuleCard:
    cards_for_kind = [
        card
        for card in registry.cards
        if item.kind in _card_target_kinds(card)
    ]

    if item.role.startswith("strategy:"):
        family = item.role.split(":", 1)[1]
        relevant = [
            binding
            for binding in bindings
            if binding.family == family and _binding_matches_item(binding, item)
        ]
    else:
        families = {
            family
            for card in cards_for_kind
            for family, _ in [_card_strategy(card)]
            if family is not None and family not in _SUPPLEMENTAL_STRATEGY_FAMILIES
        }
        relevant = [
            binding
            for binding in bindings
            if binding.family in families and _binding_matches_item(binding, item)
        ]

    if len(relevant) > 1:
        names = [f"{binding.family}={binding.choice}" for binding in relevant]
        raise AmbiguousRuleMatchError(
            f"multiple strategy bindings apply to work item {item.key!r}: {names!r}",
            work_item=item.key,
            candidates=names,
        )
    binding = relevant[0] if relevant else None

    candidates: list[RuleCard] = []
    for card in cards_for_kind:
        family, choice = _card_strategy(card)
        if item.role.startswith("strategy:"):
            expected_family = item.role.split(":", 1)[1]
            if binding is None or family != expected_family or choice != binding.choice:
                continue
        elif binding is not None:
            if family != binding.family or choice != binding.choice:
                continue
        else:
            if _selection_mode(card) != RuleSelectionMode.DERIVED.value:
                continue
            if family in _SUPPLEMENTAL_STRATEGY_FAMILIES:
                continue
        if not _required_fields_match(card, item):
            continue
        if not _legacy_semantic_match(card, item):
            continue
        candidates.append(card)

    candidate_ids = sorted(card.id for card in candidates)
    if not candidates:
        suffix = ""
        if binding is not None:
            suffix = f" for strategy {binding.family}={binding.choice}"
        raise ZeroRuleMatchError(
            f"no RuleCard matches work item {item.key!r} ({item.kind}){suffix}",
            work_item=item.key,
            candidates=(),
        )
    if len(candidates) > 1:
        raise AmbiguousRuleMatchError(
            f"multiple RuleCards match work item {item.key!r} ({item.kind}): "
            f"{candidate_ids!r}",
            work_item=item.key,
            candidates=candidate_ids,
        )
    item.strategy_binding = binding
    if binding is not None:
        consumed_bindings.add(binding.index)
    item.selected_card = candidates[0]
    return candidates[0]


def _add_supplemental_strategy_items(
    items: list[_WorkItem], bindings: Sequence[_StrategyBindingView]
) -> None:
    additions: list[_WorkItem] = []
    for binding in bindings:
        if binding.family not in _SUPPLEMENTAL_STRATEGY_FAMILIES:
            continue
        targets = [
            item
            for item in items
            if item.kind == "boundary.access.effect"
            and item.role == "effect"
            and _binding_matches_item(binding, item)
        ]
        if len(targets) != 1:
            error = ZeroRuleMatchError if not targets else AmbiguousRuleMatchError
            raise error(
                f"strategy binding {binding.family}={binding.choice} for "
                f"{binding.target_ref!r} resolves to {len(targets)} access effects",
                work_item=f"strategy:{binding.index}",
            )
        target = targets[0]
        key = f"strategy:{binding.index}:{binding.family}:{target.target_ref}"
        addition = _WorkItem(
            key=key,
            kind=target.kind,
            role=f"strategy:{binding.family}",
            target_ref=target.target_ref,
            target_pointer=target.target_pointer,
            facts=copy.deepcopy(target.facts),
            semantic_key=f"{target.semantic_key}:strategy:{binding.family}:{binding.choice}",
            aliases=target.aliases,
            source_pointers=target.source_pointers,
            dependencies=set(target.dependencies),
        )
        target.dependencies = {key}
        additions.append(addition)
    items.extend(additions)


def _append_source_pointer(item: _WorkItem, pointer: str) -> None:
    if pointer not in item.source_pointers:
        item.source_pointers = (*item.source_pointers, pointer)


def _result_names(facts: Mapping[str, Any]) -> tuple[str, ...]:
    value = facts.get("result")
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _event_item(items: Sequence[_WorkItem], event: str) -> _WorkItem | None:
    """Resolve a validated semantic event to its construction work item."""

    if event in {"scope_entry", "scope_exit"}:
        return None

    semantic = [
        item
        for item in items
        if item.role == "intent"
        and (
            event == item.target_ref
            or event in item.aliases
            or event in _result_names(item.facts)
            or any(
                isinstance(state, Mapping) and state.get("state_id") == event
                for state in item.facts.get("carry", [])
            )
        )
    ]
    if len(semantic) == 1:
        return semantic[0]

    effects = [
        item
        for item in items
        if item.role == "effect"
        and (
            event == item.target_ref
            or event in item.aliases
            or item.facts.get("tensor") == event
        )
    ]
    if len(effects) == 1:
        return effects[0]
    if len(semantic) > 1 or len(effects) > 1:
        matches = semantic or effects
        raise ComposeDependencyError(
            f"semantic event {event!r} resolves to multiple work items",
            work_item=f"event:{event}",
            candidates=sorted(item.key for item in matches),
        )
    return None


def _boundary_effect_item(
    items: Sequence[_WorkItem],
    effect: Any,
    *,
    work_item: str,
) -> _WorkItem:
    candidates = [
        item
        for item in items
        if item.role == "effect"
        and _effect_matches_access(
            effect,
            _AccessFactView(item.facts),
        )
    ]
    if len(candidates) != 1:
        error = ZeroRuleMatchError if not candidates else AmbiguousRuleMatchError
        raise error(
            f"{work_item} resolves to {len(candidates)} boundary accesses",
            work_item=work_item,
            candidates=sorted(item.key for item in candidates),
        )
    return candidates[0]


class _AccessFactView:
    """Attribute view over a serialized BoundaryAccess payload."""

    def __init__(self, facts: Mapping[str, Any]) -> None:
        self.kind = facts.get("kind")
        self.tensor = facts.get("tensor")
        self.access_id = facts.get("access_id")
        self.node_ref = facts.get("node_ref")


def _semantic_event_parts(event: Any) -> tuple[str, str | None]:
    if isinstance(event, str):
        if event in {"scope_entry", "scope_exit"}:
            return event, None
        return "legacy", event
    if isinstance(event, Mapping):
        kind = event.get("kind")
        ref = event.get("ref")
    else:
        kind = getattr(event, "kind", None)
        ref = getattr(event, "ref", None)
    kind = _enum_value(kind)
    if not isinstance(kind, str):
        raise ComposeDependencyError("semantic lifetime event has no valid kind")
    if ref is not None and not isinstance(ref, str):
        raise ComposeDependencyError("semantic lifetime event ref must be a string")
    return kind, ref


def _resolve_semantic_event(
    artifact: Gate1Artifact,
    items: Sequence[_WorkItem],
    event: Any,
    *,
    work_item: str,
) -> _WorkItem | None:
    kind, ref = _semantic_event_parts(event)
    if kind in {"scope_entry", "scope_exit"}:
        return None
    if ref is None:
        raise ComposeDependencyError(
            f"{work_item} event kind {kind!r} requires ref",
            work_item=work_item,
        )
    if kind == "access":
        matches = [
            item
            for item in items
            if item.role == "effect" and item.target_ref == ref
        ]
        if len(matches) == 1:
            return matches[0]
    elif kind == "effect":
        effects = [
            effect
            for effect in artifact.j.boundary.effects
            if getattr(effect, "effect_id", None) == ref
        ]
        if len(effects) == 1:
            return _boundary_effect_item(items, effects[0], work_item=work_item)
        if len(effects) > 1:
            raise AmbiguousRuleMatchError(
                f"{work_item} references duplicate effect id {ref!r}",
                work_item=work_item,
            )
    elif kind in {"node", "state", "legacy"}:
        resolved = _event_item(items, ref)
        if resolved is not None:
            return resolved
    raise ComposeDependencyError(
        f"{work_item} event {kind}:{ref} has no replay work item",
        work_item=work_item,
    )


def _apply_scope_storage_and_effect_dependencies(
    artifact: Gate1Artifact,
    items: Sequence[_WorkItem],
) -> None:
    """Consume P2 scope/lifetime facts without inventing frontend steps."""

    scopes = tuple(getattr(artifact.j, "scopes", ()) or ())
    storage = tuple(getattr(artifact.j, "storage", ()) or ())
    input_tensors = {tensor.name for tensor in artifact.j.intent.inputs}

    intent_items = [item for item in items if item.role == "intent"]
    pointer_to_item = {
        pointer: item
        for item in intent_items
        for pointer in item.source_pointers
        if pointer.startswith("/j/intent/")
    }

    for index, scope in enumerate(scopes):
        pointer = f"/j/scopes/{index}"
        scope_id = getattr(scope, "scope_id", None)
        captures = tuple(getattr(scope, "captures", ()) or ())
        definitions = tuple(getattr(scope, "definitions", ()) or ())
        effects = tuple(getattr(scope, "effects", ()) or ())
        members: list[_WorkItem] = []

        loop_owners = [
            item
            for item in intent_items
            if item.facts.get("scope_id") == scope_id
        ]
        for owner in loop_owners:
            members.append(owner)
            for owner_pointer in owner.source_pointers:
                if not owner_pointer.startswith("/j/intent/"):
                    continue
                body_prefix = f"{owner_pointer}/body/"
                members.extend(
                    item
                    for child_pointer, item in pointer_to_item.items()
                    if child_pointer.startswith(body_prefix)
                )

        kind = getattr(getattr(scope, "kind", None), "value", getattr(scope, "kind", None))
        if kind in {"function", "program_instance"}:
            nested = {item.key for item in loop_owners}
            members.extend(item for item in intent_items if item.key not in nested)

        members = list(dict.fromkeys(item.key for item in members))
        members = [next(item for item in items if item.key == key) for key in members]
        for member in members:
            _append_source_pointer(member, pointer)

        for value in (*captures, *definitions):
            producer = None if value in input_tensors else _event_item(items, value)
            if producer is None:
                continue
            _append_source_pointer(producer, pointer)
            if value in captures:
                for member in members:
                    if member.key != producer.key:
                        member.dependencies.add(producer.key)

        for event in effects:
            effect_item = _resolve_semantic_event(
                artifact,
                items,
                {"kind": "effect", "ref": event},
                work_item=f"scope:{scope_id}",
            )
            assert effect_item is not None
            _append_source_pointer(effect_item, pointer)

    for index, item in enumerate(storage):
        pointer = f"/j/storage/{index}"
        init_event = getattr(item, "init_event", None)
        last_use_event = getattr(item, "last_use_event", None)
        storage_id = getattr(item, "storage_id", index)
        init_item = _resolve_semantic_event(
            artifact,
            items,
            init_event,
            work_item=f"storage:{storage_id}:init",
        )
        last_item = _resolve_semantic_event(
            artifact,
            items,
            last_use_event,
            work_item=f"storage:{storage_id}:last_use",
        )
        members = tuple(getattr(item, "members", ()) or ())
        if not members:
            members = tuple(
                value
                for value in (
                    getattr(item, "logical_value", None),
                    getattr(item, "tensor_ref", None),
                )
                if isinstance(value, str)
            )
        value_items = [
            resolved
            for member in members
            if member not in input_tensors
            for resolved in [_event_item(items, member)]
            if resolved is not None
        ]
        storage_scope_id = getattr(item, "scope_id", None)
        scope_pointer = next(
            (
                f"/j/scopes/{scope_index}"
                for scope_index, scope in enumerate(scopes)
                if getattr(scope, "scope_id", None) == storage_scope_id
            ),
            None,
        )
        if scope_pointer is not None:
            value_items.extend(
                work for work in items if scope_pointer in work.source_pointers
            )
        touched = {
            work.key: work
            for work in (init_item, last_item, *value_items)
            if work is not None
        }
        init_kind, _ = _semantic_event_parts(init_event)
        last_kind, _ = _semantic_event_parts(last_use_event)
        if not touched and {init_kind, last_kind} != {"scope_entry", "scope_exit"}:
            raise ComposeDependencyError(
                f"semantic storage {storage_id!r} "
                "does not resolve to any replay work item"
            )
        for work in touched.values():
            _append_source_pointer(work, pointer)
        if (
            init_item is not None
            and last_item is not None
            and init_item.key != last_item.key
        ):
            last_item.dependencies.add(init_item.key)

    # Boundary.effects is a declaration set in the current schema, not an
    # implicit total order.  Consume every effect for applicability/provenance;
    # ordering that affects correctness must arrive through explicit storage
    # lifetime endpoints (or future partial-order metadata).
    for index, effect in enumerate(artifact.j.boundary.effects):
        current = _boundary_effect_item(
            items,
            effect,
            work_item=f"boundary.effect:{index}",
        )
        _append_source_pointer(current, f"/j/boundary/effects/{index}")


def _add_capability_dependencies(items: Sequence[_WorkItem]) -> None:
    providers: dict[str, list[_WorkItem]] = {}
    for item in items:
        assert item.selected_card is not None
        for capability in tuple(getattr(item.selected_card, "provides", ()) or ()):
            providers.setdefault(capability, []).append(item)

    for item in items:
        assert item.selected_card is not None
        for requirement in tuple(getattr(item.selected_card, "requires", ()) or ()):
            matches = [
                provider
                for provider in providers.get(requirement, [])
                if provider.key != item.key
            ]
            if len(matches) != 1:
                raise ComposeDependencyError(
                    f"RuleCard {item.selected_card.id!r} requires capability "
                    f"{requirement!r}, but {len(matches)} providers were selected",
                    work_item=item.key,
                    candidates=sorted(
                        provider.selected_card.id
                        for provider in matches
                        if provider.selected_card is not None
                    ),
                )
            item.dependencies.add(matches[0].key)


def _stable_topological_order(items: Sequence[_WorkItem]) -> list[_WorkItem]:
    by_key = {item.key: item for item in items}
    if len(by_key) != len(items):
        raise ComposeDependencyError("composer generated duplicate work-item identities")
    for item in items:
        missing = sorted(item.dependencies - set(by_key))
        if missing:
            raise ComposeDependencyError(
                f"work item {item.key!r} has missing dependencies {missing!r}",
                work_item=item.key,
            )

    remaining = {key: set(item.dependencies) for key, item in by_key.items()}
    ordered: list[_WorkItem] = []

    def key(item: _WorkItem) -> tuple[Any, ...]:
        assert item.selected_card is not None
        return (
            _phase_key(item.selected_card),
            item.semantic_key,
            item.kind,
            item.target_ref,
            item.selected_card.id,
            item.key,
        )

    while remaining:
        ready = sorted(
            (by_key[item_key] for item_key, deps in remaining.items() if not deps),
            key=key,
        )
        if not ready:
            cycle = sorted(remaining)
            raise ComposeDependencyError(
                f"replay construction dependencies contain a cycle: {cycle!r}",
                candidates=cycle,
            )
        selected = ready[0]
        ordered.append(selected)
        del remaining[selected.key]
        for deps in remaining.values():
            deps.discard(selected.key)
    return ordered


def _step_for_item(item: _WorkItem) -> ReplayStep:
    assert item.selected_card is not None
    if item.role in {"intent", "broadcast"}:
        return ReplayStep(item.selected_card.id, node_ref=item.target_ref)
    return ReplayStep(item.selected_card.id, refs=[item.target_ref])


def _canonical_reference_bindings(
    ordered: Sequence[_WorkItem],
) -> tuple[CanonicalReferenceBinding, ...]:
    stems = {
        "intent": "n",
        "domain": "d",
        "address": "a",
        "effect": "a",
        "runtime_scalar": "p",
        "schedule_slot": "c",
        "broadcast": "n",
        "output_mapping": "t",
    }
    grouped: dict[tuple[str, str], list[_WorkItem]] = {}
    for item in ordered:
        base_role = "effect" if item.role.startswith("strategy:") else item.role
        stem = stems.get(base_role, "r")
        identity = (stem, item.target_ref)
        grouped.setdefault(identity, []).append(item)

    descriptors: list[tuple[str, str, str, str, str]] = []
    for (stem, target_ref), items in grouped.items():
        signature = _digest(
            sorted(
                (
                    item.semantic_key,
                    item.kind,
                    item.role,
                    item.selected_card.id if item.selected_card is not None else "",
                )
                for item in items
            )
        )
        representative = min(
            items,
            key=lambda item: (
                item.semantic_key, item.kind, item.role, item.target_pointer
            ),
        )
        descriptors.append(
            (
                stem,
                signature,
                target_ref,
                representative.kind,
                representative.target_pointer,
            )
        )

    counters: dict[str, int] = {}
    bindings: list[CanonicalReferenceBinding] = []
    for stem, _, target_ref, kind, source_pointer in sorted(descriptors):
        index = counters.get(stem, 0)
        counters[stem] = index + 1
        bindings.append(
            CanonicalReferenceBinding(
                canonical_ref=f"{stem}{index}",
                materialized_ref=target_ref,
                kind=kind,
                source_pointer=source_pointer,
            )
        )
    return tuple(bindings)


def _canonical_plan_json(
    plan: ReplayPlan,
    bindings: Sequence[CanonicalReferenceBinding],
) -> dict[str, Any]:
    aliases: dict[str, str] = {}
    for binding in bindings:
        previous = aliases.setdefault(
            binding.materialized_ref, binding.canonical_ref
        )
        if previous != binding.canonical_ref:
            raise ComposeDependencyError(
                "one materialized replay reference maps to multiple canonical ids"
            )
    return {
        "steps": [
            {
                "rule": step.rule,
                **(
                    {"node_ref": aliases.get(step.node_ref, step.node_ref)}
                    if step.node_ref is not None
                    else {}
                ),
                **(
                    {"refs": [aliases.get(ref, ref) for ref in step.refs]}
                    if step.refs
                    else {}
                ),
            }
            for step in plan.steps
        ]
    }


def _canonical_input_views(
    artifact: Gate1Artifact,
    strategy_json: dict[str, Any],
    registry: RuleRegistry,
    items: Sequence[_WorkItem],
    reference_bindings: Sequence[CanonicalReferenceBinding],
) -> tuple[Any, Any, Any, Any, Any]:
    classes = _reference_classes(artifact)
    classes.update(
        {
            binding.materialized_ref: binding.canonical_ref
            for binding in reference_bindings
        }
    )
    by_key = {item.key: item for item in items}
    j_items = [
        item
        for item in items
        if item.target_pointer.startswith("/j/")
        or any(pointer.startswith("/j/") for pointer in item.source_pointers)
    ]
    j_json = {
        "inputs": _alpha_sanitize(
            [tensor.to_json() for tensor in artifact.j.intent.inputs], classes
        ),
        "outputs": _alpha_sanitize(
            [tensor.to_json() for tensor in artifact.j.intent.outputs], classes
        ),
        "work_graph": sorted(
            (
                {
                    "signature": item.semantic_key,
                    "kind": item.kind,
                    "role": (
                        "strategy"
                        if item.role.startswith("strategy:")
                        else item.role
                    ),
                    "facts": _alpha_sanitize(item.facts, classes),
                    "dependencies": sorted(
                        by_key[key].semantic_key
                        for key in item.dependencies
                        if key in by_key
                    ),
                }
                for item in j_items
            ),
            key=_canonical_bytes,
        ),
        # Alias/scope/storage declarations carry relations that do not always
        # instantiate their own RuleCard work item, so retain them explicitly.
        "aliases": sorted(
            (
                _alpha_sanitize(item.to_json(), classes)
                for item in artifact.j.boundary.aliases
            ),
            key=_canonical_bytes,
        ),
        "scopes": sorted(
            (
                _alpha_sanitize(item.to_json(), classes)
                for item in getattr(artifact.j, "scopes", ())
            ),
            key=_canonical_bytes,
        ),
        "storage": sorted(
            (
                _alpha_sanitize(item.to_json(), classes)
                for item in getattr(artifact.j, "storage", ())
            ),
            key=_canonical_bytes,
        ),
    }
    interface_json = _alpha_sanitize(artifact.interface.to_json(), classes)
    interface_json["kernel_id"] = "@kernel"
    interface_json["function_name"] = "@function"
    schedule_json = _alpha_sanitize(artifact.schedule.to_json(), classes)
    strategy_json = _alpha_sanitize(strategy_json, classes)

    # These collections are declaration sets rather than semantic sequences.
    slots = schedule_json.get("slots")
    if isinstance(slots, list):
        schedule_json["slots"] = sorted(slots, key=_canonical_bytes)
    compiler_slots = interface_json.get("compiler_slots")
    if isinstance(compiler_slots, list):
        interface_json["compiler_slots"] = sorted(compiler_slots)
    return j_json, interface_json, schedule_json, strategy_json, _registry_json(registry)


def compose_replay_plan(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
) -> ComposeResult:
    """Deterministically derive strict pi and complete leaf provenance.

    ``artifact.plan`` is deliberately ignored.  A caller may therefore use
    this function to verify a persisted materialization without allowing the
    saved plan to influence the result.
    """

    if not isinstance(artifact, Gate1Artifact):
        artifact = Gate1Artifact.from_json(artifact)
    if not isinstance(registry, RuleRegistry):
        registry = RuleRegistry.from_json(registry)

    bindings, strategy_json = _strategy_views(artifact)
    items = _enumerate_work_items(artifact)
    _add_supplemental_strategy_items(items, bindings)
    _apply_scope_storage_and_effect_dependencies(artifact, items)

    consumed_bindings: set[int] = set()
    for item in items:
        _select_card(item, registry, bindings, consumed_bindings)
    unconsumed = [binding for binding in bindings if binding.index not in consumed_bindings]
    if unconsumed:
        binding = unconsumed[0]
        raise ZeroRuleMatchError(
            f"strategy binding {binding.family}={binding.choice} for "
            f"{binding.target_ref!r} did not select any work item",
            work_item=f"strategy:{binding.index}",
        )

    _add_capability_dependencies(items)
    _refine_structural_keys(items, artifact)
    ordered = _stable_topological_order(items)
    steps = [_step_for_item(item) for item in ordered]
    plan = ReplayPlan(steps)

    markers: set[tuple[str, str | None, tuple[str, ...]]] = set()
    for step in plan.steps:
        marker = (step.rule, step.node_ref, tuple(step.refs))
        if marker in markers:
            raise ComposeDependencyError(
                f"composer produced duplicate replay step {marker!r}"
            )
        markers.add(marker)

    registry_cards = sorted(registry.cards, key=lambda card: card.id)
    registry_index = {card.id: index for index, card in enumerate(registry_cards)}
    leaf_sources: dict[str, tuple[ProvenanceSource, ...]] = {}
    for index, (item, step) in enumerate(zip(ordered, plan.steps, strict=True)):
        assert item.selected_card is not None
        rule_sources = [
            ProvenanceSource(
                "rule_registry",
                f"/cards/{registry_index[item.selected_card.id]}/id",
            )
        ]
        rule_sources.extend(
            ProvenanceSource(_owner_for_pointer(pointer), pointer)
            for pointer in item.source_pointers
        )
        if item.strategy_binding is not None:
            rule_sources.append(
                ProvenanceSource(
                    "strategy",
                    f"{item.strategy_binding.pointer}/choice",
                )
            )
        leaf_sources[f"/steps/{index}/rule"] = tuple(
            dict.fromkeys(rule_sources)
        )
        if step.node_ref is not None:
            target_pointers = (
                tuple(f"{pointer}/node_ref" for pointer in item.source_pointers)
                if item.role == "broadcast"
                else (item.target_pointer,)
            )
            leaf_sources[f"/steps/{index}/node_ref"] = tuple(
                ProvenanceSource(_owner_for_pointer(pointer), pointer)
                for pointer in target_pointers
            )
        else:
            leaf_sources[f"/steps/{index}/refs/0"] = (
                ProvenanceSource(_owner_for_pointer(item.target_pointer), item.target_pointer),
            )

    reference_bindings = _canonical_reference_bindings(ordered)
    j_json, interface_json, schedule_json, strategy_json, registry_json = (
        _canonical_input_views(
            artifact,
            strategy_json,
            registry,
            items,
            reference_bindings,
        )
    )
    provenance = ComposeProvenance(
        composer_version=COMPOSER_VERSION,
        j_digest=_digest(j_json),
        interface_digest=_digest(interface_json),
        schedule_digest=_digest(schedule_json),
        strategy_digest=_digest(strategy_json),
        registry_digest=_digest(registry_json),
        plan_digest=_digest(_canonical_plan_json(plan, reference_bindings)),
        leaf_sources=leaf_sources,
        reference_bindings=reference_bindings,
    )
    provenance.validate(plan)
    return ComposeResult(plan=plan, provenance=provenance)


def _owner_for_pointer(pointer: str) -> str:
    if pointer.startswith("/j/"):
        return "j"
    if pointer.startswith("/interface/"):
        return "interface"
    if pointer.startswith("/schedule/"):
        return "schedule"
    if pointer.startswith("/strategy/"):
        return "strategy"
    return "protocol"


def assert_replay_plan_matches(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    plan: ReplayPlan | Mapping[str, Any] | None = None,
) -> ComposeResult:
    """Assert byte-canonical equality with a fresh deterministic composition."""

    result = compose_replay_plan(artifact, registry)
    actual = artifact.plan if plan is None else plan
    if not isinstance(actual, ReplayPlan):
        actual = ReplayPlan.from_json(dict(actual))
    actual_digest = _digest(
        _canonical_plan_json(actual, result.provenance.reference_bindings)
    )
    if actual_digest != result.plan_digest:
        raise ComposeMismatchError(result.plan_digest, actual_digest)
    return result


def materialize_replay_plan(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
) -> tuple[Gate1Artifact, ComposeResult]:
    """Return an artifact whose pi/provenance are fresh composer outputs."""

    result = compose_replay_plan(artifact, registry)
    return (
        replace(
            artifact,
            plan=result.plan,
            compose_provenance=result.provenance.to_json(),
        ),
        result,
    )


__all__ = [
    "COMPOSER_VERSION",
    "AmbiguousRuleMatchError",
    "CanonicalReferenceBinding",
    "ComposeDependencyError",
    "ComposeError",
    "ComposeMismatchError",
    "ComposeProvenance",
    "ComposeResult",
    "ProvenanceSource",
    "ZeroRuleMatchError",
    "assert_replay_plan_matches",
    "compose_replay_plan",
    "materialize_replay_plan",
]
