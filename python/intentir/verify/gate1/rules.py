"""Frontend rule cards and deterministic plan-to-card retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .model import (
    Frontend,
    Gate1Artifact,
    ParameterKind,
    ReplayPlan,
    SchemaError,
    StrictJSONMixin,
    _enum,
    _strict_object,
    _string,
    _string_list,
)


class RuleSelectionMode(str, Enum):
    """Whether a rule is the canonical mechanism or a source-specific alternative."""

    DERIVED = "derived"
    OVERRIDE = "override"


@dataclass
class RuleCard(StrictJSONMixin):
    """One system-owned replay mechanism card.

    Concrete frontend API spellings are allowed here and only here.  The
    lift-produced strict replay plan explicitly instantiates canonical and
    override cards, while carrying only card ids and references into J/B/C or
    the interface.
    """

    id: str
    frontend: Frontend
    version: str = "1"
    applies_to: list[str] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)
    api_mapping: dict[str, str] = field(default_factory=dict)
    example: str = ""
    pitfalls: list[str] = field(default_factory=list)
    selection_mode: RuleSelectionMode = RuleSelectionMode.DERIVED
    override_family: str | None = None

    def __post_init__(self) -> None:
        self.id = _string(self.id, "RuleCard.id")
        self.frontend = _enum(Frontend, self.frontend, "RuleCard.frontend")
        self.version = _string(self.version, "RuleCard.version")
        self.applies_to = _string_list(self.applies_to, "RuleCard.applies_to", unique=True)
        self.reads = _string_list(self.reads, "RuleCard.reads", unique=True)
        if not isinstance(self.api_mapping, dict):
            raise SchemaError("RuleCard.api_mapping must be an object")
        self.api_mapping = {
            _string(key, "RuleCard.api_mapping key"): _string(value, f"RuleCard.api_mapping.{key}")
            for key, value in self.api_mapping.items()
        }
        self.example = _string(self.example, "RuleCard.example", nonempty=False)
        self.pitfalls = _string_list(self.pitfalls, "RuleCard.pitfalls", unique=True)
        self.selection_mode = _enum(
            RuleSelectionMode,
            self.selection_mode,
            "RuleCard.selection_mode",
        )
        if self.override_family is not None:
            self.override_family = _string(
                self.override_family, "RuleCard.override_family"
            )
            if re.fullmatch(r"[a-z][a-z0-9_]*", self.override_family) is None:
                raise SchemaError(
                    "RuleCard.override_family must be a lower_snake_case identifier"
                )
        if self.selection_mode is RuleSelectionMode.DERIVED:
            if self.override_family is not None:
                raise SchemaError(
                    "derived RuleCard must not declare override_family"
                )
        elif self.override_family is None:
            raise SchemaError(
                "override RuleCard requires override_family"
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RuleCard":
        data = _strict_object(
            value,
            path="RuleCard",
            allowed={
                "id", "frontend", "version", "applies_to", "reads",
                "api_mapping", "example", "pitfalls", "selection_mode",
                "override_family",
            },
            required={"id", "frontend"},
        )
        api_mapping = data.get("api_mapping", {})
        if not isinstance(api_mapping, dict):
            raise SchemaError("RuleCard.api_mapping must be an object")
        return cls(
            id=data["id"],
            frontend=data["frontend"],
            version=data.get("version", "1"),
            applies_to=_string_list(data.get("applies_to", []), "RuleCard.applies_to", unique=True),
            reads=_string_list(data.get("reads", []), "RuleCard.reads", unique=True),
            api_mapping=dict(api_mapping),
            example=data.get("example", ""),
            pitfalls=_string_list(data.get("pitfalls", []), "RuleCard.pitfalls", unique=True),
            selection_mode=data.get("selection_mode", RuleSelectionMode.DERIVED.value),
            override_family=data.get("override_family"),
        )

    def to_json(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "frontend": self.frontend.value,
            "version": self.version,
            "applies_to": list(self.applies_to),
            "reads": list(self.reads),
            "api_mapping": dict(self.api_mapping),
            "example": self.example,
            "pitfalls": list(self.pitfalls),
            "selection_mode": self.selection_mode.value,
        }
        if self.override_family is not None:
            result["override_family"] = self.override_family
        return result


class RuleRegistry(StrictJSONMixin):
    """Ordered, duplicate-safe registry of frontend rule cards."""

    def __init__(self, cards: list[RuleCard] | None = None) -> None:
        self._cards: dict[str, RuleCard] = {}
        for card in cards or []:
            self.register(card)

    @property
    def cards(self) -> tuple[RuleCard, ...]:
        return tuple(self._cards.values())

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, rule_id: object) -> bool:
        return isinstance(rule_id, str) and rule_id in self._cards

    @property
    def digest(self) -> str:
        """Stable registry identity recorded with prompts/evidence."""

        payload = json.dumps(
            self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def register(self, card: RuleCard) -> None:
        if not isinstance(card, RuleCard):
            card = RuleCard.from_json(card)
        if card.id in self._cards:
            raise SchemaError(f"duplicate rule card id: {card.id}")
        self._cards[card.id] = card

    def get(self, rule_id: str, *, frontend: Frontend | str | None = None) -> RuleCard:
        rule_id = _string(rule_id, "RuleRegistry.rule_id")
        try:
            card = self._cards[rule_id]
        except KeyError as exc:
            raise SchemaError(f"unknown replay rule: {rule_id}") from exc
        if frontend is not None:
            expected = _enum(Frontend, frontend, "RuleRegistry.frontend")
            if card.frontend is not expected:
                raise SchemaError(
                    f"rule {rule_id!r} belongs to {card.frontend.value}, not {expected.value}"
                )
        return card

    def resolve(
        self,
        plan: ReplayPlan,
        *,
        frontend: Frontend | str | None = None,
    ) -> list[RuleCard]:
        """Resolve cards in canonical plan order, deduplicating rule ids."""

        if not isinstance(plan, ReplayPlan):
            plan = ReplayPlan.from_json(plan)
        result: list[RuleCard] = []
        seen: set[str] = set()
        for step in plan.steps:
            if step.rule in seen:
                continue
            result.append(self.get(step.rule, frontend=frontend))
            seen.add(step.rule)
        return result

    def validate_strict_plan(
        self,
        plan: ReplayPlan,
        *,
        frontend: Frontend | str | None = None,
        artifact: Gate1Artifact | None = None,
    ) -> ReplayPlan:
        """Validate pi as a complete ordered list of replay-rule instances."""

        if not isinstance(plan, ReplayPlan):
            plan = ReplayPlan.from_json(plan)
        seen_steps: set[tuple[str, str, tuple[str, ...]]] = set()
        seen_overrides: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
        for index, step in enumerate(plan.steps):
            card = self.get(step.rule, frontend=frontend)
            target_ref = step.node_ref or (step.refs[0] if step.refs else None)
            if target_ref is None:
                raise SchemaError(
                    f"replay rule {step.rule!r} requires a target reference "
                    f"(plan.steps[{index}])"
                )
            if step.node_ref is not None and step.node_ref in step.refs:
                raise SchemaError(
                    f"plan target {step.node_ref!r} must not be repeated "
                    f"inside refs (plan.steps[{index}])"
                )
            marker = (step.rule, target_ref, tuple(step.refs))
            if marker in seen_steps:
                raise SchemaError(
                    f"duplicate replay rule {step.rule!r} for target {target_ref!r}"
                )
            seen_steps.add(marker)
            if card.selection_mode is not RuleSelectionMode.OVERRIDE:
                continue
            assert card.override_family is not None
            key = (card.override_family, target_ref)
            value = (step.rule, tuple(step.refs))
            if key in seen_overrides:
                previous_rule, previous_refs = seen_overrides[key]
                kind = "duplicate" if value == (previous_rule, previous_refs) else "conflicting"
                raise SchemaError(
                    f"{kind} pi override for family {card.override_family!r} "
                    f"and target {target_ref!r}"
                )
            seen_overrides[key] = value
        if artifact is not None:
            self._validate_strict_coverage(plan, artifact, frontend=frontend)
        return plan

    def _validate_strict_coverage(
        self,
        plan: ReplayPlan,
        artifact: Gate1Artifact,
        *,
        frontend: Frontend | str | None = None,
    ) -> None:
        """Fail closed when a known J/B/C realization is absent from strict pi."""

        if not isinstance(artifact, Gate1Artifact):
            artifact = Gate1Artifact.from_json(artifact)
        selected: dict[str, set[str]] = {}
        for step in plan.steps:
            target = step.target_ref
            assert target is not None
            selected.setdefault(step.rule, set()).add(target)

        def require(
            rules: set[str],
            targets: set[str],
            description: str,
        ) -> None:
            if any(selected.get(rule, set()) & targets for rule in rules):
                return
            raise SchemaError(
                f"strict pi is missing {description}; expected one of "
                f"{sorted(rules)!r} targeting one of {sorted(targets)!r}"
            )

        function_json = artifact.j.intent.to_json()["intent_function"]

        def operation_at(path: str) -> dict[str, Any]:
            indices = [int(value) for value in re.findall(r"body\[(\d+)\]", path)]
            if not indices:
                raise SchemaError(f"unsupported J node path in strict pi: {path!r}")
            operations = function_json.get("body", [])
            operation: dict[str, Any] | None = None
            for depth, index in enumerate(indices):
                if index >= len(operations):
                    raise SchemaError(f"J node path is out of range: {path!r}")
                operation = operations[index]
                if depth + 1 < len(indices):
                    operations = operation.get("body", [])
            assert operation is not None
            return operation

        target_tags: dict[str, set[str]] = {}

        def tag(target: str | None, *tags: str) -> None:
            if target is None:
                return
            target_tags.setdefault(target, set()).update(tags)

        for node_id, path in artifact.j.node_ids.items():
            op_name = operation_at(path).get("op")
            if isinstance(op_name, str):
                tag(node_id, op_name)
                tag(path, op_name)
        for tensor in artifact.j.intent.inputs:
            tag(tensor.name, "intent.inputs")
        for tensor in artifact.j.intent.outputs:
            tag(tensor.name, "intent.outputs")
        for domain_id in artifact.j.boundary.domains:
            tag(domain_id, "boundary.domains", "interface.grid")
        for access in artifact.j.boundary.accesses:
            tag(access.access_id, "boundary.accesses")
        for broadcast in artifact.j.boundary.broadcasts:
            tag(broadcast.node_ref, "boundary.broadcasts")
        for reduction in artifact.j.boundary.reductions:
            tag(reduction.node_ref, "boundary.reductions")
        for mapping in artifact.j.boundary.output_mappings:
            tag(mapping.output, "boundary.output_mappings")
            tag(mapping.value, "boundary.output_mappings")
            tag(mapping.node_ref, "boundary.output_mappings")
        for parameter in artifact.interface.parameters:
            tag(parameter.name, "interface.parameters")
        for slot in artifact.schedule.slots:
            tag(slot.name, "schedule.slots")

        selected_by_target: dict[str, set[str]] = {}
        for step in plan.steps:
            target = step.target_ref
            assert target is not None
            card = self.get(step.rule, frontend=frontend)
            tags = target_tags.get(target, set())
            if card.applies_to and not (set(card.applies_to) & tags):
                raise SchemaError(
                    f"strict pi rule {step.rule!r} is not applicable to target "
                    f"{target!r}; target categories are {sorted(tags)!r}"
                )
            selected_by_target.setdefault(target, set()).add(step.rule)

        exclusive_mechanisms = (
            {"bind_program_domain", "bind_wide_program_domain"},
            {"materialize_index_tile", "materialize_pointer_tile"},
            {"realize_masked_access", "realize_unmasked_access"},
            {"realize_reduce", "realize_flattened_reduce"},
            {"realize_k_tiled_dot", "realize_outer_product"},
        )
        for target, rules_for_target in selected_by_target.items():
            for alternatives in exclusive_mechanisms:
                conflict = rules_for_target & alternatives
                if len(conflict) > 1:
                    raise SchemaError(
                        f"strict pi selects mutually exclusive replay mechanisms "
                        f"{sorted(conflict)!r} for target {target!r}"
                    )

        rule_by_op = {
            "map_unary": {"realize_elementwise"},
            "map_binary": {"realize_elementwise"},
            "const": {"realize_const"},
            "reduce": {"realize_reduce", "realize_flattened_reduce"},
            "contraction": {"realize_k_tiled_dot", "realize_outer_product"},
            "cast": {"realize_cast"},
            "select": {"realize_select"},
            "loop": {"realize_loop_carry"},
            "gather": {"realize_indexed_access"},
        }
        for node_id, path in artifact.j.node_ids.items():
            operation = operation_at(path)
            op_name = operation.get("op")
            rules = rule_by_op.get(op_name)
            if rules is None:
                raise SchemaError(
                    f"strict pi has no registered coverage rule for IntentIR op "
                    f"{op_name!r} at {node_id!r}"
                )
            require(rules, {node_id, path}, f"J node {node_id!r} ({op_name})")

        for access in artifact.j.boundary.accesses:
            target = {access.access_id}
            require(
                {"materialize_index_tile", "materialize_pointer_tile"},
                target,
                f"address materialization for access {access.access_id!r}",
            )
            access_rule = (
                "realize_masked_access"
                if access.predicate is not None
                else "realize_unmasked_access"
            )
            require(
                {access_rule},
                target,
                f"{access_rule} for access {access.access_id!r}",
            )

        for broadcast in artifact.j.boundary.broadcasts:
            require(
                {"realize_broadcast"},
                {broadcast.node_ref},
                f"broadcast for node {broadcast.node_ref!r}",
            )

        for mapping in artifact.j.boundary.output_mappings:
            targets = {mapping.output, mapping.value}
            if mapping.node_ref is not None:
                targets.add(mapping.node_ref)
            require(
                {"realize_output_mapping"},
                targets,
                f"output mapping for {mapping.output!r}",
            )

        scalar_parameters = [
            parameter
            for parameter in artifact.interface.parameters
            if parameter.kind is ParameterKind.SCALAR
        ]
        for parameter in scalar_parameters:
            targets = {parameter.name}
            if parameter.tensor_ref is not None:
                targets.add(parameter.tensor_ref)
            require(
                {"bind_runtime_scalar"},
                targets,
                f"runtime scalar {parameter.name!r}",
            )

        for slot in artifact.schedule.slots:
            require(
                {"declare_schedule_slot"},
                {slot.name},
                f"schedule slot {slot.name!r}",
            )

        domain_rules = (
            selected.get("bind_program_domain", set())
            | selected.get("bind_wide_program_domain", set())
        ) & set(artifact.j.boundary.domains)
        if len(domain_rules) < len(artifact.interface.grid):
            raise SchemaError(
                "strict pi must bind at least one boundary domain for every grid axis"
            )

    def canonicalize_overrides(
        self,
        plan: ReplayPlan,
        *,
        frontend: Frontend | str | None = None,
    ) -> ReplayPlan:
        """Compatibility alias for callers from the historical sparse-pi protocol."""

        return self.validate_strict_plan(plan, frontend=frontend)

    def baseline_cards(
        self,
        *,
        frontend: Frontend | str | None = None,
    ) -> list[RuleCard]:
        """Return canonical mechanism cards for registry inspection.

        Strict pi must still instantiate these cards explicitly; this method no
        longer injects hidden replay steps into the writer payload.
        """

        expected = (
            _enum(Frontend, frontend, "RuleRegistry.frontend")
            if frontend is not None
            else None
        )
        return [
            card
            for card in self._cards.values()
            if card.selection_mode is RuleSelectionMode.DERIVED
            and (expected is None or card.frontend is expected)
        ]

    def rewriter_view(
        self,
        plan: ReplayPlan,
        *,
        frontend: Frontend | str | None = None,
        artifact: Gate1Artifact | None = None,
    ) -> dict[str, Any]:
        strict_plan = self.validate_strict_plan(
            plan, frontend=frontend, artifact=artifact
        )
        cards = self.resolve(strict_plan, frontend=frontend)
        return {
            "plan": strict_plan.to_json(),
            "selection": [
                {
                    "rule": step.rule,
                    "selection_mode": self.get(
                        step.rule, frontend=frontend
                    ).selection_mode.value,
                    **(
                        {
                            "override_family": self.get(
                                step.rule, frontend=frontend
                            ).override_family
                        }
                        if self.get(step.rule, frontend=frontend).override_family
                        is not None
                        else {}
                    ),
                    "target_ref": step.node_ref or step.refs[0],
                    **({"node_ref": step.node_ref} if step.node_ref is not None else {}),
                    **({"refs": list(step.refs)} if step.refs else {}),
                }
                for step in strict_plan.steps
            ],
            "cards": [card.to_json() for card in cards],
        }

    def api_tokens(self, frontend: Frontend | str | None = None) -> set[str]:
        """Return concrete API spellings useful to the static leakage scanner."""

        expected = _enum(Frontend, frontend, "RuleRegistry.frontend") if frontend is not None else None
        tokens: set[str] = set()
        for card in self._cards.values():
            if expected is not None and card.frontend is not expected:
                continue
            for spelling in card.api_mapping.values():
                # A rule mapping may also describe a generic operator or a
                # construction recipe (for example ``+`` or ``1 / operand``).
                # Those are valid J/B expression syntax, not frontend API
                # names, and must not poison static leakage scanning.  Keep
                # qualified API names plus standalone identifier spellings.
                qualified = re.findall(
                    r"[A-Za-z_]\w*(?:(?:\.|::)[A-Za-z_]\w*)+", spelling
                )
                if qualified:
                    tokens.update(qualified)
                elif spelling.isidentifier():
                    tokens.add(spelling)
        return tokens

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RuleRegistry":
        data = _strict_object(
            value, path="RuleRegistry", allowed={"cards"}, required={"cards"},
        )
        cards = data["cards"]
        if not isinstance(cards, list):
            raise SchemaError("RuleRegistry.cards must be an array")
        return cls([RuleCard.from_json(card) for card in cards])

    def to_json(self) -> dict[str, Any]:
        return {"cards": [card.to_json() for card in self._cards.values()]}


def default_triton_rule_registry() -> RuleRegistry:
    """Return the reviewed Triton cards used by the strict SCRC replay plan.

    Cards deliberately describe mechanisms and expose all relevant frontend
    spellings.  They do not choose an operator or boundary value: those choices
    must come from J and references in pi.
    """

    cards = [
        RuleCard(
            "bind_program_domain",
            Frontend.TRITON,
            applies_to=["boundary.domains", "interface.grid"],
            reads=["domain expression", "grid axis"],
            api_mapping={"program_coordinate": "tl.program_id"},
            example="Bind each declared grid axis with tl.program_id(axis).",
            pitfalls=["Do not infer grid rank from a tensor shape."],
        ),
        RuleCard(
            "bind_wide_program_domain",
            Frontend.TRITON,
            selection_mode=RuleSelectionMode.OVERRIDE,
            override_family="program_coordinate_width",
            applies_to=["boundary.domains", "interface.grid"],
            reads=["domain expression", "grid axis", "coordinate integer width"],
            api_mapping={
                "program_coordinate_i64": "tl.program_id(axis).to(tl.int64)"
            },
            example=(
                "Bind the declared grid axis and preserve an explicitly widened "
                "64-bit program coordinate before forming offsets."
            ),
            pitfalls=[
                "Select this only when pi records the source's wide-coordinate mechanism.",
                "Do not widen a source that uses the frontend's native coordinate type.",
            ],
        ),
        RuleCard(
            "materialize_index_tile",
            Frontend.TRITON,
            version="2",
            applies_to=["boundary.accesses"],
            reads=["index_expression", "schedule slot"],
            api_mapping={"lane_range": "tl.arange"},
            example=(
                "Build the complete integer tile offset first, then add that tile "
                "to each pointer."
            ),
            pitfalls=[
                "Keep every C value symbolic until injection.",
                "Do not use this card when pi selects staged pointer-base addition.",
            ],
        ),
        RuleCard(
            "materialize_pointer_tile",
            Frontend.TRITON,
            version="2",
            selection_mode=RuleSelectionMode.OVERRIDE,
            override_family="address_association",
            applies_to=["boundary.accesses"],
            reads=["index_expression", "schedule slot"],
            api_mapping={"lane_range": "tl.arange"},
            example=(
                "Preserve staged row-major address construction as pointer + scalar "
                "base + lane tile; do not pre-sum base and lanes into one integer tile."
            ),
            pitfalls=[
                "Use the same staged pointer association independently for every buffer.",
                "Precomputed coordinate components are not a complete integer offset; "
                "preserve pointer + row-term + column-term association.",
            ],
        ),
        RuleCard(
            "realize_masked_access",
            Frontend.TRITON,
            version="3",
            applies_to=["boundary.accesses", "boundary.effects"],
            reads=["kind", "index_expression", "predicate", "fill_value"],
            api_mapping={
                "load": "tl.load",
                "store": "tl.store",
                "vector_conjunction": "&",
            },
            example="Use the declared predicate/fill for loads and guard for stores.",
            pitfalls=[
                "A store guard is independent from a load fill value.",
                "Do not strengthen a declared predicate with redundant domain facts.",
                "Use vector conjunction &, never scalar and or &&.",
            ],
        ),
        RuleCard(
            "materialize_mask_tile",
            Frontend.TRITON,
            selection_mode=RuleSelectionMode.OVERRIDE,
            override_family="mask_materialization",
            applies_to=["boundary.accesses"],
            reads=[
                "component predicates", "compound predicate",
                "predicate reuse graph", "local rank insertion",
            ],
            api_mapping={
                "row_rank_insertion": "component[:, None]",
                "column_rank_insertion": "component[None, :]",
                "vector_conjunction": "&",
            },
            example=(
                "Compute source-declared one-dimensional predicate components "
                "once, rank-insert those boolean values, and reuse the composed "
                "tile mask for every referenced access."
            ),
            pitfalls=[
                "Do not recompute component comparisons after rank insertion.",
                "Do not merge predicates whose source staging is independent.",
            ],
        ),
        RuleCard(
            "realize_elementwise",
            Frontend.TRITON,
            version="4",
            applies_to=["map_unary", "map_binary"],
            reads=["fn", "operands", "dtype"],
            api_mapping={
                "add": "+",
                "sub": "-",
                "mul": "*",
                "div": "/",
                "abs": "tl.abs",
                "exp": "tl.exp",
                "log": "tl.log",
                "sqrt": "tl.sqrt",
                "rsqrt": "tl.rsqrt",
                "recip": "1.0 / operand",
                "sigmoid": "tl.sigmoid",
                "pow": (
                    "from triton.language.extra.cuda import libdevice; "
                    "libdevice.pow(lhs, rhs)"
                ),
                "where": "tl.where",
            },
            example="Select the concrete operation only from the referenced J node.",
            pitfalls=[
                "Preserve non-commutative operand order.",
                "A J scalar input remains a runtime operand; never bake its sampled value.",
                "For pow, use only the allowlisted Triton libdevice import and "
                "keep a runtime exponent as a runtime operand.",
            ],
        ),
        RuleCard(
            "realize_const",
            Frontend.TRITON,
            applies_to=["const"],
            reads=["value", "dtype", "shape", "result"],
            api_mapping={"tensor_splat": "tl.full"},
            example=(
                "Materialize the referenced J constant in its declared dtype and "
                "local output tile shape."
            ),
            pitfalls=[
                "Read the value from J; never infer it from the kernel name or output name."
            ],
        ),
        RuleCard(
            "realize_reduce",
            Frontend.TRITON,
            version="4",
            applies_to=["reduce", "boundary.reductions"],
            reads=[
                "fn", "global axes", "tile_axis", "identity",
                "masked_contribution", "semantic extent of each global axis",
            ],
            api_mapping={
                "sum": "tl.sum",
                "max": "tl.max",
                "min": "tl.min",
                "prod": (
                    "define a @triton.jit binary multiply combiner and call "
                    "tl.reduce(tile, axis=tile_axis, combine_fn=combiner)"
                ),
                "mean": "tl.sum(tile, axis=tile_axis) / semantic_extent(global_axis)",
            },
            example=(
                "Choose sum/max/min from J, but pass boundary.reductions.tile_axis "
                "to the frontend reduction."
            ),
            pitfalls=[
                "Do not substitute a different reduction identity.",
                "Do not pass the global J tensor axis when the local tile rank differs.",
                "For mean, divide by the logical J-axis extent, never a C tile size.",
                "For prod, emit the multiply combiner in the replay module; do not "
                "replace the reduction with a Python loop.",
            ],
        ),
        RuleCard(
            "realize_flattened_reduce",
            Frontend.TRITON,
            selection_mode=RuleSelectionMode.OVERRIDE,
            override_family="reduction_call_shape",
            applies_to=["reduce", "boundary.reductions"],
            reads=[
                "fn", "global axes", "local tile rank", "identity",
                "masked_contribution", "semantic extent of each global axis",
            ],
            api_mapping={
                "sum": "tl.sum(tile)",
                "max": "tl.max(tile)",
                "min": "tl.min(tile)",
                "mean": "tl.sum(tile) / semantic_extent(global_axes)",
            },
            example=(
                "When pi selects the source's axis-omitted reduction mechanism, "
                "call the J-selected reducer without an axis argument so the "
                "frontend flattens the complete local tile."
            ),
            pitfalls=[
                "Do not use this card for a source call with an explicit tile axis.",
                "Global J axes and masked contribution remain binding semantics.",
            ],
        ),
        RuleCard(
            "realize_broadcast",
            Frontend.TRITON,
            version="2",
            applies_to=["boundary.broadcasts"],
            reads=["operand", "axes"],
            api_mapping={
                "scalar_to_tile": "use the scalar operand directly",
                "rank_insertion": "tl.expand_dims",
            },
            example=(
                "A scalar reduction result consumed by the same 1-D tile broadcasts "
                "implicitly; use tl.expand_dims only for an actual local rank insertion."
            ),
            pitfalls=[
                "Semantic broadcast axes do not automatically imply a local expand_dims."
            ],
        ),
        RuleCard(
            "realize_k_tiled_dot",
            Frontend.TRITON,
            version="3",
            applies_to=["contraction"],
            reads=[
                "subscripts", "operands", "result dtype", "output tile slots",
                "contracted K semantic extent", "K tile slot", "K tile domain",
                "lhs/rhs access expressions", "lhs/rhs masks", "lhs/rhs fills",
            ],
            api_mapping={
                "dot": "tl.dot",
                "zeros": "tl.zeros",
                "ceil_div": "tl.cdiv",
                "float32": "tl.float32",
            },
            example=(
                "Initialize one output tile in the J result dtype, iterate symbolic "
                "K-tile indices up to ceil-div(K, BLOCK_K), materialize the referenced "
                "masked lhs/rhs accesses, and accumulate dot(lhs, rhs) in operand order."
            ),
            pitfalls=[
                "Preserve lhs/rhs order and both address expressions.",
                "Use a tile-index loop with unit step; do not replace ceil-div with floor.",
                "Keep the accumulator and dot result in the declared J result dtype.",
                "Do not model schedule-only K tiling as a J reduce or loop node.",
            ],
        ),
        RuleCard(
            "realize_outer_product",
            Frontend.TRITON,
            applies_to=["contraction"],
            reads=[
                "subscripts", "operands", "result dtype",
                "lhs/rhs local ranks", "output axes",
            ],
            api_mapping={
                "lhs_rank_insertion": "lhs[:, None]",
                "rhs_rank_insertion": "rhs[None, :]",
                "combine": "*",
            },
            example=(
                "For a contraction such as m,n->mn with no contracted label, "
                "rank-insert the two vectors and multiply them without a dot or "
                "reduction."
            ),
            pitfalls=[
                "Do not introduce a K loop when no subscript label is contracted.",
                "Preserve lhs/rhs axis order from the declared subscripts.",
            ],
        ),
        RuleCard(
            "realize_indexed_access",
            Frontend.TRITON,
            applies_to=["gather"],
            reads=[
                "operands", "mode", "axis", "data-dependent index expression",
                "result dtype and shape",
            ],
            api_mapping={"indexed_load": "tl.load"},
            example=(
                "Realize the J gather through the referenced B access and its "
                "data-dependent index value."
            ),
            pitfalls=[
                "Do not replace a runtime index with its current sampled value."
            ],
        ),
        RuleCard(
            "bind_runtime_scalar",
            Frontend.TRITON,
            version="2",
            applies_to=[
                "interface.parameters", "intent.inputs", "boundary.domains"
            ],
            reads=[
                "parameter name", "dtype", "tensor_ref", "referencing domain id"
            ],
            example=(
                "Bind an extent scalar to its domain references, or bind a semantic "
                "scalar through tensor_ref to the corresponding J scalar input."
            ),
            pitfalls=[
                "Do not replace a runtime extent with a C slot.",
                "Do not replace a semantic scalar with its current specialization value.",
            ],
        ),
        RuleCard(
            "declare_schedule_slot",
            Frontend.TRITON,
            version="2",
            applies_to=["schedule.slots", "interface.parameters"],
            reads=[
                "slot name", "slot kind", "value type",
                "matching interface constexpr parameter",
            ],
            api_mapping={"constexpr_annotation": "tl.constexpr"},
            example=(
                "Declare a kind=constexpr slot through its matching interface "
                "parameter without choosing a value; keep kind=compiler slots "
                "as launch-only options."
            ),
            pitfalls=[
                "The concrete C value is injected after generation.",
                "A compiler slot is not a kernel parameter and must not appear in the body.",
            ],
        ),
        RuleCard(
            "realize_cast",
            Frontend.TRITON,
            applies_to=["cast"],
            reads=["to_dtype", "operand"],
            api_mapping={
                "float16": "tl.float16",
                "bfloat16": "tl.bfloat16",
                "float32": "tl.float32",
                "int32": "tl.int32",
            },
            example="Cast only when the referenced J node requests a dtype change.",
        ),
        RuleCard(
            "realize_select",
            Frontend.TRITON,
            applies_to=["select"],
            reads=["condition", "true operand", "false operand"],
            api_mapping={"select": "tl.where"},
            example="Preserve the J operand order in conditional value selection.",
        ),
        RuleCard(
            "realize_loop_carry",
            Frontend.TRITON,
            applies_to=["loop", "scan"],
            reads=["trip axis", "carry init", "yield refs", "schedule slot"],
            api_mapping={"ceil_div": "tl.cdiv"},
            example="Rebuild the declared recurrence while keeping C symbolic.",
            pitfalls=["A semantic carry is not a schedule-only loop variable."],
        ),
        RuleCard(
            "realize_unmasked_access",
            Frontend.TRITON,
            applies_to=["boundary.accesses", "boundary.effects"],
            reads=["kind", "index_expression"],
            api_mapping={"load": "tl.load", "store": "tl.store"},
            example="Use an unmasked access only when B declares the domain legal.",
        ),
        RuleCard(
            "realize_output_mapping",
            Frontend.TRITON,
            applies_to=["boundary.output_mappings"],
            reads=["output", "value", "node_ref"],
            example="Store exactly the J value declared for each external output.",
            pitfalls=["Do not substitute a convenient intermediate result."],
        ),
    ]
    return RuleRegistry(cards)


__all__ = [
    "RuleCard", "RuleRegistry", "RuleSelectionMode",
    "default_triton_rule_registry",
]
