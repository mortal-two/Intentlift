"""Frontend rule cards and deterministic plan-to-card retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .model import (
    Frontend,
    ReplayPlan,
    SchemaError,
    StrictJSONMixin,
    _enum,
    _strict_object,
    _string,
    _string_list,
)


@dataclass
class RuleCard(StrictJSONMixin):
    """A system-owned implementation card selected by ``ReplayStep.rule``.

    Concrete frontend API spellings are allowed here and only here.  The
    lift-produced replay plan contains just the card identifier and references
    back into J.
    """

    id: str
    frontend: Frontend
    version: str = "1"
    applies_to: list[str] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)
    api_mapping: dict[str, str] = field(default_factory=dict)
    example: str = ""
    pitfalls: list[str] = field(default_factory=list)

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

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RuleCard":
        data = _strict_object(
            value,
            path="RuleCard",
            allowed={
                "id", "frontend", "version", "applies_to", "reads",
                "api_mapping", "example", "pitfalls",
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
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "frontend": self.frontend.value,
            "version": self.version,
            "applies_to": list(self.applies_to),
            "reads": list(self.reads),
            "api_mapping": dict(self.api_mapping),
            "example": self.example,
            "pitfalls": list(self.pitfalls),
        }


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
        """Resolve cards in first-use order, deduplicating repeated steps."""

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

    def rewriter_view(
        self,
        plan: ReplayPlan,
        *,
        frontend: Frontend | str | None = None,
    ) -> dict[str, Any]:
        cards = self.resolve(plan, frontend=frontend)
        return {
            "selection": [
                {
                    "step": index,
                    "rule": step.rule,
                    **({"node_ref": step.node_ref} if step.node_ref is not None else {}),
                    **({"refs": list(step.refs)} if step.refs else {}),
                }
                for index, step in enumerate(plan.steps)
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
    """Return the reviewed Triton MVP cards used by the five-kernel probe.

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
            "realize_elementwise",
            Frontend.TRITON,
            version="3",
            applies_to=["map_unary", "map_binary", "cast", "select"],
            reads=["fn", "operands", "dtype"],
            api_mapping={
                "add": "+",
                "sub": "-",
                "mul": "*",
                "div": "/",
                "exp": "tl.exp",
                "log": "tl.log",
                "sqrt": "tl.sqrt",
                "rsqrt": "tl.rsqrt",
                "recip": "1.0 / operand",
                "sigmoid": "tl.sigmoid",
                "where": "tl.where",
            },
            example="Select the concrete operation only from the referenced J node.",
            pitfalls=[
                "Preserve non-commutative operand order.",
                "A J scalar input remains a runtime operand; never bake its sampled value.",
            ],
        ),
        RuleCard(
            "realize_reduce",
            Frontend.TRITON,
            version="3",
            applies_to=["reduce", "boundary.reductions"],
            reads=[
                "fn", "global axes", "tile_axis", "identity",
                "masked_contribution", "semantic extent of each global axis",
            ],
            api_mapping={
                "sum": "tl.sum",
                "max": "tl.max",
                "min": "tl.min",
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
            applies_to=["contraction", "loop"],
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
            applies_to=["schedule.slots", "interface.parameters"],
            reads=["slot name", "slot kind", "value type"],
            api_mapping={"constexpr_annotation": "tl.constexpr"},
            example="Declare the visible constexpr slot without choosing its value.",
            pitfalls=["The concrete C value is injected after generation."],
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


__all__ = ["RuleCard", "RuleRegistry", "default_triton_rule_registry"]
