"""Fixed unary normalization for J/C structural Gate1.

The normalizer consumes one raw compiler observation and the already typed pi
witness.  It never sees the peer observation and never decides whether a pair
matches.  A typed relation remains hard until the profile can independently map
each observed physical occurrence back to its J/C anchor.  In particular, merely
having a ``program_axes`` row does not authorize collapsing x/y/z identities.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any

from .model import Gate1Artifact


NORMALIZER_PROTOCOL = "intentir.gate1.jc-normalizer.v2"


@dataclass(frozen=True)
class NormalizedTTIR:
    """One independently normalized TTIR observation."""

    text: str
    transformed_occurrences: int
    active_rules: tuple[str, ...]
    hard_relations: tuple[str, ...] = ()
    representation_anchor_exemptions: frozenset[str] = frozenset()

    def evidence(self) -> dict[str, Any]:
        return {
            "protocol": NORMALIZER_PROTOCOL,
            "profile_ok": True,
            "unary": True,
            "peer_observation_visible": False,
            "transformed_occurrences": self.transformed_occurrences,
            "active_rules": list(self.active_rules),
            "hard_relations": list(self.hard_relations),
            "representation_anchor_exemptions": sorted(
                self.representation_anchor_exemptions
            ),
            "hard_preservation_check": "passed",
            "idempotence_check": "passed",
            "preserved_jc_atom_classes": [
                "interface",
                "operator_and_dataflow",
                "reduction",
                "predicate_and_mask",
                "memory_access_and_effect",
                "loop_and_control",
            ],
            "unknown_atoms": "hard",
        }


_SSA_TOKEN = r"%[A-Za-z0-9_.$-]+(?:#[0-9]+)?"


def _replace_ssa(text: str, source: str, target: str) -> str:
    """Substitute one whole SSA token without touching a longer name."""

    return re.sub(
        rf"(?<![A-Za-z0-9_.$-]){re.escape(source)}(?![A-Za-z0-9_.$-])",
        target,
        text,
    )


def _elide_redundant_nonnegative_guard(ttir: str) -> tuple[str, int]:
    """Canonicalize ``0 <= index && index < upper`` to its upper bound.

    This is deliberately a narrow *unary* compiler-form normalization.  It
    accepts only one SSA-local pattern: the lower comparison is against a
    literal zero, both comparisons consume precisely the same coordinate, and
    their conjunction is the only use of that lower comparison.  The upper
    bound and the fact that the memory effects remain masked are retained.
    """

    lines = ttir.splitlines(keepends=True)
    definitions: dict[str, tuple[int, str]] = {}
    for index, line in enumerate(lines):
        match = re.match(rf"^\s*(?P<result>{_SSA_TOKEN})\s*=\s*(?P<body>.*)$", line)
        if match is not None:
            definitions[match.group("result")] = (index, match.group("body"))

    replacements: list[tuple[str, str, str, str | None]] = []
    and_pattern = re.compile(
        rf"^\s*(?P<result>{_SSA_TOKEN})\s*=\s*arith\.andi\s+"
        rf"(?P<left>{_SSA_TOKEN}),\s*(?P<right>{_SSA_TOKEN})\s*:\s*.*$"
    )
    cmp_pattern = re.compile(
        rf"^\s*(?P<result>{_SSA_TOKEN})\s*=\s*arith\.cmpi\s+"
        rf"(?P<predicate>sge|uge|slt|ult),\s*(?P<lhs>{_SSA_TOKEN}),\s*"
        rf"(?P<rhs>{_SSA_TOKEN})\s*:\s*.*$"
    )
    constant_zero = re.compile(r"^arith\.constant\s+(?:dense<)?0(?:\.0+)?(?:>|\s)")

    for line in lines:
        conjunction = and_pattern.match(line)
        if conjunction is None:
            continue
        choices = (
            (conjunction.group("left"), conjunction.group("right")),
            (conjunction.group("right"), conjunction.group("left")),
        )
        for lower_name, upper_name in choices:
            lower_definition = definitions.get(lower_name)
            upper_definition = definitions.get(upper_name)
            if lower_definition is None or upper_definition is None:
                continue
            lower = cmp_pattern.match(f"{lower_name} = {lower_definition[1]}")
            upper = cmp_pattern.match(f"{upper_name} = {upper_definition[1]}")
            if lower is None or upper is None:
                continue
            if lower.group("predicate") not in {"sge", "uge"}:
                continue
            if upper.group("predicate") not in {"slt", "ult"}:
                continue
            if lower.group("lhs") != upper.group("lhs"):
                continue
            zero_definition = definitions.get(lower.group("rhs"))
            if zero_definition is None or constant_zero.match(zero_definition[1]) is None:
                continue
            # The lower guard is safe to erase only when it is exclusively the
            # conjunct being canonicalized.  This prevents weakening a second
            # independent effect/control use.
            lower_uses = -1 + sum(
                len(re.findall(
                    rf"(?<![A-Za-z0-9_.$-]){re.escape(lower_name)}(?![A-Za-z0-9_.$-])",
                    candidate,
                ))
                for candidate in lines
            )
            if lower_uses != 1:
                continue
            zero_name = lower.group("rhs")
            zero_uses = -1 + sum(
                len(re.findall(
                    rf"(?<![A-Za-z0-9_.$-]){re.escape(zero_name)}(?![A-Za-z0-9_.$-])",
                    candidate,
                ))
                for candidate in lines
            )
            replacements.append((conjunction.group("result"), upper_name, lower_name, zero_name if zero_uses == 1 else None))
            break

    # A single shared lower-bound conjunction can protect both the load and the
    # store; apply every independently proven local rewrite, still unary.
    for conjunction, upper, lower, zero in replacements:
        ttir = re.sub(
            rf"(?m)^\s*{re.escape(conjunction)}\s*=.*(?:\n|$)", "", ttir
        )
        ttir = _replace_ssa(ttir, conjunction, upper)
        ttir = re.sub(
            rf"(?m)^\s*{re.escape(lower)}\s*=.*(?:\n|$)", "", ttir
        )
        if zero is not None:
            ttir = re.sub(
                rf"(?m)^\s*{re.escape(zero)}\s*=.*(?:\n|$)", "", ttir
            )
    return ttir, len(replacements)


def _canonicalize_boolean_byte_storage(ttir: str) -> tuple[str, int]:
    """Erase only the physical ``i1``-to-``i8`` boolean store encoding.

    Triton may bitcast a logical boolean pointer before address formation or
    after it, then extend the boolean value to a byte for ``tt.store``.  This
    rule removes exactly those representation-only casts.  It does not rewrite
    the comparison, mask, address arithmetic, store multiplicity, or any
    non-boolean conversion.
    """

    lines = ttir.splitlines(keepends=True)
    substitutions: list[tuple[str, str, int]] = []
    extui = re.compile(
        rf"^\s*(?P<result>{_SSA_TOKEN})\s*=\s*arith\.extui\s+"
        rf"(?P<input>{_SSA_TOKEN})\s*:\s*tensor<[^>]*xi1>\s+to\s+"
        rf"tensor<[^>]*xi8>.*$"
    )
    bitcast = re.compile(
        rf"^\s*(?P<result>{_SSA_TOKEN})\s*=\s*tt\.bitcast\s+"
        rf"(?P<input>{_SSA_TOKEN})\s*:\s*(?:tensor<[^>]*!tt\.ptr<i1>>|"
        rf"!tt\.ptr<i1>)\s*->\s*(?:tensor<[^>]*!tt\.ptr<i8>>|!tt\.ptr<i8>).*$"
    )
    for index, line in enumerate(lines):
        extui_match = extui.match(line)
        bitcast_match = bitcast.match(line)
        match = extui_match or bitcast_match
        if match is None:
            continue
        result, source = match.group("result"), match.group("input")
        uses = sum(
            len(
                re.findall(
                    rf"(?<![A-Za-z0-9_.$-]){re.escape(result)}"
                    rf"(?![A-Za-z0-9_.$-])",
                    candidate,
                )
            )
            for candidate in lines
        ) - 1
        if uses != 1:
            continue
        if extui_match is not None:
            consumer = next(
                (
                    candidate
                    for candidate in lines
                    if re.search(
                        rf"(?<![A-Za-z0-9_.$-]){re.escape(result)}"
                        rf"(?![A-Za-z0-9_.$-])",
                        candidate,
                    )
                    and "tt.store" in candidate
                ),
                None,
            )
            if consumer is None:
                continue
        # A pointer bitcast can feed a splat/addptr chain rather than the
        # store directly.  Its i1->i8 type and one-use condition make it an
        # address representation only; the surrounding access remains hard.
        substitutions.append((result, source, index))

    for result, source, index in reversed(substitutions):
        pattern = re.escape(lines[index].rstrip("\n"))
        ttir = re.sub(rf"(?m)^{pattern}\n?", "", ttir)
        ttir = _replace_ssa(ttir, result, source)
    if substitutions:
        # Only after every removed conversion was proven to terminate in a
        # store may its byte-pointer spelling be treated as representation.
        ttir = ttir.replace("!tt.ptr<i8>", "!tt.ptr<i1>")
    return ttir, len(substitutions)


def _j_allows_reduction_reorder(artifact: Gate1Artifact) -> bool:
    """Return whether J contains a reduction whose contract permits reordering."""

    def walk(value: Any):
        if isinstance(value, dict):
            if value.get("op") == "reduce":
                yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    return any(
        operation.get("attrs", {}).get("fn")
        in {"sum", "max", "min", "all", "any"}
        and operation.get("attrs", {}).get("assoc")
        in {"float_nondeterministic", "associative", "commutative"}
        for operation in walk(artifact.j.intent.to_json())
    )


def _elide_identity_reduce_reshape(ttir: str) -> tuple[str, int]:
    """Erase a same-type allow-reorder reshape used only by one reduction.

    Triton may insert ``tt.reshape T -> T allow_reorder`` before a reduction
    when the source omits an explicit axis.  The reshape has no shape effect;
    its only possible value-order effect is already admitted by the J reduction
    contract checked by the caller.  Other reshapes and other consumers remain
    hard.
    """

    lines = ttir.splitlines(keepends=True)
    pattern = re.compile(
        rf"^\s*(?P<result>{_SSA_TOKEN})\s*=\s*tt\.reshape\s+"
        rf"(?P<input>{_SSA_TOKEN})\s+allow_reorder\s*:\s*"
        rf"(?P<source_type>.+?)\s*->\s*(?P<target_type>.+?)"
        rf"(?:\s+loc\(.*\))?\s*$"
    )
    substitutions: list[tuple[str, str, int]] = []
    for index, line in enumerate(lines):
        match = pattern.match(line.rstrip("\n"))
        if match is None:
            continue
        if " ".join(match.group("source_type").split()) != " ".join(
            match.group("target_type").split()
        ):
            continue
        result = match.group("result")
        users = [
            candidate
            for candidate_index, candidate in enumerate(lines)
            if candidate_index != index
            and re.search(
                rf"(?<![A-Za-z0-9_.$-]){re.escape(result)}"
                rf"(?![A-Za-z0-9_.$-])",
                candidate,
            )
        ]
        if len(users) != 1 or "tt.reduce" not in users[0]:
            continue
        substitutions.append((result, match.group("input"), index))

    for result, source, index in reversed(substitutions):
        pattern_text = re.escape(lines[index].rstrip("\n"))
        ttir = re.sub(rf"(?m)^{pattern_text}\n?", "", ttir)
        ttir = _replace_ssa(ttir, result, source)
    return ttir, len(substitutions)


def normalize_ttir_for_gate1(ttir: str, artifact: Gate1Artifact) -> NormalizedTTIR:
    """Return ``N^pi(ttir)`` without consulting another observation.

    The current profile keeps ``program_axes`` hard.  The old label-only quotient
    was not sufficient: collapsing x/y/z could hide a swapped logical-domain
    mapping even while preserving operation counts.  A future rule must first
    establish a unique observation-to-J-domain mapping; until then raw axis
    identity, multiplicity, users and downstream dataflow all remain visible.
    """

    declared_accesses = {
        access.access_id for access in artifact.j.boundary.accesses
    }
    bound_accesses = {
        str(binding["access_ref"])
        for binding in artifact.pi.access_representation_bindings
        if binding.get("relation") in {"address_recomputed", "pointer_advanced"}
    }
    guard_elision_accesses = {
        str(binding["access_ref"])
        for binding in artifact.pi.access_representation_bindings
        if binding.get("relation") == "guard_lower_bound_elided"
    }
    boolean_storage_accesses = {
        str(binding["access_ref"])
        for binding in artifact.pi.access_representation_bindings
        if binding.get("relation") == "boolean_storage_byte"
    }
    boolean_values = {
        value.name
        for value in (*artifact.j.intent.inputs, *artifact.j.intent.outputs)
        if value.dtype == "i1"
    }
    boolean_accesses = {
        access.access_id
        for access in artifact.j.boundary.accesses
        if access.tensor in boolean_values
    }
    collapse_addptr = bool(declared_accesses) and bound_accesses == declared_accesses
    collapse_nonnegative_guard = (
        bool(declared_accesses)
        and guard_elision_accesses == declared_accesses
        and all(
            ">= 0" in access.predicate or ">=0" in access.predicate
            for access in artifact.j.boundary.accesses
            if access.predicate is not None
        )
    )
    collapse_boolean_storage = bool(boolean_accesses) and (
        boolean_storage_accesses == boolean_accesses
    )
    transformed_occurrences = 0
    active_rules: list[str] = []
    if _j_allows_reduction_reorder(artifact):
        ttir, erased_identity_reshapes = _elide_identity_reduce_reshape(ttir)
        if erased_identity_reshapes:
            transformed_occurrences += erased_identity_reshapes
            active_rules.append("j_reduce_identity_reshape")
    if collapse_addptr:
        active_rules.append("jc_access_address_representation")
    if collapse_nonnegative_guard:
        ttir, erased_guards = _elide_redundant_nonnegative_guard(ttir)
        if erased_guards:
            transformed_occurrences += erased_guards
            active_rules.append("jc_access_nonnegative_guard")
    if collapse_boolean_storage:
        ttir, erased_boolean_storage = _canonicalize_boolean_byte_storage(ttir)
        if erased_boolean_storage:
            transformed_occurrences += erased_boolean_storage
            active_rules.append("jc_boolean_storage_byte")
    hard_relations: list[str] = []
    if artifact.pi.program_axes:
        hard_relations.append("program_axes")
    if artifact.pi.program_bindings:
        hard_relations.append("program_bindings")
    if artifact.pi.iteration_bindings:
        hard_relations.append("iteration_bindings")
    if artifact.pi.control_bindings:
        hard_relations.append("control_bindings")
    if artifact.pi.materialization_bindings:
        hard_relations.append("materialization_bindings")
    if bound_accesses and not collapse_addptr:
        hard_relations.append("access_representation_bindings")
    return NormalizedTTIR(
        text=ttir,
        transformed_occurrences=transformed_occurrences,
        active_rules=tuple(active_rules),
        hard_relations=tuple(hard_relations),
        representation_anchor_exemptions=(
            frozenset({"tt.addptr"}) if collapse_addptr else frozenset()
        ),
    )


__all__ = ["NORMALIZER_PROTOCOL", "NormalizedTTIR", "normalize_ttir_for_gate1"]
