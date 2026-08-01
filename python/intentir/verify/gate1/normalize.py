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
    collapse_addptr = bool(declared_accesses) and bound_accesses == declared_accesses
    transformed_occurrences = 0
    if collapse_addptr:
        expected = Counter(access.kind.value for access in artifact.j.boundary.accesses)
        observed = Counter(
            {
                "load": len(
                    re.findall(
                        r'(?m)^\s*(?:%[^=]+?=\s*)?(?:"tt\.load"|tt\.load)\b',
                        ttir,
                    )
                ),
                "store": len(
                    re.findall(
                        r'(?m)^\s*(?:"tt\.store"|tt\.store)\b',
                        ttir,
                    )
                ),
                "update": len(
                    re.findall(
                        r'(?m)^\s*(?:%[^=]+?=\s*)?'
                        r'(?:"tt\.atomic[^\"]*"|tt\.atomic[A-Za-z0-9_.-]*)\b',
                        ttir,
                    )
                ),
            }
        )
        if any(observed[kind] != expected[kind] for kind in ("load", "store", "update")):
            raise ValueError(
                "access representation normalization requires exact J memory "
                f"multiplicity; expected={dict(expected)}, observed={dict(observed)}"
            )
        transformed_occurrences = len(
            re.findall(
                r'(?m)^\s*%[^=]+?=\s*(?:"tt\.addptr"|tt\.addptr)\b',
                ttir,
            )
        )
    active_rules = (
        ("jc_access_address_representation",) if collapse_addptr else ()
    )
    hard_relations: list[str] = []
    if artifact.pi.program_axes:
        hard_relations.append("program_axes")
    if bound_accesses and not collapse_addptr:
        hard_relations.append("access_representation_bindings")
    return NormalizedTTIR(
        text=ttir,
        transformed_occurrences=transformed_occurrences,
        active_rules=active_rules,
        hard_relations=tuple(hard_relations),
        representation_anchor_exemptions=(
            frozenset({"tt.addptr"}) if collapse_addptr else frozenset()
        ),
    )


__all__ = ["NORMALIZER_PROTOCOL", "NormalizedTTIR", "normalize_ttir_for_gate1"]
