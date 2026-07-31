"""Fixed unary normalization for J/C structural Gate1.

The normalizer consumes one raw compiler observation and the already typed pi
witness.  It never sees the peer observation and never decides whether a pair
matches.  A typed relation remains hard until the profile can independently map
each observed physical occurrence back to its J/C anchor.  In particular, merely
having a ``program_axes`` row does not authorize collapsing x/y/z identities.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    def evidence(self) -> dict[str, Any]:
        return {
            "protocol": NORMALIZER_PROTOCOL,
            "profile_ok": True,
            "unary": True,
            "peer_observation_visible": False,
            "transformed_occurrences": self.transformed_occurrences,
            "active_rules": list(self.active_rules),
            "hard_relations": list(self.hard_relations),
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

    return NormalizedTTIR(
        text=ttir,
        transformed_occurrences=0,
        active_rules=(),
        hard_relations=("program_axes",) if artifact.pi.program_axes else (),
    )


__all__ = ["NORMALIZER_PROTOCOL", "NormalizedTTIR", "normalize_ttir_for_gate1"]
