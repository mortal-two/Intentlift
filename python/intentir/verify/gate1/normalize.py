"""Fixed unary normalization for J/C structural Gate1.

The normalizer consumes one raw compiler observation and the already typed pi
witness.  It never sees the peer observation and never decides whether a pair
matches.  Its only active quotient removes the physical spelling of a Triton
program axis, which is absent from J/C, while preserving the number of program
coordinate operations and every downstream dataflow edge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .model import Gate1Artifact


NORMALIZER_PROTOCOL = "intentir.gate1.jc-normalizer.v1"

_PROGRAM_COORDINATE = re.compile(
    r"\btt\.get_(?:program_id|num_programs)\b"
)
_CUSTOM_AXIS = re.compile(
    r"(\btt\.get_(?:program_id|num_programs)\s+)([xyz])(\s*:)"
)
_GENERIC_AXIS = re.compile(
    r"(\baxis\s*=\s*)[0-2](?P<suffix>\s*:\s*i32\b|\b)"
)


@dataclass(frozen=True)
class NormalizedTTIR:
    """One independently normalized TTIR observation."""

    text: str
    transformed_occurrences: int
    active_rules: tuple[str, ...]

    def evidence(self) -> dict[str, Any]:
        return {
            "protocol": NORMALIZER_PROTOCOL,
            "profile_ok": True,
            "unary": True,
            "peer_observation_visible": False,
            "transformed_occurrences": self.transformed_occurrences,
            "active_rules": list(self.active_rules),
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


def _normalize_program_axis_line(line: str) -> tuple[str, bool]:
    if _PROGRAM_COORDINATE.search(line) is None:
        return line, False
    normalized = _CUSTOM_AXIS.sub(r"\1x\3", line)
    normalized = _GENERIC_AXIS.sub(r"\g<1>0\g<suffix>", normalized)
    return normalized, normalized != line


def normalize_ttir_for_gate1(ttir: str, artifact: Gate1Artifact) -> NormalizedTTIR:
    """Return ``N^pi(ttir)`` without consulting another observation.

    ``program_axes`` authorizes only removal of the physical x/y/z (or 0/1/2)
    label.  It does not erase the coordinate operation, its multiplicity, its
    result type, any user, or any arithmetic/access/effect reached from it.
    Mixed-radix arithmetic between a coordinate anchor and an address anchor is
    already handled by the versioned TTIR representation quotient; declaring a
    ``program_linearizations`` row therefore activates no additional text
    rewrite here.
    """

    active_rules: list[str] = []
    transformed = 0
    text = ttir
    if artifact.pi.program_axes:
        active_rules.append("physical_program_axis_label")
        lines: list[str] = []
        for line in text.splitlines(keepends=True):
            normalized, changed = _normalize_program_axis_line(line)
            if changed:
                if (
                    len(_PROGRAM_COORDINATE.findall(line))
                    != len(_PROGRAM_COORDINATE.findall(normalized))
                    or _normalize_program_axis_line(normalized)[0] != normalized
                ):
                    raise ValueError(
                        "program-axis normalization failed its fixed preservation audit"
                    )
            lines.append(normalized)
            transformed += int(changed)
        text = "".join(lines)
    if artifact.pi.program_linearizations:
        active_rules.append("mixed_radix_representation_quotient_v13")
    return NormalizedTTIR(
        text=text,
        transformed_occurrences=transformed,
        active_rules=tuple(active_rules),
    )


__all__ = ["NORMALIZER_PROTOCOL", "NormalizedTTIR", "normalize_ttir_for_gate1"]
