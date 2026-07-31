"""Exact, read-only textual view of pi for one source-blind Replay episode.

The view only renders the already-authored closed relations in declaration order.
It does not join them into target statements, infer a lowering, choose a spelling,
or add a realization fact.  ``pi.json`` remains the exact authoritative payload.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping


REPLAY_REALIZATION_VIEW_PROTOCOL = "intentir.replay-realization-view.v1"


def _compact(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class ReplayRealizationView:
    pi: dict[str, Any]
    text: str

    def to_json(self) -> dict[str, Any]:
        return copy.deepcopy(self.pi)


def materialize_replay_realization(
    pi_value: Mapping[str, Any],
) -> ReplayRealizationView:
    """Render every exact pi field without interpreting or rewriting it."""

    pi = copy.deepcopy(dict(pi_value))
    lines = [
        f"realization_view {REPLAY_REALIZATION_VIEW_PROTOCOL}",
        (
            "authority exact_read_only_pi; no_J_semantics; no_C_value; no_Source; "
            "no_target_recipe; no_generation; no_repair; no_Gate"
        ),
    ]
    for section, value in pi.items():
        lines.append("")
        lines.append(f"{section}:")
        if isinstance(value, list):
            for index, item in enumerate(value):
                lines.append(f"  {section}[{index}] " + _compact(item))
        elif isinstance(value, Mapping):
            for name, item in value.items():
                lines.append(f"  {section}.{name} " + _compact(item))
        else:
            lines.append("  " + _compact(value))
    return ReplayRealizationView(pi, "\n".join(lines) + "\n")


__all__ = [
    "REPLAY_REALIZATION_VIEW_PROTOCOL",
    "ReplayRealizationView",
    "materialize_replay_realization",
]
