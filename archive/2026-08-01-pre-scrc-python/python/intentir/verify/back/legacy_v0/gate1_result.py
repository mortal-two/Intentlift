"""Archived pre-v8 Gate1 result schema.

This model records the removed replay-repair counter and fuzzy masked-J
leakage ratio.  It remains executable only for historical artifact readers;
the active Gate1 result is ``Gate1ExecutionResult`` in ``verify.gate1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ...gate1.model import (
    Gate1Artifact,
    Gate1Status,
    SchemaError,
    StrictJSONMixin,
    _enum,
    _integer,
    _strict_object,
    _string_list,
)


@dataclass
class Gate1Result(StrictJSONMixin):
    """Legacy result kept for reading old repair/ratio-era reports."""

    status: Gate1Status
    artifact: Gate1Artifact | None = None
    diagnostics: list[str] = field(default_factory=list)
    relift_attempts: int = 0
    repair_attempts: int = 0
    leakage_ratio: float | None = None

    def __post_init__(self) -> None:
        self.status = _enum(Gate1Status, self.status, "Gate1Result.status")
        if self.artifact is not None and not isinstance(
            self.artifact, Gate1Artifact
        ):
            self.artifact = Gate1Artifact.from_json(self.artifact)
        self.diagnostics = _string_list(
            self.diagnostics, "Gate1Result.diagnostics"
        )
        self.relift_attempts = _integer(
            self.relift_attempts,
            "Gate1Result.relift_attempts",
            minimum=0,
        )
        self.repair_attempts = _integer(
            self.repair_attempts,
            "Gate1Result.repair_attempts",
            minimum=0,
        )
        if self.leakage_ratio is not None:
            if isinstance(self.leakage_ratio, bool) or not isinstance(
                self.leakage_ratio, (int, float)
            ):
                raise SchemaError(
                    "Gate1Result.leakage_ratio must be a finite number"
                )
            self.leakage_ratio = float(self.leakage_ratio)
            if not math.isfinite(self.leakage_ratio) or self.leakage_ratio < 0:
                raise SchemaError(
                    "Gate1Result.leakage_ratio must be a finite "
                    "non-negative number"
                )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Gate1Result":
        data = _strict_object(
            value,
            path="Gate1Result",
            allowed={
                "status",
                "artifact",
                "diagnostics",
                "relift_attempts",
                "repair_attempts",
                "leakage_ratio",
            },
            required={"status"},
        )
        return cls(
            status=data["status"],
            artifact=(
                Gate1Artifact.from_json(data["artifact"])
                if data.get("artifact") is not None
                else None
            ),
            diagnostics=_string_list(
                data.get("diagnostics", []), "Gate1Result.diagnostics"
            ),
            relift_attempts=data.get("relift_attempts", 0),
            repair_attempts=data.get("repair_attempts", 0),
            leakage_ratio=data.get("leakage_ratio"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status.value,
            "diagnostics": list(self.diagnostics),
            "relift_attempts": self.relift_attempts,
            "repair_attempts": self.repair_attempts,
        }
        if self.artifact is not None:
            result["artifact"] = self.artifact.to_json()
        if self.leakage_ratio is not None:
            result["leakage_ratio"] = self.leakage_ratio
        return result


__all__ = ["Gate1Result"]
