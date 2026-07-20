"""Verification-neutral protocol shared by target backends.

This module describes a lowering request and its conformance receipt.  It does
not authorize lowering from Gate status: callers supply an ``IntentProgram``;
the backend reports which typed Layer-B obligations it consumed or checked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from intentir.ir.contracts import ObligationLedger, ObligationStatus
from intentir.ir.program import IntentProgram, ScheduleHint


class BackendProtocolError(ValueError):
    """Raised when a lowering request or conformance receipt is inconsistent."""


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:+-]+", value) is None:
        raise BackendProtocolError(f"{context} must be a non-empty identifier")
    return value


def _digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise BackendProtocolError(f"{context} must be a SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class BackendTarget:
    family: str
    architecture: str
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _identifier(self.family, "target family"))
        object.__setattr__(self, "architecture", _identifier(self.architecture, "target architecture"))
        features = tuple(_identifier(item, "target feature") for item in self.features)
        if len(features) != len(set(features)):
            raise BackendProtocolError("target features must be unique")
        object.__setattr__(self, "features", features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "architecture": self.architecture,
            "features": list(self.features),
        }


@dataclass(frozen=True)
class LoweringRequest:
    program: IntentProgram
    target: BackendTarget
    variant: str
    retuned_schedule: ScheduleHint | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.program, IntentProgram):
            raise BackendProtocolError("lowering request requires IntentProgram")
        if not isinstance(self.target, BackendTarget):
            raise BackendProtocolError("lowering request requires BackendTarget")
        object.__setattr__(self, "variant", _identifier(self.variant, "lowering variant"))
        if self.retuned_schedule is not None and not isinstance(self.retuned_schedule, ScheduleHint):
            raise BackendProtocolError("retuned_schedule must be ScheduleHint")
        effective = self.effective_program
        if effective.j_digest != self.program.j_digest:
            raise BackendProtocolError("retuning Layer C changed J")
        if effective.contract_digest != self.program.contract_digest:
            raise BackendProtocolError("retuning Layer C changed the logical ABI")

    @property
    def effective_program(self) -> IntentProgram:
        return (
            self.program
            if self.retuned_schedule is None
            else self.program.with_schedule_hint(self.retuned_schedule)
        )


class ComparisonKind(str, Enum):
    SOURCE_TO_LOWERING = "source_to_lowering"
    REPLAY_TO_LOWERING = "replay_to_lowering"
    INTENT_TO_LOWERING = "intent_to_lowering"
    ORACLE_TO_LOWERING = "oracle_to_lowering"


@dataclass(frozen=True)
class ComparisonEvidence:
    kind: ComparisonKind
    passed: bool
    evidence_digest: str

    def __post_init__(self) -> None:
        try:
            kind = self.kind if isinstance(self.kind, ComparisonKind) else ComparisonKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise BackendProtocolError(f"unknown comparison kind {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.passed, bool):
            raise BackendProtocolError("comparison passed must be boolean")
        object.__setattr__(self, "evidence_digest", _digest(self.evidence_digest, "comparison evidence digest"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "passed": self.passed,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True)
class BackendConformanceRecord:
    j_digest: str
    contract_digest: str
    program_digest: str
    target: BackendTarget
    variant: str
    obligations: ObligationLedger
    test_domain: Mapping[str, Any]
    comparisons: tuple[ComparisonEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "j_digest", _digest(self.j_digest, "conformance J digest"))
        object.__setattr__(self, "contract_digest", _digest(self.contract_digest, "conformance contract digest"))
        object.__setattr__(self, "program_digest", _digest(self.program_digest, "conformance program digest"))
        if not isinstance(self.target, BackendTarget):
            raise BackendProtocolError("conformance target must be BackendTarget")
        object.__setattr__(self, "variant", _identifier(self.variant, "conformance variant"))
        if not isinstance(self.obligations, ObligationLedger):
            raise BackendProtocolError("conformance obligations must be ObligationLedger")
        if self.obligations.contract_digest != self.contract_digest:
            raise BackendProtocolError("obligation ledger is bound to another contract")
        if not isinstance(self.test_domain, Mapping) or not all(
            isinstance(key, str) for key in self.test_domain
        ):
            raise BackendProtocolError("test_domain must be a JSON object")
        comparisons = tuple(
            item if isinstance(item, ComparisonEvidence) else ComparisonEvidence(**item)
            for item in self.comparisons
        )
        if not comparisons:
            raise BackendProtocolError(
                "backend conformance requires source, replay, intent, or "
                "oracle comparison evidence"
            )
        object.__setattr__(self, "comparisons", comparisons)

    @property
    def consumed_obligations(self) -> tuple[str, ...]:
        return tuple(
            item.obligation_id for item in self.obligations.results
            if item.status in {ObligationStatus.CONSUMED, ObligationStatus.CHECKED}
        )

    @property
    def unsupported_obligations(self) -> tuple[str, ...]:
        return self.obligations.unsupported

    def assert_conformant(self) -> None:
        self.obligations.assert_supported()
        failed = [item.kind.value for item in self.comparisons if not item.passed]
        if failed:
            raise BackendProtocolError(
                "backend comparison evidence failed: " + ", ".join(failed)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "j_digest": self.j_digest,
            "contract_digest": self.contract_digest,
            "program_digest": self.program_digest,
            "target": self.target.to_dict(),
            "variant": self.variant,
            "consumed_obligations": list(self.consumed_obligations),
            "unsupported_obligations": list(self.unsupported_obligations),
            "obligations": self.obligations.to_dict(),
            "test_domain": dict(self.test_domain),
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


__all__ = [
    "BackendConformanceRecord", "BackendProtocolError", "BackendTarget",
    "ComparisonEvidence", "ComparisonKind", "LoweringRequest",
]
