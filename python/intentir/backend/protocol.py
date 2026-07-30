"""Gate-neutral backend request accepting canonical J and separate pure C."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from intentir.ir.canonical_j import elaborate_j
from intentir.verify.gate1.model import ImplementationConfig, SemanticArchive


class BackendProtocolError(ValueError):
    pass


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:+-]+", value) is None:
        raise BackendProtocolError(f"{context} must be a non-empty identifier")
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
    j: SemanticArchive
    target: BackendTarget
    variant: str
    c: ImplementationConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            raise BackendProtocolError("lowering request requires canonical J")
        object.__setattr__(self, "j", elaborate_j(self.j))
        if not isinstance(self.target, BackendTarget):
            raise BackendProtocolError("lowering request requires BackendTarget")
        object.__setattr__(self, "variant", _identifier(self.variant, "lowering variant"))
        if self.c is not None and not isinstance(self.c, ImplementationConfig):
            raise BackendProtocolError("C must use the canonical pure schedule schema")

class ComparisonKind(str, Enum):
    SOURCE_TO_LOWERING = "source_to_lowering"
    REPLAY_TO_LOWERING = "replay_to_lowering"
    INTENT_TO_LOWERING = "intent_to_lowering"
    ORACLE_TO_LOWERING = "oracle_to_lowering"


@dataclass(frozen=True)
class ComparisonEvidence:
    kind: ComparisonKind
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ComparisonKind):
            object.__setattr__(self, "kind", ComparisonKind(self.kind))
        if not isinstance(self.passed, bool):
            raise BackendProtocolError("comparison passed must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BackendConformanceRecord:
    target: BackendTarget
    variant: str
    boundary_consumption: Mapping[str, Any]
    test_domain: Mapping[str, Any]
    comparisons: tuple[ComparisonEvidence, ...]

    def __post_init__(self) -> None:
        if not self.comparisons:
            raise BackendProtocolError("backend conformance requires comparison evidence")
        if not bool(self.boundary_consumption.get("supported", False)):
            raise BackendProtocolError("backend did not consume complete Layer B")

    def assert_conformant(self) -> None:
        failed = [item.kind.value for item in self.comparisons if not item.passed]
        if failed:
            raise BackendProtocolError("backend comparison failed: " + ", ".join(failed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "variant": self.variant,
            "boundary_consumption": dict(self.boundary_consumption),
            "test_domain": dict(self.test_domain),
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


__all__ = [
    "BackendConformanceRecord", "BackendProtocolError", "BackendTarget",
    "ComparisonEvidence", "ComparisonKind", "LoweringRequest",
]
