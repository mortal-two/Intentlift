"""Small sequential batch shell around independent SCRC workload cases."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .workload_manifest import (
    SCRCWorkloadEntry,
    SourceResolutionStatus,
)


SCRC_BATCH_PROTOCOL = "intentir.scrc.sequential-batch.v1"


class BatchClassification(str, Enum):
    READY = "ready"
    MANIFEST_INVALID = "manifest_invalid"
    CASE_UNAVAILABLE = "case_unavailable"
    HARNESS_INVALID = "harness_invalid"
    LIFT_RESPONSE_REJECTED = "lift_response_rejected"
    TYPECHECK_FAILED = "typecheck_failed"
    REPLAY_RESPONSE_REJECTED = "replay_response_rejected"
    REPLAY_COMPILE_FAILED = "replay_compile_failed"
    NO_CLOSURE = "no_closure"
    REFERENCE_UNSUPPORTED = "reference_unsupported"
    GATE2_FAILED = "gate2_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    SCRC_CERTIFIED = "scrc_certified"


@dataclass(frozen=True)
class BatchCaseResult:
    case_id: str
    ordinal: int
    classification: BatchClassification
    representative_tier: str
    diagnostic: str = ""
    source_file: str | None = None
    main_status: str | None = None
    j_necessity_status: str | None = None
    evidence_path: str | None = None
    call_budget: Mapping[str, int] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ordinal": self.ordinal,
            "classification": self.classification.value,
            "representative_tier": self.representative_tier,
            "diagnostic": self.diagnostic,
            "source_file": self.source_file,
            "main_status": self.main_status,
            "j_necessity_status": self.j_necessity_status,
            "evidence_path": self.evidence_path,
            "call_budget": dict(self.call_budget),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class BatchReport:
    mode: str
    cases: tuple[BatchCaseResult, ...]

    @property
    def counts(self) -> dict[str, int]:
        result = Counter(item.classification.value for item in self.cases)
        return dict(sorted(result.items()))

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": SCRC_BATCH_PROTOCOL,
            "mode": self.mode,
            "total": len(self.cases),
            "status_counts": self.counts,
            "cases": [item.to_json() for item in self.cases],
        }


CaseExecutor = Callable[[SCRCWorkloadEntry], BatchCaseResult]


class SequentialSCRCBatchRunner:
    """Execute cases in declaration order and isolate case-local failures."""

    def run(
        self,
        entries: Sequence[SCRCWorkloadEntry],
        executor: CaseExecutor,
        *,
        mode: str,
    ) -> BatchReport:
        results: list[BatchCaseResult] = []
        for entry in entries:
            try:
                result = executor(entry)
                if result.case_id != entry.case_id or result.ordinal != entry.ordinal:
                    raise ValueError("batch executor returned a result for another case")
            except Exception as exc:  # case isolation is the point of this shell
                result = BatchCaseResult(
                    entry.case_id,
                    entry.ordinal,
                    BatchClassification.INFRASTRUCTURE_FAILED,
                    entry.representative_tier,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
        return BatchReport(mode, tuple(results))


def dry_run_executor(
    *,
    dataset_root: str | Path | None = None,
) -> CaseExecutor:
    def execute(entry: SCRCWorkloadEntry) -> BatchCaseResult:
        source = entry.resolve_source(dataset_root=dataset_root)
        classification = {
            SourceResolutionStatus.RESOLVED: BatchClassification.READY,
            SourceResolutionStatus.CASE_UNAVAILABLE: (
                BatchClassification.CASE_UNAVAILABLE
            ),
            SourceResolutionStatus.MANIFEST_INVALID: (
                BatchClassification.MANIFEST_INVALID
            ),
        }[source.status]
        return BatchCaseResult(
            entry.case_id,
            entry.ordinal,
            classification,
            entry.representative_tier,
            diagnostic=source.diagnostic,
            source_file=source.location or None,
            details={
                "source_module": entry.source_module,
                "symbol": entry.symbol,
                "frontend": entry.frontend,
                "observer_profile": entry.observer_profile,
                "q_policy": entry.q_policy,
                "adapter_readiness": entry.adapter_readiness,
            },
        )

    return execute


__all__ = [
    "BatchCaseResult",
    "BatchClassification",
    "BatchReport",
    "SCRC_BATCH_PROTOCOL",
    "SequentialSCRCBatchRunner",
    "dry_run_executor",
]
