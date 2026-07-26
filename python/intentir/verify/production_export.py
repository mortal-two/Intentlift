"""Acceptance-side export of an exact SCRC artifact as a J+C snapshot.

This module is intentionally outside the backend.  It may inspect certification
evidence, but its output type cannot represent that evidence and therefore
enforces the production visibility boundary mechanically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.production_snapshot import ProductionSnapshot, content_digest
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate1.orchestrate import canonical_artifact_digest


EXPORT_RECEIPT_SCHEMA = "intentir.production.snapshot-export.v1"


class ProductionExportError(ValueError):
    """Raised when evidence is not sufficient for a production export."""


def _read_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionExportError(f"cannot read {context} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProductionExportError(f"{context} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProductionExportError(f"cannot hash evidence {path}: {exc}") from exc
    return digest.hexdigest()


def _artifact(value: Mapping[str, Any]) -> Gate1Artifact:
    try:
        return canonicalize_gate1_artifact(Gate1Artifact.from_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ProductionExportError(f"certified artifact is invalid: {exc}") from exc


def _accepted_report_artifact(report: Mapping[str, Any]) -> Gate1Artifact:
    if report.get("main_status") != "SCRC_CERTIFIED":
        raise ProductionExportError("run report is not SCRC_CERTIFIED")
    if not isinstance(report.get("accepted_attempt"), int):
        raise ProductionExportError("certified report has no accepted attempt")
    gate2 = report.get("gate2")
    if not isinstance(gate2, Mapping) or gate2.get("passed") is not True:
        raise ProductionExportError("certified report has no passing Gate2 suite")
    if gate2.get("failed_case_ids") not in (None, []):
        raise ProductionExportError("certified report still names failed Gate2 cases")
    attempts = report.get("attempts")
    if not isinstance(attempts, list):
        raise ProductionExportError("certified report attempts must be an array")
    accepted = [
        item
        for item in attempts
        if isinstance(item, Mapping)
        and item.get("formal_round") == report["accepted_attempt"]
    ]
    if len(accepted) != 1:
        raise ProductionExportError(
            "certified report accepted attempt is missing or ambiguous"
        )
    attempt = accepted[0]
    if attempt.get("typecheck_status") != "pass":
        raise ProductionExportError("accepted attempt did not pass TypeCheck")
    gate1 = attempt.get("gate1")
    if not isinstance(gate1, Mapping) or gate1.get("status") != "replay_closed":
        raise ProductionExportError("accepted attempt is not Gate1 replay_closed")
    artifact_payload = attempt.get("artifact")
    if not isinstance(artifact_payload, Mapping):
        raise ProductionExportError("accepted attempt has no embedded artifact")
    artifact = _artifact(artifact_payload)
    if gate1.get("artifact_digest") != canonical_artifact_digest(artifact):
        raise ProductionExportError("accepted Gate1 artifact digest mismatch")
    return artifact


def _recheck_artifact(
    evidence_path: Path,
    recheck: Mapping[str, Any],
) -> Gate1Artifact:
    if recheck.get("status") != "scrc_certified" or recheck.get("gate2_passed") is not True:
        raise ProductionExportError("Gate2 recheck is not scrc_certified")
    for field_name in (
        "reuses_exact_gate1_artifact",
        "reuses_exact_gate1_replay",
    ):
        if recheck.get(field_name) is not True:
            raise ProductionExportError(f"Gate2 recheck does not assert {field_name}")
    if recheck.get("model_calls") != 0:
        raise ProductionExportError("Gate2 recheck must be a zero-model-call reuse")
    raw_artifact_path = recheck.get("artifact")
    raw_parent = recheck.get("parent_evidence")
    if not isinstance(raw_artifact_path, str) or not isinstance(raw_parent, str):
        raise ProductionExportError("Gate2 recheck lacks exact parent evidence paths")
    artifact_path = Path(raw_artifact_path)
    if not artifact_path.is_absolute():
        artifact_path = evidence_path.parent / artifact_path
    parent = Path(raw_parent)
    if not parent.is_absolute():
        parent = evidence_path.parent / parent
    artifact_payload = _read_object(artifact_path, context="Gate1 artifact")
    artifact = _artifact(artifact_payload)
    gate1 = _read_object(parent / "gate1.json", context="parent Gate1 evidence")
    if gate1.get("status") != "replay_closed":
        raise ProductionExportError("Gate2 recheck parent is not replay_closed")
    if gate1.get("artifact_digest") != canonical_artifact_digest(artifact):
        raise ProductionExportError("Gate2 recheck parent artifact digest mismatch")
    return artifact


@dataclass(frozen=True)
class ProductionExportReceipt:
    evidence_sha256: str
    canonical_artifact_digest: str
    j_digest: str
    c_digest: str | None
    snapshot_digest: str
    evidence_kind: str
    schema_version: str = EXPORT_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_sha256": self.evidence_sha256,
            "canonical_artifact_digest": self.canonical_artifact_digest,
            "j_digest": self.j_digest,
            "c_digest": self.c_digest,
            "snapshot_digest": self.snapshot_digest,
            "evidence_kind": self.evidence_kind,
        }

    @property
    def receipt_digest(self) -> str:
        return content_digest(self.to_dict())


def export_production_snapshot(
    evidence: str | Path,
) -> tuple[ProductionSnapshot, ProductionExportReceipt]:
    """Validate one run report/recheck and return its exact J+C snapshot."""

    path = Path(evidence).resolve()
    payload = _read_object(path, context="SCRC evidence")
    if "main_status" in payload:
        artifact = _accepted_report_artifact(payload)
        evidence_kind = "certified_run_report"
    elif "changes_parent_verdict" in payload and "parent_evidence" in payload:
        artifact = _recheck_artifact(path, payload)
        evidence_kind = "zero_model_gate2_recheck"
    else:
        raise ProductionExportError("unsupported SCRC evidence envelope")
    snapshot = ProductionSnapshot(artifact.j, artifact.c)
    receipt = ProductionExportReceipt(
        evidence_sha256=_sha256_file(path),
        canonical_artifact_digest=canonical_artifact_digest(artifact),
        j_digest=snapshot.j_digest,
        c_digest=snapshot.c_digest,
        snapshot_digest=snapshot.snapshot_digest,
        evidence_kind=evidence_kind,
    )
    return snapshot, receipt


__all__ = [
    "EXPORT_RECEIPT_SCHEMA",
    "ProductionExportError",
    "ProductionExportReceipt",
    "export_production_snapshot",
]
