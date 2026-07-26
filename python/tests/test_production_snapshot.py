from __future__ import annotations

import copy
import json

import pytest

from direct_fixtures import make_reduce_artifact
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.production_snapshot import (
    PRODUCTION_SNAPSHOT_SCHEMA,
    ProductionSnapshot,
    ProductionSnapshotError,
)
from intentir.verify.gate1.orchestrate import canonical_artifact_digest
from intentir.verify.production_export import (
    ProductionExportError,
    export_production_snapshot,
)


def _certified_report() -> dict:
    artifact = canonicalize_gate1_artifact(make_reduce_artifact())
    digest = canonical_artifact_digest(artifact)
    return {
        "main_status": "SCRC_CERTIFIED",
        "accepted_attempt": 2,
        "attempts": [
            {
                "formal_round": 1,
                "artifact": artifact.to_json(),
                "typecheck_status": "pass",
                "gate1": {"status": "no_closure", "artifact_digest": digest},
            },
            {
                "formal_round": 2,
                "artifact": artifact.to_json(),
                "typecheck_status": "pass",
                "gate1": {"status": "replay_closed", "artifact_digest": digest},
            },
        ],
        "gate2": {"passed": True, "failed_case_ids": []},
        "j_necessity_status": None,
        "call_budget": {},
    }


def test_snapshot_round_trip_contains_only_j_c_and_content_identity() -> None:
    artifact = canonicalize_gate1_artifact(make_reduce_artifact())
    snapshot = ProductionSnapshot(artifact.j, artifact.c)

    payload = snapshot.to_dict()
    assert set(payload) == {
        "schema_version",
        "j",
        "c",
        "j_digest",
        "c_digest",
        "snapshot_digest",
    }
    assert payload["schema_version"] == PRODUCTION_SNAPSHOT_SCHEMA
    restored = ProductionSnapshot.from_json(snapshot.to_json())
    assert restored.to_dict() == payload


@pytest.mark.parametrize("forbidden", ["pi", "source", "replay", "gate2", "q"])
def test_snapshot_rejects_non_backend_channels(forbidden: str) -> None:
    artifact = canonicalize_gate1_artifact(make_reduce_artifact())
    payload = ProductionSnapshot(artifact.j, artifact.c).to_dict()
    payload[forbidden] = {}

    with pytest.raises(ProductionSnapshotError, match="forbidden fields"):
        ProductionSnapshot.from_dict(payload)


def test_snapshot_rejects_tampered_j_or_digest() -> None:
    artifact = canonicalize_gate1_artifact(make_reduce_artifact())
    payload = ProductionSnapshot(artifact.j, artifact.c).to_dict()
    payload["j_digest"] = "0" * 64
    with pytest.raises(ProductionSnapshotError, match="J digest mismatch"):
        ProductionSnapshot.from_dict(payload)

    payload = ProductionSnapshot(artifact.j, artifact.c).to_dict()
    payload["j"]["kernel_id"] = "tampered"
    with pytest.raises(ProductionSnapshotError, match="digest mismatch"):
        ProductionSnapshot.from_dict(payload)


def test_exporter_selects_exact_accepted_attempt_and_strips_evidence(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_certified_report()), encoding="utf-8")

    snapshot, receipt = export_production_snapshot(report_path)

    accepted = _certified_report()["attempts"][1]["artifact"]
    assert snapshot.j.to_json() == accepted["j"]
    assert snapshot.c.to_json() == accepted["c"]
    assert receipt.evidence_kind == "certified_run_report"
    assert receipt.snapshot_digest == snapshot.snapshot_digest
    assert "pi" not in snapshot.to_dict()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(main_status="GATE2_FAILED"), "not SCRC"),
        (lambda value: value["gate2"].update(passed=False), "passing Gate2"),
        (
            lambda value: value["attempts"][1]["gate1"].update(status="no_closure"),
            "not Gate1",
        ),
        (
            lambda value: value["attempts"][1]["gate1"].update(
                artifact_digest="0" * 64
            ),
            "digest mismatch",
        ),
    ],
)
def test_exporter_fails_closed_on_incomplete_or_misaligned_evidence(
    tmp_path, mutation, message
) -> None:
    report = copy.deepcopy(_certified_report())
    mutation(report)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ProductionExportError, match=message):
        export_production_snapshot(path)
