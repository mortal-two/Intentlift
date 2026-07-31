from __future__ import annotations

import copy
import hashlib
import json

import pytest

from direct_fixtures import make_reduce_artifact
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.production_snapshot import (
    PRODUCTION_CATALOG_SCHEMA,
    PRODUCTION_SNAPSHOT_SCHEMA,
    ProductionSnapshot,
    ProductionSnapshotError,
    load_production_snapshot,
)
from intentir.verify.gate1.orchestrate import GATE1_PROTOCOL, canonical_artifact_digest
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.production_export import (
    ProductionExportError,
    append_production_export,
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


def _v13_certified_report() -> dict:
    report = _certified_report()
    accepted = report["attempts"][1]
    artifact = accepted["artifact"]
    artifact["schema"] = "intentir.scrc.artifact.v13"
    artifact["pi"]["program_axes"] = []
    artifact["pi"]["participant_mappings"] = []
    payload = {
        **artifact,
        "observer_level": 2,
        "protocol": GATE1_PROTOCOL,
    }
    accepted["gate1"]["artifact_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


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


def test_exporter_projects_v13_j_c_without_decoding_new_pi(tmp_path) -> None:
    report = _v13_certified_report()
    report_path = tmp_path / "report-v13.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    snapshot, receipt = export_production_snapshot(report_path)

    accepted = report["attempts"][1]["artifact"]
    assert snapshot.j.to_json() == accepted["j"]
    assert snapshot.c.to_json() == accepted["c"]
    assert receipt.evidence_kind == "certified_run_report"
    assert "pi" not in snapshot.to_dict()


def test_exporter_checks_certified_wire_digest_before_j_elaboration(tmp_path) -> None:
    report = _certified_report()
    accepted = report["attempts"][1]
    accepted["artifact"]["j"]["boundary"]["domains"]["d0"] = (
        "0 <= k0 and k0 < x0"
    )
    wire_artifact = Gate1Artifact.from_json(accepted["artifact"])
    accepted["gate1"]["artifact_digest"] = canonical_artifact_digest(wire_artifact)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    snapshot, _receipt = export_production_snapshot(path)

    assert snapshot.j.boundary.domains["d0"] == "(0 <= k0) and (k0 < x0)"


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


def test_certified_export_appends_content_addressed_catalog_idempotently(
    tmp_path,
) -> None:
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    (catalog_dir / "catalog.json").write_text(
        json.dumps(
            {
                "protocol": PRODUCTION_CATALOG_SCHEMA,
                "entry_count": 0,
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_certified_report()), encoding="utf-8")

    first = append_production_export(
        catalog_dir,
        name="reduce",
        case_id="high_performance:test_reduce",
        evidence=report_path,
    )
    second = append_production_export(
        catalog_dir,
        name="reduce",
        case_id="high_performance:test_reduce",
        evidence=report_path,
    )
    loaded, path = load_production_snapshot(catalog_dir, "reduce")
    catalog = json.loads((catalog_dir / "catalog.json").read_text())

    assert first[0].snapshot_digest == second[0].snapshot_digest
    assert loaded.snapshot_digest == first[0].snapshot_digest
    assert path.name == f"{loaded.snapshot_digest}.json"
    assert catalog["entry_count"] == 1


def test_catalog_append_rejects_stable_name_rebinding(tmp_path) -> None:
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    (catalog_dir / "catalog.json").write_text(
        json.dumps(
            {
                "protocol": PRODUCTION_CATALOG_SCHEMA,
                "entry_count": 0,
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_certified_report()), encoding="utf-8")
    append_production_export(
        catalog_dir,
        name="reduce",
        case_id="high_performance:test_reduce",
        evidence=report_path,
    )

    with pytest.raises(ProductionExportError, match="different content"):
        append_production_export(
            catalog_dir,
            name="reduce",
            case_id="high_performance:different_case",
            evidence=report_path,
        )
