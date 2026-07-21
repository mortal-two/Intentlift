from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_verify_package_exports_only_active_gates() -> None:
    import intentir.verify as verify

    assert verify.__all__ == ["gate1", "gate2"]


def test_wrong_control_plane_is_outside_importable_python_tree() -> None:
    gate1 = ROOT / "python/intentir/verify/gate1"
    retired_names = [
        "com" + "poser.py",
        "rules.py",
        "rule_" + "release.py",
        "interventions.py",
        "pi_" + "ablations.py",
        "program_" + "adapter.py",
    ]
    assert all(not (gate1 / name).exists() for name in retired_names)
    archived = ROOT / "archive/2026-07-21-wrong-control-plane"
    assert archived.is_dir()


def test_duplicate_semantic_wire_is_not_importable() -> None:
    ir = ROOT / "python/intentir/ir"
    assert not (ir / "program.py").exists()
    assert not (ir / "abi.py").exists()
    assert not (ir / "contracts.py").exists()


def test_retired_live_drivers_are_archived() -> None:
    assert not (ROOT / "test/scrc_live_e2e.py").exists()
    assert not (ROOT / "test/scrc_heldout_batch.py").exists()
    archive = ROOT / "archive/2026-07-21-wrong-control-plane/test"
    assert (archive / "scrc_live_e2e.py").is_file()
    assert (archive / "scrc_heldout_batch.py").is_file()
