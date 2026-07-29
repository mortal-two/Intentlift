from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from direct_fixtures import make_copy_artifact
from intentir.extraction.providers.provider import ProviderConfig


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "tools/scrc_gate1_recheck.py"
_SPEC = importlib.util.spec_from_file_location(
    "intentir_scrc_gate1_recheck_tool", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
scrc_recheck = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scrc_recheck
_SPEC.loader.exec_module(scrc_recheck)


def _artifact_under(parent: Path) -> Path:
    parent.mkdir(parents=True)
    path = parent / "artifact.json"
    path.write_text(
        json.dumps(make_copy_artifact().to_json()), encoding="utf-8"
    )
    return path


def _install_success_fakes(monkeypatch, tmp_path: Path, case_id: str):
    provider = ProviderConfig(
        "test-provider", "https://provider.invalid/v1", "test-key", "test-model"
    )
    prepared = SimpleNamespace(
        entry=SimpleNamespace(case_id=case_id),
        source_location=f"module://{case_id}",
    )
    calls: dict[str, object] = {}

    monkeypatch.setitem(scrc_recheck.TRITON_WORKLOAD_ADAPTERS, case_id, object())
    monkeypatch.setattr(scrc_recheck, "get_provider", lambda _name: provider)
    monkeypatch.setattr(
        scrc_recheck,
        "prepare_triton_workload",
        lambda _adapter, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        scrc_recheck,
        "validate_workload_gate1_recheck_parent",
        lambda parent, selected, artifact: calls.setdefault(
            "validated", (parent, selected, artifact)
        ),
    )

    def allocate(root, selected_case):
        calls["root"] = Path(root)
        calls["case"] = selected_case
        path = Path(root) / "case" / "recheck-001"
        path.mkdir(parents=True)
        return path

    monkeypatch.setattr(scrc_recheck, "next_gate1_recheck_dir", allocate)

    def initialize(path, selected, **kwargs):
        calls["initialized"] = (path, selected, kwargs)

    monkeypatch.setattr(
        scrc_recheck,
        "initialize_workload_gate1_recheck_evidence",
        initialize,
    )

    result = SimpleNamespace(
        protocol="intentir.scrc.gate1-diagnostic-recheck.v1",
        diagnostic_only=True,
        artifact_digest="artifact-digest",
        typecheck=SimpleNamespace(status=SimpleNamespace(value="pass")),
        gate1=SimpleNamespace(status=SimpleNamespace(value="replay_closed")),
        passed=True,
    )

    def recheck(selected, artifact, **kwargs):
        calls["recheck"] = (selected, artifact, kwargs)
        return result

    monkeypatch.setattr(scrc_recheck, "recheck_prepared_workload_gate1", recheck)
    monkeypatch.setattr(
        scrc_recheck,
        "save_workload_gate1_recheck",
        lambda path, selected, value: calls.setdefault(
            "saved", (path, selected, value)
        ),
    )
    return calls, result


def test_cli_rechecks_one_fixed_artifact_in_new_diagnostic_evidence(
    monkeypatch, tmp_path, capsys
) -> None:
    case_id = "high_performance:test_fixed_recheck"
    parent = tmp_path / "old-run" / "attempt-03"
    artifact = _artifact_under(parent)
    calls, result = _install_success_fakes(monkeypatch, tmp_path, case_id)

    exit_code = scrc_recheck.main(
        [
            "--case",
            case_id,
            "--artifact",
            str(artifact),
            "--parent-evidence",
            str(parent),
            "--run-root",
            str(tmp_path / "rechecks"),
            "--provider",
            "test-provider",
            "--model",
            "test-model-override",
            "--effort",
            "xhigh",
            "--replay-backend",
            "miniswe",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"] == result.protocol
    assert payload["diagnostic_only"] is True
    assert payload["enters_gate2"] is False
    assert payload["changes_prior_verdict"] is False
    assert payload["status"] == "replay_closed"
    assert payload["artifact_digest"] == "artifact-digest"
    assert payload["replay_backend"] == "miniswe"
    assert calls["case"] == case_id
    assert calls["recheck"][2]["provider"].model == "test-model-override"
    assert calls["validated"][0] == parent.resolve()
    assert calls["recheck"][2]["effort"] == "xhigh"
    assert calls["recheck"][2]["replay_backend"] == "miniswe"
    assert calls["initialized"][2]["parent_evidence"] == parent.resolve()
    assert calls["initialized"][2]["replay_backend"] == "miniswe"
    assert calls["saved"][2] is result


def test_cli_rejects_artifact_outside_parent_without_calling_provider(
    monkeypatch, tmp_path, capsys
) -> None:
    parent = tmp_path / "old-run" / "attempt-03"
    parent.mkdir(parents=True)
    artifact = tmp_path / "elsewhere" / "artifact.json"
    artifact.parent.mkdir()
    artifact.write_text(
        json.dumps(make_copy_artifact().to_json()), encoding="utf-8"
    )
    monkeypatch.setattr(
        scrc_recheck,
        "get_provider",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("provider must not be called")
        ),
    )

    exit_code = scrc_recheck.main(
        [
            "--case",
            "high_performance:any",
            "--artifact",
            str(artifact),
            "--parent-evidence",
            str(parent),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "artifact must be inside parent evidence" in captured.err


def test_cli_rejects_parent_lineage_before_provider_call(
    monkeypatch, tmp_path, capsys
) -> None:
    case_id = "high_performance:test_wrong_lineage"
    parent = tmp_path / "old-run" / "attempt-03"
    artifact = _artifact_under(parent)
    monkeypatch.setitem(scrc_recheck.TRITON_WORKLOAD_ADAPTERS, case_id, object())
    monkeypatch.setattr(
        scrc_recheck,
        "prepare_triton_workload",
        lambda _adapter, **_kwargs: SimpleNamespace(
            entry=SimpleNamespace(case_id=case_id)
        ),
    )
    monkeypatch.setattr(
        scrc_recheck,
        "validate_workload_gate1_recheck_parent",
        lambda *_args: (_ for _ in ()).throw(
            ValueError("parent Gate1 request differs from selected workload")
        ),
    )
    monkeypatch.setattr(
        scrc_recheck,
        "get_provider",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("provider must not be called")
        ),
    )

    exit_code = scrc_recheck.main(
        [
            "--case",
            case_id,
            "--artifact",
            str(artifact),
            "--parent-evidence",
            str(parent),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "parent Gate1 request differs" in captured.err


def test_cli_returns_one_for_completed_typecheck_rejection(
    monkeypatch, tmp_path, capsys
) -> None:
    case_id = "high_performance:test_fixed_recheck"
    parent = tmp_path / "old-run" / "attempt-03"
    artifact = _artifact_under(parent)
    calls, result = _install_success_fakes(monkeypatch, tmp_path, case_id)
    result.passed = False
    result.typecheck = SimpleNamespace(status=SimpleNamespace(value="fail"))
    result.gate1 = None

    exit_code = scrc_recheck.main(
        [
            "--case",
            case_id,
            "--artifact",
            str(artifact),
            "--parent-evidence",
            str(parent),
            "--run-root",
            str(tmp_path / "rechecks"),
            "--json",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "typecheck_failed"
    assert payload["diagnostic_only"] is True
    assert calls["saved"][2] is result
