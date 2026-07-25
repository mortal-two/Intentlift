from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from direct_fixtures import make_copy_artifact


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "tools/scrc_gate2_recheck.py"
_SPEC = importlib.util.spec_from_file_location(
    "intentir_scrc_gate2_recheck_tool", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
scrc_recheck = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scrc_recheck
_SPEC.loader.exec_module(scrc_recheck)


def _parent(tmp_path: Path, *, gate1_status: str = "replay_closed") -> tuple[Path, Path]:
    parent = tmp_path / "old-run" / "attempt-03"
    parent.mkdir(parents=True)
    artifact = parent / "artifact.json"
    artifact.write_text(
        json.dumps(make_copy_artifact().to_json()), encoding="utf-8"
    )
    (parent / "gate1.json").write_text(
        json.dumps({"status": gate1_status}), encoding="utf-8"
    )
    replay = "import triton\nimport triton.language as tl\n@triton.jit\ndef scrc_k0(p0, p1):\n    tl.store(p1, tl.load(p0))\n"
    (parent / "replay.py").write_text(replay, encoding="utf-8")
    (parent / "replay_receipt.json").write_text(
        json.dumps(
            {
                "function_name": "scrc_k0",
                "response_sha256": hashlib.sha256(replay.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return parent, artifact


def test_cli_reuses_closed_artifact_and_replay_without_model_calls(
    monkeypatch, tmp_path, capsys
) -> None:
    case_id = "high_performance:test_gate2_recheck"
    parent, artifact = _parent(tmp_path)
    source = object()
    prepared = SimpleNamespace(
        request=SimpleNamespace(source=SimpleNamespace(gate_source=source))
    )
    calls: dict[str, object] = {}
    monkeypatch.setitem(scrc_recheck.TRITON_WORKLOAD_ADAPTERS, case_id, object())
    monkeypatch.setattr(
        scrc_recheck,
        "prepare_triton_workload",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        scrc_recheck,
        "validate_workload_gate1_recheck_parent",
        lambda *args: calls.setdefault("validated", args),
    )
    suite = object()
    monkeypatch.setattr(
        scrc_recheck,
        "build_workload_gate2_suite",
        lambda selected, value: suite,
    )
    result = SimpleNamespace(
        passed=True,
        status_counts={"gate2_pass": 1},
        to_json=lambda: {"passed": True, "status_counts": {"gate2_pass": 1}},
    )

    class FakeRunner:
        def run_suite(self, selected_source, replay, selected_suite):
            calls["run"] = (selected_source, replay, selected_suite)
            return result

    monkeypatch.setattr(scrc_recheck, "Gate2Runner", FakeRunner)

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

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"] == scrc_recheck.GATE2_CAPABILITY_RECHECK_PROTOCOL
    assert payload["status"] == "scrc_certified"
    assert payload["model_calls"] == 0
    assert payload["reuses_exact_gate1_artifact"] is True
    assert payload["reuses_exact_gate1_replay"] is True
    assert calls["run"][0] is source
    assert calls["run"][2] is suite
    assert Path(payload["evidence_path"], "gate2.json").is_file()


def test_cli_rejects_parent_without_gate1_closure(
    monkeypatch, tmp_path, capsys
) -> None:
    parent, artifact = _parent(tmp_path, gate1_status="no_closure")
    monkeypatch.setattr(
        scrc_recheck,
        "prepare_triton_workload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must fail before workload execution")
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
    assert "requires replay_closed parent Gate1" in capsys.readouterr().err


def test_cli_rejects_replay_that_differs_from_receipt(
    monkeypatch, tmp_path, capsys
) -> None:
    parent, artifact = _parent(tmp_path)
    (parent / "replay.py").write_text("tampered\n", encoding="utf-8")
    monkeypatch.setattr(
        scrc_recheck,
        "prepare_triton_workload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must fail before workload execution")
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
    assert "differs from replay receipt" in capsys.readouterr().err
