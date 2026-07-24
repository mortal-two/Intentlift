from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier, Lock
from types import SimpleNamespace

from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.scrc_batch import BatchClassification
from intentir.verify.scrc_runner import CallBudget, MainStatus


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "tools/scrc_batch.py"
_SPEC = importlib.util.spec_from_file_location("intentir_scrc_batch_tool", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
scrc_batch = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scrc_batch
_SPEC.loader.exec_module(scrc_batch)


def test_batch_cli_lists_six_representatives_without_provider(capsys) -> None:
    assert scrc_batch.main(["--list", "--tier", "representative", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 6
    assert len(payload["entries"]) == 6


def test_batch_cli_validates_current_28_source_manifest(capsys) -> None:
    assert scrc_batch.main(["--validate-manifest", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"]
    assert payload["total"] == 28
    assert payload["resolved"] == 28


def test_batch_cli_dry_run_writes_machine_readable_report(tmp_path) -> None:
    output = tmp_path / "representative.json"

    assert scrc_batch.main(
        [
            "--dry-run",
            "--tier",
            "representative",
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text())
    assert payload["mode"] == "dry_run"
    assert payload["total"] == 6
    assert payload["status_counts"] == {"ready": 6}
    assert all(
        case["source_file"].startswith(("module://", "dataset://"))
        for case in payload["cases"]
    )


def test_batch_cli_runs_from_repo_without_pythonpath() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--list", "--tier", "representative"],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "high_performance:liger_dyt" in result.stdout


def test_batch_cli_rejects_unknown_tier_and_filtered_validation(capsys) -> None:
    assert scrc_batch.main(["--list", "--tier", "not-a-tier"]) == 2
    assert "unknown SCRC workload tier" in capsys.readouterr().err

    assert scrc_batch.main(
        ["--validate-manifest", "--tier", "representative"]
    ) == 2
    assert "always validates the complete 28-item projection" in capsys.readouterr().err


def test_batch_cli_never_implicitly_runs_all_28(capsys) -> None:
    assert scrc_batch.main(["--run"]) == 2
    assert "does not implicitly run all 28" in capsys.readouterr().err


def test_batch_cli_candidate_fanout_requires_one_live_case(capsys) -> None:
    assert scrc_batch.main(["--list", "--candidates", "5"]) == 2
    assert "only valid with --run" in capsys.readouterr().err

    assert scrc_batch.main(
        ["--run", "--tier", "representative", "--candidates", "5"]
    ) == 2
    assert "one explicit --case" in capsys.readouterr().err


def test_batch_cli_candidate_width_fails_closed(capsys) -> None:
    assert scrc_batch.main(
        [
            "--run",
            "--case",
            "high_performance:liger_dyt",
            "--candidates",
            "0",
        ]
    ) == 2
    assert "must be in [1, 32]" in capsys.readouterr().err


def test_batch_cli_fans_out_one_case_and_runs_gate2_only_for_winner(
    tmp_path, monkeypatch, capsys
) -> None:
    import intentir.extraction.providers.provider as provider_module
    import intentir.verify.scrc_workload_run as workload_module
    import intentir.verify.workload_adapters as adapters_module

    provider = ProviderConfig(
        "test-provider", "https://provider.invalid/v1", "test-key", "test-model"
    )
    prepared = SimpleNamespace(source_location="module://liger_kernel.ops.dyt")
    barrier = Barrier(5, timeout=2.0)
    counter_lock = Lock()
    calls = {
        "search": 0,
        "gate2": 0,
        "saved_search": 0,
        "initialized_efforts": [],
    }

    class SearchReport:
        closed = True
        superseded = False
        accepted_attempt = 1
        attempts = ()
        stop_reason = None
        winner_branch_id = None
        stop_observed_at = None
        call_budget = CallBudget(1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0)

    def fake_search(
        _prepared, *, provider, effort, run_masked_j, lift_backend,
        stop_requested=None,
    ):
        assert provider is not None
        assert effort == "xhigh"
        assert run_masked_j
        assert lift_backend == "responses"
        assert stop_requested is None
        barrier.wait()
        with counter_lock:
            calls["search"] += 1
        return SimpleNamespace(report=SearchReport())

    def fake_complete(_prepared, _search):
        assert calls["search"] == 5
        calls["gate2"] += 1
        report = SimpleNamespace(
            main_status=MainStatus.SCRC_CERTIFIED,
            attempts=(),
            accepted_attempt=1,
            gate2=SimpleNamespace(status_counts={"pass": 1}),
            j_necessity_status=None,
            call_budget=CallBudget(1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1),
        )
        return SimpleNamespace(report=report)

    def fake_initialize(
        run_dir,
        _prepared,
        *,
        provider,
        effort,
        run_masked_j,
        lift_backend,
        search_metadata,
    ):
        assert run_dir.is_dir()
        assert provider.name == "test-provider"
        assert run_masked_j
        assert lift_backend == "responses"
        assert search_metadata["candidate_count"] == 5
        assert search_metadata["protocol"] == (
            "intentir.scrc.candidate-fanout.v3"
        )
        assert search_metadata["selection_rule"] == (
            "wave_barrier_branch_id_formal_round.v3"
        )
        calls["initialized_efforts"].append(effort)

    def fake_next_fanout_dir(root, _case_id):
        path = Path(root) / "liger_dyt" / "fanout-001"
        path.mkdir(parents=True)
        return path

    monkeypatch.setattr(provider_module, "get_provider", lambda _name: provider)
    monkeypatch.setattr(adapters_module, "prepare_triton_workload", lambda *_a, **_k: prepared)
    monkeypatch.setattr(workload_module, "next_candidate_fanout_dir", fake_next_fanout_dir)
    monkeypatch.setattr(workload_module, "initialize_workload_evidence", fake_initialize)
    monkeypatch.setattr(workload_module, "search_prepared_workload_gate1", fake_search)
    monkeypatch.setattr(workload_module, "complete_prepared_workload_gate2", fake_complete)
    monkeypatch.setattr(workload_module, "scrc_gate1_search_diagnostic", lambda _r: "")
    monkeypatch.setattr(workload_module, "scrc_attempt_stage_counts", lambda _r: {})
    monkeypatch.setattr(workload_module, "save_workload_gate1_search", lambda *_a, **_k: calls.__setitem__("saved_search", calls["saved_search"] + 1))
    monkeypatch.setattr(workload_module, "save_workload_execution", lambda *_a, **_k: None)
    monkeypatch.setattr(workload_module, "save_candidate_fanout_report", lambda *_a, **_k: None)
    monkeypatch.setattr(workload_module, "classify_scrc_report", lambda _r: BatchClassification.SCRC_CERTIFIED)
    monkeypatch.setattr(workload_module, "scrc_report_diagnostic", lambda _r: "")

    result = scrc_batch.main(
        [
            "--run",
            "--case",
            "high_performance:liger_dyt",
            "--candidates",
            "5",
            "--effort",
            "xhigh",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] == 5
    assert payload["winner_branch_id"] == 1
    assert payload["winner_formal_round"] == 1
    assert payload["status_counts"] == {
        "gate1_closed": 1,
        "gate1_closed_not_selected": 4,
    }
    assert payload["winner_result"]["main_status"] == "SCRC_CERTIFIED"
    assert calls == {
        "search": 5,
        "gate2": 1,
        "saved_search": 4,
        "initialized_efforts": ["xhigh"] * 5,
    }
