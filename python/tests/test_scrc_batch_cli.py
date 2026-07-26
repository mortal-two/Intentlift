from __future__ import annotations

import importlib.util
from io import StringIO
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


_SEARCH_BUDGET = CallBudget(1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0)
_CERTIFIED_BUDGET = CallBudget(1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1)
_PROGRESS_FIELDS = {
    "protocol",
    "event",
    "case_id",
    "ordinal",
    "branch_id",
    "stage",
    "attempt",
    "formal_round",
    "elapsed",
    "status",
}


def _progress_events(raw: str) -> list[dict]:
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _closed_search():
    report = SimpleNamespace(
        closed=True,
        superseded=False,
        accepted_attempt=1,
        attempts=(),
        stop_reason=None,
        winner_branch_id=None,
        stop_observed_at=None,
        j_necessity_status=None,
        call_budget=_SEARCH_BUDGET,
    )
    return SimpleNamespace(report=report, lift=object(), replay_writers=())


def _no_closure_search():
    report = SimpleNamespace(
        closed=False,
        superseded=False,
        accepted_attempt=None,
        attempts=(),
        stop_reason=None,
        winner_branch_id=None,
        stop_observed_at=None,
        j_necessity_status=None,
        call_budget=_SEARCH_BUDGET,
    )
    return SimpleNamespace(report=report, lift=object(), replay_writers=())


def _certified_execution():
    report = SimpleNamespace(
        main_status=MainStatus.SCRC_CERTIFIED,
        attempts=(),
        accepted_attempt=1,
        gate2=SimpleNamespace(status_counts={"pass": 1}),
        j_necessity_status=None,
        call_budget=_CERTIFIED_BUDGET,
    )
    return SimpleNamespace(report=report)


def test_batch_cli_has_independent_replay_backend_switch() -> None:
    selected = scrc_batch._parser().parse_args(
        [
            "--list",
            "--lift-backend",
            "miniswe",
            "--lift-workflow",
            "staged",
            "--replay-backend",
            "miniswe",
        ]
    )
    defaults = scrc_batch._parser().parse_args(["--list"])

    assert selected.lift_backend == "miniswe"
    assert selected.lift_workflow == "staged"
    assert selected.replay_backend == "miniswe"
    assert defaults.lift_workflow == "single"
    assert defaults.replay_backend == "responses"


def test_batch_cli_accepts_an_explicit_provider_config_path(tmp_path) -> None:
    path = tmp_path / "providers.json"
    selected = scrc_batch._parser().parse_args(
        ["--list", "--provider-config", str(path)]
    )

    assert selected.provider_config == path


def _install_live_batch_fakes(monkeypatch, *, search, complete):
    import intentir.extraction.providers.provider as provider_module
    import intentir.verify.scrc_workload_run as workload_module
    import intentir.verify.workload_adapters as adapters_module
    import intentir.verify.workload_registry as registry_module

    provider = ProviderConfig(
        "test-provider", "https://provider.invalid/v1", "test-key", "test-model"
    )
    entries = scrc_batch.build_workload_manifest().select(tier="representative")
    for entry in entries:
        monkeypatch.setitem(
            adapters_module.TRITON_WORKLOAD_ADAPTERS,
            entry.case_id,
            entry.case_id,
        )

    monkeypatch.setattr(
        provider_module,
        "get_provider",
        lambda _name, _path=None: provider,
    )
    monkeypatch.setattr(
        registry_module,
        "prepare_registered_scrc_workload",
        lambda case_id, **_kwargs: SimpleNamespace(
            case_id=case_id,
            source_location=f"module://{case_id}",
        ),
    )
    monkeypatch.setattr(
        workload_module,
        "initialize_workload_evidence",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(workload_module, "search_prepared_workload_gate1", search)
    monkeypatch.setattr(workload_module, "complete_prepared_workload_gate2", complete)
    monkeypatch.setattr(
        workload_module,
        "save_workload_execution",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        workload_module,
        "save_workload_failure",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        workload_module,
        "save_workload_gate1_search",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        workload_module,
        "save_candidate_fanout_report",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        workload_module,
        "classify_scrc_report",
        lambda _r: BatchClassification.SCRC_CERTIFIED,
    )
    monkeypatch.setattr(workload_module, "scrc_report_diagnostic", lambda _r: "")
    monkeypatch.setattr(
        workload_module,
        "scrc_gate1_search_diagnostic",
        lambda _r: "",
    )
    monkeypatch.setattr(workload_module, "scrc_attempt_stage_counts", lambda _r: {})
    return entries


def test_batch_cli_lists_six_representatives_without_provider(capsys) -> None:
    assert scrc_batch.main(["--list", "--tier", "representative", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 6
    assert len(payload["entries"]) == 6


def test_batch_cli_repeated_cases_form_registry_ordered_explicit_set(
    capsys,
) -> None:
    entries = scrc_batch.build_workload_manifest().entries[:2]

    assert scrc_batch.main(
        [
            "--list",
            "--case",
            entries[1].case_id,
            "--case",
            entries[0].case_id,
            "--json",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [entry["case_id"] for entry in payload["entries"]] == [
        entries[0].case_id,
        entries[1].case_id,
    ]


def test_batch_cli_lists_frontend_cases_from_the_neutral_registry(capsys) -> None:
    assert scrc_batch.main(
        ["--list", "--case", "tilelang:add2d", "--json"]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"] == "intentir.scrc.registered-workloads.v1"
    assert payload["total"] == 1
    assert payload["entries"][0]["case_id"] == "tilelang:add2d"
    assert payload["entries"][0]["frontend"] == "tilelang"
    assert payload["entries"][0]["representative_tier"] == "frontend.tilelang"
    assert payload["entries"][0]["ordinal"] > 28


def test_batch_cli_frontend_tiers_select_registered_native_cases(capsys) -> None:
    assert scrc_batch.main(
        ["--list", "--tier", "frontend", "--json"]
    ) == 0
    both = json.loads(capsys.readouterr().out)
    assert {entry["frontend"] for entry in both["entries"]} == {
        "tilelang",
        "cuda",
    }

    assert scrc_batch.main(
        ["--list", "--tier", "frontend.cuda", "--json"]
    ) == 0
    cuda = json.loads(capsys.readouterr().out)
    assert [entry["case_id"] for entry in cuda["entries"]] == [
        "cuda:tilelang_add2d"
    ]


def test_batch_cli_rejects_duplicate_or_mixed_explicit_case_selection(
    capsys,
) -> None:
    case_id = scrc_batch.build_workload_manifest().entries[0].case_id

    assert scrc_batch.main(
        ["--list", "--case", case_id, "--case", case_id]
    ) == 2
    assert "duplicate --case" in capsys.readouterr().err

    assert scrc_batch.main(
        ["--list", "--case", case_id, "--tier", "representative"]
    ) == 2
    assert "mutually exclusive" in capsys.readouterr().err


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


def test_non_live_cli_modes_never_emit_progress(capsys) -> None:
    commands = (
        ["--list", "--tier", "representative", "--json"],
        ["--validate-manifest", "--json"],
        ["--dry-run", "--tier", "representative", "--json"],
    )
    for command in commands:
        assert scrc_batch.main(command) == 0
        captured = capsys.readouterr()
        json.loads(captured.out)
        assert captured.err == ""


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


def test_batch_cli_candidate_fanout_requires_live_run(capsys) -> None:
    assert scrc_batch.main(["--list", "--candidates", "5"]) == 2
    assert "only valid with --run" in capsys.readouterr().err


def test_batch_cli_defaults_live_evidence_outside_repository() -> None:
    args = scrc_batch._parser().parse_args(
        ["--run", "--case", "high_performance:liger_dyt"]
    )

    assert args.run_root == Path("/tmp/intentir-scrc-task11")
    assert _ROOT not in args.run_root.parents


def test_batch_cli_candidate_width_fails_closed(capsys) -> None:
    for candidates in ("0", "6"):
        assert scrc_batch.main(
            [
                "--run",
                "--case",
                "high_performance:liger_dyt",
                "--candidates",
                candidates,
            ]
        ) == 2
        assert "must be in [1, 5]" in capsys.readouterr().err


def test_batch_cli_replay_width_fails_closed(capsys) -> None:
    for width in ("0", "5"):
        assert scrc_batch.main(
            [
                "--run",
                "--case",
                "high_performance:liger_dyt",
                "--replay-width",
                width,
            ]
        ) == 2
        assert "must be in [1, 4]" in capsys.readouterr().err


def test_batch_cli_gate1_round_budget_is_positive_and_bounded(capsys) -> None:
    for rounds in ("0", "13"):
        assert scrc_batch.main(
            [
                "--run",
                "--case",
                "high_performance:liger_dyt",
                "--gate1-rounds",
                rounds,
            ]
        ) == 2
        assert "must be in [1, 12]" in capsys.readouterr().err


def test_batch_cli_pre_replay_rejection_budget_is_bounded(capsys) -> None:
    for rejections in ("-1", "13"):
        assert scrc_batch.main(
            [
                "--run",
                "--case",
                "high_performance:liger_dyt",
                "--pre-replay-rejections",
                rejections,
            ]
        ) == 2
        assert "must be in [0, 12]" in capsys.readouterr().err


def test_batch_cli_parallel_controls_are_live_only_and_globally_bounded(
    tmp_path, capsys
) -> None:
    assert scrc_batch.main(
        ["--list", "--tier", "representative", "--case-workers", "2"]
    ) == 2
    assert "only valid with --run" in capsys.readouterr().err

    assert scrc_batch.main(
        [
            "--run",
            "--tier",
            "representative",
            "--case-workers",
            "3",
            "--candidates",
            "3",
            "--max-branch-workers",
            "8",
            "--run-root",
            str(tmp_path / "runs"),
        ]
    ) == 2
    assert (
        "case_workers * candidates * replay_width <= max_branch_workers"
        in capsys.readouterr().err
    )

    assert scrc_batch.main(
        [
            "--run",
            "--tier",
            "representative",
            "--case-workers",
            "2",
            "--gate2-workers",
            "3",
            "--run-root",
            str(tmp_path / "runs"),
        ]
    ) == 2
    assert "must be <= --case-workers" in capsys.readouterr().err


def test_batch_cli_fans_out_one_case_and_runs_gate2_only_for_winner(
    tmp_path, monkeypatch, capsys
) -> None:
    import intentir.extraction.providers.provider as provider_module
    import intentir.verify.scrc_workload_run as workload_module
    import intentir.verify.workload_registry as registry_module

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
        lift_workflow, replay_backend, replay_width, max_gate1_rounds,
        max_pre_replay_rejections,
        stop_requested=None, progress_callback=None,
    ):
        assert provider is not None
        assert effort == "xhigh"
        assert run_masked_j
        assert lift_backend == "responses"
        assert lift_workflow == "single"
        assert replay_backend == "miniswe"
        assert replay_width == 2
        assert max_gate1_rounds == 6
        assert max_pre_replay_rejections == 8
        assert stop_requested is None
        assert progress_callback is not None
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
        lift_workflow,
        replay_backend,
        search_metadata,
    ):
        assert run_dir.is_dir()
        assert provider.name == "test-provider"
        assert run_masked_j
        assert lift_backend == "responses"
        assert lift_workflow == "single"
        assert replay_backend == "miniswe"
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

    monkeypatch.setattr(
        provider_module,
        "get_provider",
        lambda _name, _path=None: provider,
    )
    monkeypatch.setattr(
        registry_module,
        "prepare_registered_scrc_workload",
        lambda *_a, **_k: prepared,
    )
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
            "--replay-backend",
            "miniswe",
            "--replay-width",
            "2",
            "--max-branch-workers",
            "10",
            "--gate1-rounds",
            "6",
            "--pre-replay-rejections",
            "8",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"] == "intentir.scrc.batch.v3"
    assert payload["scheduling"] == {
        "case_workers": 1,
        "candidates": 5,
        "max_branch_workers": 10,
        "gate2_workers": 1,
        "gate1_rounds": 6,
        "pre_replay_rejections": 8,
    }
    assert payload["status_counts"] == {"scrc_certified": 1}
    assert len(payload["cases"]) == 1
    result_payload = payload["cases"][0]
    details = result_payload["details"]
    assert details["candidate_count"] == 5
    assert details["winner_branch_id"] == 1
    assert details["winner_formal_round"] == 1
    assert details["candidate_status_counts"] == {
        "gate1_closed": 1,
        "gate1_closed_not_selected": 4,
    }
    assert result_payload["main_status"] == "SCRC_CERTIFIED"
    assert calls == {
        "search": 5,
        "gate2": 1,
        "saved_search": 4,
        "initialized_efforts": ["xhigh"] * 5,
    }


def test_batch_cli_single_branch_threads_replay_backend(
    tmp_path, monkeypatch, capsys
) -> None:
    seen: list[tuple[str, str]] = []

    def search(_prepared, **kwargs):
        seen.append((kwargs["lift_backend"], kwargs["replay_backend"]))
        return _closed_search()

    entries = _install_live_batch_fakes(
        monkeypatch,
        search=search,
        complete=lambda *_a, **_k: _certified_execution(),
    )
    result = scrc_batch.main(
        [
            "--run",
            "--case",
            entries[0].case_id,
            "--lift-backend",
            "responses",
            "--replay-backend",
            "miniswe",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 0
    capsys.readouterr()
    assert seen == [("responses", "miniswe")]


def test_batch_cli_runs_tilelang_through_the_same_candidate_search(
    tmp_path, monkeypatch, capsys
) -> None:
    seen: list[tuple[str, int, int]] = []

    def search(prepared, **kwargs):
        seen.append(
            (
                prepared.case_id,
                kwargs["replay_width"],
                kwargs["max_pre_replay_rejections"],
            )
        )
        return _closed_search()

    _install_live_batch_fakes(
        monkeypatch,
        search=search,
        complete=lambda *_a, **_k: _certified_execution(),
    )
    result = scrc_batch.main(
        [
            "--run",
            "--case",
            "tilelang:add2d",
            "--lift-backend",
            "miniswe",
            "--replay-backend",
            "miniswe",
            "--replay-width",
            "2",
            "--max-branch-workers",
            "2",
            "--pre-replay-rejections",
            "5",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cases"][0]["classification"] == "scrc_certified"
    assert payload["cases"][0]["ordinal"] > 28
    assert seen == [("tilelang:add2d", 2, 5)]


def test_batch_cli_runs_multiple_cases_concurrently_and_aggregates_by_ordinal(
    tmp_path, monkeypatch, capsys
) -> None:
    search_barrier = Barrier(4, timeout=2.0)
    counter_lock = Lock()
    active = 0
    maximum_active = 0

    def fake_search(_prepared, **_kwargs):
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        search_barrier.wait()
        with counter_lock:
            active -= 1
        return _closed_search()

    entries = _install_live_batch_fakes(
        monkeypatch,
        search=fake_search,
        complete=lambda *_a, **_k: _certified_execution(),
    )
    selected_entries = entries[:2]
    explicit_cases = [
        token
        for entry in reversed(selected_entries)
        for token in ("--case", entry.case_id)
    ]

    result = scrc_batch.main(
        [
            "--run",
            *explicit_cases,
            "--case-workers",
            "2",
            "--candidates",
            "2",
            "--max-branch-workers",
            "4",
            "--gate2-workers",
            "2",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    progress_events = _progress_events(captured.err)
    assert maximum_active == 4
    assert payload["protocol"] == "intentir.scrc.batch.v3"
    assert payload["scheduling"] == {
        "case_workers": 2,
        "candidates": 2,
        "max_branch_workers": 4,
        "gate2_workers": 2,
        "gate1_rounds": 3,
        "pre_replay_rejections": 3,
    }
    assert [case["ordinal"] for case in payload["cases"]] == sorted(
        entry.ordinal for entry in selected_entries
    )
    assert payload["status_counts"] == {
        "scrc_certified": len(selected_entries)
    }
    assert "intentir.scrc.progress.v1" not in captured.out
    assert progress_events
    assert all(
        event["protocol"] == "intentir.scrc.progress.v1"
        and set(event) <= _PROGRESS_FIELDS
        for event in progress_events
    )
    case_started = [
        index
        for index, event in enumerate(progress_events)
        if event["event"] == "case_started"
    ]
    case_finished = [
        index
        for index, event in enumerate(progress_events)
        if event["event"] == "case_finished"
    ]
    assert len(case_started) == len(selected_entries)
    assert len(case_finished) == len(selected_entries)
    assert max(case_started) < min(case_finished)
    assert sum(
        event["event"] == "gate2_started" for event in progress_events
    ) == len(selected_entries)
    assert sum(
        event["event"] == "gate2_finished" for event in progress_events
    ) == len(selected_entries)


def test_batch_cli_bounds_shared_gate2_concurrency(
    tmp_path, monkeypatch, capsys
) -> None:
    gate2_barrier = Barrier(2, timeout=2.0)
    counter_lock = Lock()
    active = 0
    maximum_active = 0

    def fake_complete(_prepared, _search):
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        gate2_barrier.wait()
        with counter_lock:
            active -= 1
        return _certified_execution()

    entries = _install_live_batch_fakes(
        monkeypatch,
        search=lambda *_a, **_k: _closed_search(),
        complete=fake_complete,
    )

    result = scrc_batch.main(
        [
            "--run",
            "--tier",
            "representative",
            "--case-workers",
            "4",
            "--max-branch-workers",
            "4",
            "--gate2-workers",
            "2",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert maximum_active == 2
    assert payload["status_counts"] == {"scrc_certified": len(entries)}


def test_batch_cli_one_case_failure_does_not_cancel_parallel_peers(
    tmp_path, monkeypatch, capsys
) -> None:
    calls: list[str] = []
    counter_lock = Lock()

    def fake_search(prepared, **_kwargs):
        with counter_lock:
            calls.append(prepared.case_id)
        if prepared.case_id == failing_case:
            raise RuntimeError("isolated case failure")
        return _closed_search()

    entries = scrc_batch.build_workload_manifest().select(tier="representative")
    failing_case = entries[1].case_id
    _install_live_batch_fakes(
        monkeypatch,
        search=fake_search,
        complete=lambda *_a, **_k: _certified_execution(),
    )

    result = scrc_batch.main(
        [
            "--run",
            "--tier",
            "representative",
            "--case-workers",
            "3",
            "--max-branch-workers",
            "3",
            "--gate2-workers",
            "1",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert sorted(calls) == sorted(entry.case_id for entry in entries)
    assert payload["status_counts"] == {
        "infrastructure_failed": 1,
        "scrc_certified": len(entries) - 1,
    }
    failed = next(
        case for case in payload["cases"] if case["case_id"] == failing_case
    )
    assert "isolated case failure" in failed["diagnostic"]


def test_batch_cli_mixed_incomplete_fanout_fails_as_infrastructure(
    tmp_path, monkeypatch, capsys
) -> None:
    counter_lock = Lock()
    calls = 0

    def fake_search(_prepared, **_kwargs):
        nonlocal calls
        with counter_lock:
            calls += 1
            call = calls
        if call == 1:
            raise RuntimeError("one branch failed")
        return _no_closure_search()

    _install_live_batch_fakes(
        monkeypatch,
        search=fake_search,
        complete=lambda *_a, **_k: _certified_execution(),
    )

    result = scrc_batch.main(
        [
            "--run",
            "--case",
            "high_performance:liger_dyt",
            "--candidates",
            "2",
            "--max-branch-workers",
            "2",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    case = payload["cases"][0]
    assert case["classification"] == "infrastructure_failed"
    assert case["main_status"] is None
    assert case["call_budget"] == {}
    assert not case["details"]["candidate_call_budget_complete"]
    assert not case["details"]["aggregate_call_budget_complete"]
    assert case["details"]["candidate_observed_call_budget"]
    assert case["details"]["candidate_status_counts"] == {
        "infrastructure_failed": 1,
        "no_closure": 1,
    }
    assert case["evidence_path"].endswith("/fanout-001")
    assert "one branch failed" in case["diagnostic"]


def test_batch_cli_complete_no_closure_fanout_is_not_infrastructure(
    tmp_path, monkeypatch, capsys
) -> None:
    gate2_calls = 0

    def fake_complete(_prepared, _search):
        nonlocal gate2_calls
        gate2_calls += 1
        return _certified_execution()

    _install_live_batch_fakes(
        monkeypatch,
        search=lambda *_a, **_k: _no_closure_search(),
        complete=fake_complete,
    )

    result = scrc_batch.main(
        [
            "--run",
            "--case",
            "high_performance:liger_dyt",
            "--candidates",
            "2",
            "--max-branch-workers",
            "2",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    case = payload["cases"][0]
    assert gate2_calls == 0
    assert case["classification"] == "no_closure"
    assert case["main_status"] == "NO_CLOSURE"
    assert case["details"]["candidate_status_counts"] == {"no_closure": 2}
    assert case["evidence_path"].endswith("/fanout-001")


def test_batch_cli_all_source_observer_gaps_get_dedicated_fanout_class(
    tmp_path, monkeypatch, capsys
) -> None:
    import intentir.verify.scrc_workload_run as workload_module

    diagnostic = (
        "observer unsupported: source TTIR: TTIRParseError: "
        "multi-block CFG is not supported"
    )

    def fake_search(*_args, **_kwargs):
        search = _no_closure_search()
        search.report.test_diagnostic = diagnostic
        return search

    _install_live_batch_fakes(
        monkeypatch,
        search=fake_search,
        complete=lambda *_a, **_k: _certified_execution(),
    )
    monkeypatch.setattr(
        workload_module,
        "scrc_gate1_search_diagnostic",
        lambda report: report.test_diagnostic,
    )

    result = scrc_batch.main(
        [
            "--run",
            "--case",
            "high_performance:liger_dyt",
            "--candidates",
            "2",
            "--max-branch-workers",
            "2",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    case = payload["cases"][0]
    assert payload["status_counts"] == {"observer_unsupported": 1}
    assert case["classification"] == "observer_unsupported"
    assert case["main_status"] == "NO_CLOSURE"
    assert diagnostic in case["diagnostic"]
    assert case["details"]["aggregate_call_budget_complete"]


def test_batch_cli_mixed_source_observer_and_semantic_failures_stay_no_closure(
    tmp_path, monkeypatch, capsys
) -> None:
    import intentir.verify.scrc_workload_run as workload_module

    counter_lock = Lock()
    calls = 0

    def fake_search(*_args, **_kwargs):
        nonlocal calls
        with counter_lock:
            calls += 1
            diagnostic = (
                "observer unsupported: source TTIR: TTIRParseError: "
                "multi-block CFG is not supported"
                if calls == 1
                else "Gate1 hard mismatch: dataflow"
            )
        search = _no_closure_search()
        search.report.test_diagnostic = diagnostic
        return search

    _install_live_batch_fakes(
        monkeypatch,
        search=fake_search,
        complete=lambda *_a, **_k: _certified_execution(),
    )
    monkeypatch.setattr(
        workload_module,
        "scrc_gate1_search_diagnostic",
        lambda report: report.test_diagnostic,
    )

    result = scrc_batch.main(
        [
            "--run",
            "--case",
            "high_performance:liger_dyt",
            "--candidates",
            "2",
            "--max-branch-workers",
            "2",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 2
    case = json.loads(capsys.readouterr().out)["cases"][0]
    assert case["classification"] == "no_closure"
    assert "observer unsupported" in case["diagnostic"]
    assert "Gate1 hard mismatch" in case["diagnostic"]


def test_batch_cli_winner_with_infrastructure_sibling_has_partial_budget(
    tmp_path, monkeypatch, capsys
) -> None:
    counter_lock = Lock()
    calls = 0

    def fake_search(_prepared, **_kwargs):
        nonlocal calls
        with counter_lock:
            calls += 1
            call = calls
        if call == 1:
            raise RuntimeError("one sibling failed")
        return _closed_search()

    _install_live_batch_fakes(
        monkeypatch,
        search=fake_search,
        complete=lambda *_a, **_k: _certified_execution(),
    )

    result = scrc_batch.main(
        [
            "--run",
            "--case",
            "high_performance:liger_dyt",
            "--candidates",
            "2",
            "--max-branch-workers",
            "2",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert result == 0
    case = json.loads(capsys.readouterr().out)["cases"][0]
    assert case["classification"] == "scrc_certified"
    assert case["call_budget"] == {}
    assert not case["details"]["candidate_call_budget_complete"]
    assert not case["details"]["aggregate_call_budget_complete"]
    assert case["details"]["candidate_status_counts"] == {
        "gate1_closed": 1,
        "infrastructure_failed": 1,
    }


def test_live_progress_can_be_disabled_without_changing_final_json(
    tmp_path, monkeypatch, capsys
) -> None:
    entries = _install_live_batch_fakes(
        monkeypatch,
        search=lambda *_a, **_k: _closed_search(),
        complete=lambda *_a, **_k: _certified_execution(),
    )

    result = scrc_batch.main(
        [
            "--run",
            "--case",
            entries[0].case_id,
            "--run-root",
            str(tmp_path / "runs"),
            "--no-progress",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert json.loads(captured.out)["status_counts"] == {"scrc_certified": 1}


def test_progress_heartbeat_reports_each_active_case_with_closed_fields() -> None:
    stream = StringIO()
    emitter = scrc_batch._LiveProgressEmitter(
        enabled=True,
        stream=stream,
        heartbeat_interval=60.0,
        clock=lambda: 10.0,
    )
    emitter.start()
    emitter.case_started("case-a", 2)
    emitter.stage_transition(
        "case-a",
        2,
        branch_id=1,
        stage="lift",
        attempt=2,
        status="started",
    )
    emitter.case_started("case-b", 5)
    emitter.stage_transition(
        "case-b",
        5,
        branch_id=3,
        stage="gate1",
        formal_round=1,
        status="started",
    )
    emitter.emit_heartbeat()
    emitter.case_finished("case-a", 2, status="no_closure")
    emitter.case_finished("case-b", 5, status="scrc_certified")
    emitter.close()

    events = _progress_events(stream.getvalue())
    heartbeats = [event for event in events if event["event"] == "heartbeat"]
    assert [(event["case_id"], event["stage"]) for event in heartbeats] == [
        ("case-a", "lift"),
        ("case-b", "gate1"),
    ]
    assert heartbeats[0]["attempt"] == 2
    assert heartbeats[1]["formal_round"] == 1
    assert all(set(event) <= _PROGRESS_FIELDS for event in events)
    encoded = stream.getvalue()
    assert "source" not in encoded
    assert "artifact" not in encoded
    assert "diagnostic" not in encoded
    assert "provider" not in encoded
    assert "hidden" not in encoded


def test_broken_progress_stream_is_non_fatal() -> None:
    class BrokenStream:
        def write(self, _value):
            raise OSError("stderr is closed")

        def flush(self):
            raise OSError("stderr is closed")

    emitter = scrc_batch._LiveProgressEmitter(
        enabled=True,
        stream=BrokenStream(),
        heartbeat_interval=60.0,
    )
    emitter.start()
    emitter.case_started("case-a", 1)
    emitter.stage_transition(
        "case-a",
        1,
        branch_id=1,
        stage="lift",
        attempt=1,
        status="started",
    )
    emitter.case_finished("case-a", 1, status="infrastructure_failed")
    emitter.close()
