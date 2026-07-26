from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from direct_fixtures import make_reduce_artifact
from intentir.extraction.providers.provider import ProviderConfig


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "tools/scrc_gate1_continue.py"
_SPEC = importlib.util.spec_from_file_location(
    "intentir_scrc_gate1_continue_tool", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
scrc_continue = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scrc_continue
_SPEC.loader.exec_module(scrc_continue)


def _parent(tmp_path: Path) -> Path:
    parent = tmp_path / "old-run" / "attempt-03"
    parent.mkdir(parents=True)
    artifact = make_reduce_artifact()
    (parent / "artifact.json").write_text(json.dumps(artifact.to_json()))
    comparison = {
        "hard_pass": False,
        "hard_passes": {"dataflow": False},
        "primary_difference_category": "dataflow",
        "rank_tuple": [0, 3, 0.75, -4, 1, 0],
        "differences": [],
    }
    (parent / "gate1.json").write_text(
        json.dumps(
            {
                "status": "no_closure",
                "source_compile": {"ttir": "source-ttir"},
                "normal": {
                    "diagnostic": "observer mismatch",
                    "compile": {"ttir": "replay-ttir"},
                    "comparison": comparison,
                },
            }
        )
    )
    return parent


def test_ranked_continuation_cli_loads_exact_seed_and_saves_no_closure(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    case_id = "high_performance:test_ranked_continue"
    parent = _parent(tmp_path)
    provider = ProviderConfig(
        "test-provider", "https://provider.invalid/v1", "key", "model"
    )
    prepared = SimpleNamespace(entry=SimpleNamespace(case_id=case_id))
    calls: dict[str, object] = {}

    monkeypatch.setitem(scrc_continue.TRITON_WORKLOAD_ADAPTERS, case_id, object())
    monkeypatch.setattr(
        scrc_continue,
        "prepare_triton_workload",
        lambda _adapter, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        scrc_continue,
        "validate_workload_gate1_recheck_parent",
        lambda selected_parent, selected, artifact: calls.setdefault(
            "validated", (selected_parent, selected, artifact)
        ),
    )
    monkeypatch.setattr(
        scrc_continue,
        "type_check",
        lambda *_args: SimpleNamespace(passed=True),
    )
    monkeypatch.setattr(
        scrc_continue,
        "fingerprint_ttir",
        lambda text: SimpleNamespace(text=text),
    )
    monkeypatch.setattr(
        scrc_continue,
        "_compact_gate1_observer_diff",
        lambda _comparison, **_kwargs: {"bounded": True},
    )
    monkeypatch.setattr(scrc_continue, "get_provider", lambda _name: provider)

    def allocate(root, selected_case):
        calls["allocated"] = (Path(root), selected_case)
        path = Path(root) / "case" / "run-001"
        path.mkdir(parents=True)
        return path

    monkeypatch.setattr(scrc_continue, "next_workload_run_dir", allocate)
    monkeypatch.setattr(
        scrc_continue,
        "initialize_workload_evidence",
        lambda path, selected, **kwargs: calls.setdefault(
            "initialized", (path, selected, kwargs)
        ),
    )
    report = SimpleNamespace(
        closed=False,
        status=SimpleNamespace(value="NO_CLOSURE"),
    )
    search = SimpleNamespace(report=report)

    def search_gate1(selected, **kwargs):
        calls["search"] = (selected, kwargs)
        return search

    monkeypatch.setattr(
        scrc_continue, "search_prepared_workload_gate1", search_gate1
    )
    monkeypatch.setattr(
        scrc_continue,
        "save_workload_gate1_search",
        lambda path, selected, value, **kwargs: calls.setdefault(
            "saved", (path, selected, value, kwargs)
        ),
    )

    exit_code = scrc_continue.main(
        [
            "--case",
            case_id,
            "--parent-evidence",
            str(parent),
            "--provider",
            "test-provider",
            "--effort",
            "xhigh",
            "--replay-width",
            "3",
            "--retry-parent-gate1",
            "--run-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert exit_code == 1
    output_lines = capsys.readouterr().out.splitlines()
    payload = json.loads("\n".join(output_lines))
    assert payload["protocol"] == scrc_continue.RANKED_CONTINUATION_PROTOCOL
    assert payload["status"] == "NO_CLOSURE"
    seed = calls["search"][1]["initial_feedback"]
    assert seed.selection_rank == (0, 3, 0.75, -4, 1, 0)
    assert seed.previous_artifact.to_json() == (
        scrc_continue.canonicalize_gate1_artifact(
            make_reduce_artifact()
        ).to_json()
    )
    assert seed.diagnostics["observer_diff"] == {"bounded": True}
    metadata = calls["initialized"][2]["search_metadata"]
    assert metadata["initial_selection_rank"] == [0, 3, 0.75, -4, 1, 0]
    assert metadata["failed_parent_policy"] == "gate1-rank-best-so-far-v1"
    assert metadata["replay_width"] == 3
    assert calls["search"][1]["replay_width"] == 3
    assert metadata["retry_parent_gate1"] is True
    assert calls["search"][1]["retry_initial_artifact"] is True
    assert calls["saved"][3]["selected_for_gate2"] is False


def test_ranked_continuation_cli_rejects_closed_parent(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    case_id = "high_performance:test_closed_parent"
    parent = _parent(tmp_path)
    gate1_path = parent / "gate1.json"
    gate1 = json.loads(gate1_path.read_text())
    gate1["status"] = "replay_closed"
    gate1_path.write_text(json.dumps(gate1))
    monkeypatch.setitem(scrc_continue.TRITON_WORKLOAD_ADAPTERS, case_id, object())
    monkeypatch.setattr(
        scrc_continue,
        "prepare_triton_workload",
        lambda _adapter, **_kwargs: SimpleNamespace(
            entry=SimpleNamespace(case_id=case_id)
        ),
    )
    monkeypatch.setattr(
        scrc_continue,
        "validate_workload_gate1_recheck_parent",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        scrc_continue,
        "type_check",
        lambda *_args: SimpleNamespace(passed=True),
    )

    exit_code = scrc_continue.main(
        ["--case", case_id, "--parent-evidence", str(parent)]
    )

    assert exit_code == 2
    assert "requires an exact Gate1 no_closure parent" in capsys.readouterr().err
