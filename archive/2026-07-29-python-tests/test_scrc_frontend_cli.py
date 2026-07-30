from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from intentir.verify.scrc_batch import BatchClassification
from intentir.verify.scrc_runner import MainStatus


_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "intentir_scrc_frontend_tool", _ROOT / "tools" / "scrc_frontend.py"
)
assert _SPEC is not None and _SPEC.loader is not None
scrc_frontend = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scrc_frontend)


def test_frontend_cli_accepts_codex_lift_and_replay_backends() -> None:
    args = scrc_frontend._parser().parse_args(
        [
            "--case",
            "cuda:tilelang_add2d",
            "--lift-backend",
            "codex",
            "--replay-backend",
            "codex",
        ]
    )

    assert args.lift_backend == "codex"
    assert args.replay_backend == "codex"


def test_frontend_list_keeps_triton_when_native_dataset_is_missing(tmp_path) -> None:
    payload = scrc_frontend._listed_cases(str(tmp_path / "missing"))

    assert payload["count"] == 28
    assert payload["native_unavailable_count"] == 1
    assert payload["native_unavailable"][0]["status"] == "case_unavailable"


def test_frontend_cli_exit_code_tracks_formal_main_status(monkeypatch, tmp_path) -> None:
    prepared = SimpleNamespace(request=SimpleNamespace(frontend="cuda"))
    provider = object()

    monkeypatch.setattr(
        scrc_frontend,
        "prepare_registered_scrc_workload",
        lambda _case_id, dataset_root=None: prepared,
    )
    monkeypatch.setattr(scrc_frontend, "get_provider", lambda *_args: provider)
    monkeypatch.setattr(
        scrc_frontend,
        "next_workload_run_dir",
        lambda *_args: tmp_path / "run",
    )
    monkeypatch.setattr(scrc_frontend, "initialize_workload_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scrc_frontend,
        "execute_prepared_workload",
        lambda *_args, **_kwargs: SimpleNamespace(
            report=SimpleNamespace(main_status=MainStatus.SCRC_CERTIFIED)
        ),
    )
    monkeypatch.setattr(scrc_frontend, "save_workload_execution", lambda *_args: None)

    assert scrc_frontend.main(
        [
            "--case",
            "cuda:tilelang_add2d",
            "--lift-backend",
            "codex",
            "--replay-backend",
            "codex",
            "--run-root",
            str(tmp_path),
        ]
    ) == 0

    monkeypatch.setattr(
        scrc_frontend,
        "execute_prepared_workload",
        lambda *_args, **_kwargs: SimpleNamespace(
            report=SimpleNamespace(main_status=MainStatus.NO_CLOSURE)
        ),
    )
    monkeypatch.setattr(
        scrc_frontend,
        "classify_scrc_report",
        lambda _report: BatchClassification.NO_CLOSURE,
    )
    assert scrc_frontend.main(
        ["--case", "cuda:tilelang_add2d", "--run-root", str(tmp_path)]
    ) == 1
