from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


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
