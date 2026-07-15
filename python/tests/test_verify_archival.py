"""Architecture guards for the selective pre-SCRC verifier archive."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys


PYTHON_ROOT = Path(__file__).resolve().parents[1]


def test_verify_package_exports_only_active_gates() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import intentir.verify as verify; "
                "assert verify.__all__ == ['gate1', 'gate2']; "
                "assert 'intentir.verify.interpret' not in sys.modules; "
                "assert 'intentir.verify.back' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr


def test_public_interpreter_path_is_only_a_legacy_compatibility_shim() -> None:
    shim = importlib.import_module("intentir.verify.interpret")
    archived = importlib.import_module(
        "intentir.verify.back.legacy_v0.interpret"
    )

    assert shim.interpret is archived.interpret
    assert shim.InterpError is archived.InterpError
    assert shim._philox_uniform is archived._philox_uniform


def test_active_scrc_verification_does_not_import_archived_interpreter() -> None:
    active_files = [
        *(PYTHON_ROOT / "intentir" / "verify" / "gate1").glob("*.py"),
        *(PYTHON_ROOT / "intentir" / "verify" / "gate2").glob("*.py"),
        PYTHON_ROOT / "intentir" / "extraction" / "lifters" / "scrc.py",
    ]

    for path in active_files:
        source = path.read_text(encoding="utf-8")
        assert "intentir.verify.interpret" not in source, path
        assert "intentir.verify.back" not in source, path


def test_legacy_internal_consumers_use_the_explicit_archive_path() -> None:
    legacy_consumers = [
        PYTHON_ROOT / "intentir" / "extraction" / "data" / "corpus.py",
        PYTHON_ROOT / "intentir" / "extraction" / "data" / "run_universe.py",
        PYTHON_ROOT
        / "intentir"
        / "extraction"
        / "lifters"
        / "cuda_frontend.py",
    ]

    for path in legacy_consumers:
        source = path.read_text(encoding="utf-8")
        assert "intentir.verify.back.legacy_v0.interpret" in source, path
        assert "from intentir.verify.interpret" not in source, path
