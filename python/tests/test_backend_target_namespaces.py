from __future__ import annotations

import importlib
from pathlib import Path


def test_flat_cuda_modules_are_identity_compatible_aliases() -> None:
    pairs = (
        ("intentir.backend.compile", "intentir.backend.cuda.compile"),
        ("intentir.backend.launch", "intentir.backend.cuda.launch"),
        ("intentir.backend.orchestrate", "intentir.backend.cuda.orchestrate"),
        ("intentir.backend.selection", "intentir.backend.cuda.selection"),
        ("intentir.backend.schedules", "intentir.backend.cuda.schedules"),
        ("intentir.backend.cuda_dispatch", "intentir.backend.cuda.dispatch"),
    )

    for compatibility_name, canonical_name in pairs:
        compatibility = importlib.import_module(compatibility_name)
        canonical = importlib.import_module(canonical_name)
        assert compatibility is canonical


def test_cuda_implementation_has_no_flat_cuda_module_dependencies() -> None:
    modules = (
        "intentir.backend.cuda.compile",
        "intentir.backend.cuda.launch",
        "intentir.backend.cuda.orchestrate",
        "intentir.backend.cuda.selection",
        "intentir.backend.cuda.schedules",
        "intentir.backend.cuda.dispatch",
        "intentir.backend.cuda.pipeline",
    )
    forbidden = (
        "intentir.backend.compile",
        "intentir.backend.launch",
        "intentir.backend.orchestrate",
        "intentir.backend.selection",
        "intentir.backend.schedules",
        "intentir.backend.cuda_dispatch",
    )

    for module_name in modules:
        module = importlib.import_module(module_name)
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert not any(name in source for name in forbidden)
