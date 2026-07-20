from __future__ import annotations

from pathlib import Path


_BACKEND_ROOT = Path(__file__).parents[1] / "intentir" / "backend"


def test_backend_root_contains_no_flat_implementation_modules() -> None:
    legacy_files = (
        "abi.py",
        "autotune.py",
        "backward.py",
        "compile.py",
        "cpu.py",
        "launch.py",
        "launch_plan.py",
        "layer_c.py",
        "opgraph.py",
        "orchestrate.py",
        "pipeline.py",
        "runtime_contract.py",
        "selection.py",
        "schedules.py",
        "cuda_dispatch.py",
        "variant_manifest.py",
        "variant_planner.py",
    )

    assert not [name for name in legacy_files if (_BACKEND_ROOT / name).exists()]


def test_cuda_implementation_does_not_reference_removed_modules() -> None:
    forbidden = (
        "intentir.backend.compile",
        "intentir.backend.launch",
        "intentir.backend.orchestrate",
        "intentir.backend.selection",
        "intentir.backend.schedules",
        "intentir.backend.cuda_dispatch",
        "intentir.backend.pipeline",
    )

    for module_path in (_BACKEND_ROOT / "cuda").glob("*.py"):
        source = module_path.read_text(encoding="utf-8")
        assert not any(name in source for name in forbidden), module_path


def test_backend_package_does_not_reexport_cuda_api() -> None:
    source = (_BACKEND_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "from intentir.backend.cuda" not in source
    assert "import intentir.backend.cuda" not in source
    assert "__getattr__" not in source


def test_backend_dependency_direction_is_target_explicit() -> None:
    backend_sources = tuple(_BACKEND_ROOT.rglob("*.py"))
    for module_path in backend_sources:
        source = module_path.read_text(encoding="utf-8")
        assert "intentir.benchmarks" not in source, module_path

    for subtree in ("cpu", "toolchain"):
        for module_path in (_BACKEND_ROOT / subtree).rglob("*.py"):
            source = module_path.read_text(encoding="utf-8")
            assert "intentir.backend.cuda" not in source, module_path

    ir_root = _BACKEND_ROOT.parent / "ir"
    for module_path in ir_root.rglob("*.py"):
        source = module_path.read_text(encoding="utf-8")
        assert "intentir.backend" not in source, module_path
