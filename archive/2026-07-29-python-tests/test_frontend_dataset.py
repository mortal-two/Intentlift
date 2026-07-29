from __future__ import annotations

from pathlib import Path

from intentir.verify.frontend_dataset import (
    load_frontend_dataset,
    resolve_cuda_include_dirs,
)
from intentir.verify.gate1.frontend import (
    active_gate1_frontends,
    frontend_gate1_capability,
)
from intentir.verify.gate1.model import Frontend
from intentir.verify.gate1.profile import observer_profile_for


def test_all_declared_public_frontends_have_active_gate1_capabilities() -> None:
    assert active_gate1_frontends() == (
        Frontend.TRITON,
        Frontend.CUTE,
        Frontend.TILELANG,
        Frontend.CUDA,
    )
    for frontend in active_gate1_frontends():
        capability = frontend_gate1_capability(frontend)
        assert capability.profile == observer_profile_for(frontend)


def test_external_multifrontend_dataset_retains_reference_kernel_corpus() -> None:
    catalog = load_frontend_dataset()
    validation = catalog.validate()
    assert validation.passed, validation.errors
    assert validation.counts["tilelang"] >= 30
    assert validation.counts["cuda"] >= 30
    assert all(item.location.startswith("dataset://") for item in catalog.kernels)

    generated = tuple(
        item
        for item in catalog.select(frontend="cuda")
        if item.metadata.get("origin") == "tilelang_export"
    )
    handwritten = tuple(
        item
        for item in catalog.select(frontend="cuda")
        if item.metadata.get("origin") == "handwritten"
    )
    assert len(generated) == len(catalog.select(frontend="tilelang")) == 31
    assert len(handwritten) == 4
    for item in generated:
        assert item.metadata["protocol"] == "intentir.tilelang-cuda-snapshot.v2"
        assert item.metadata["compiler"]["version"] == "0.1.8"
        parameters = item.metadata["interface"]["parameters"]
        assert parameters
        assert any(parameter["mode"] == "output" for parameter in parameters)
        assert item.metadata["cuda_abi"]["parameters"]
        assert item.path.with_suffix(".ptx").is_file()
        assert item.path.with_suffix(".json").is_file()
        assert all(Path(path).is_dir() for path in resolve_cuda_include_dirs(item))
