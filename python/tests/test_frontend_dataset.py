from __future__ import annotations

from intentir.verify.frontend_dataset import load_frontend_dataset
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
    assert validation.counts["cuda"] >= 4
    assert all(item.location.startswith("dataset://") for item in catalog.kernels)

