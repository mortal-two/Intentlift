from __future__ import annotations

import pytest

from intentir.verify.gate1.compiler import TritonCompiler
from intentir.verify.workload_adapters import (
    BufferRole,
    TRITON_WORKLOAD_ADAPTERS,
    prepare_triton_workload,
)
from intentir.verify.workload_manifest import build_workload_manifest


def test_six_adapters_match_the_representative_manifest_projection() -> None:
    representatives = build_workload_manifest().select(tier="representative")

    assert list(TRITON_WORKLOAD_ADAPTERS) == [
        entry.case_id for entry in representatives
    ]
    assert len(TRITON_WORKLOAD_ADAPTERS) == 6


@pytest.mark.parametrize("adapter", TRITON_WORKLOAD_ADAPTERS.values())
def test_real_adapter_preparation_validates_source_abi_and_anonymizes(adapter) -> None:
    prepared = prepare_triton_workload(adapter)
    request = prepared.request
    code, _origin = request.source.gate_source.read_code()

    assert prepared.entry.case_id == adapter.case_id
    assert prepared.source_location.startswith(("module://", "dataset://"))
    assert set(prepared.source_slice.parameters) <= set(
        prepared.source_to_opaque
    )
    assert "def scrc_k0(" in code
    assert prepared.entry.symbol not in code
    assert request.binding.values == {
        prepared.source_to_opaque[name]: value
        for name, value in adapter.c_values
    }
    assert set(request.binding.gate1_case.arguments) == {
        prepared.source_to_opaque[item.formal]
        for item in (*adapter.buffers, *adapter.scalars)
    }
    assert any(item.role is not BufferRole.INPUT for item in adapter.buffers)


def test_adapter_json_contains_only_harness_bindings_not_semantic_formulas() -> None:
    for adapter in TRITON_WORKLOAD_ADAPTERS.values():
        payload = adapter.to_json()
        encoded = str(payload).casefold()
        for forbidden in (
            "source_code",
            "replay",
            "operator",
            "predicate",
            "formula",
            "gate1_diff",
        ):
            assert forbidden not in encoded


@pytest.mark.parametrize("adapter", TRITON_WORKLOAD_ADAPTERS.values())
def test_real_adapter_source_reaches_triton_frontend(adapter) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires Triton CUDA")
    prepared = prepare_triton_workload(adapter)

    result = TritonCompiler(timeout=240).compile(
        prepared.request.source.gate_source,
        prepared.request.binding.gate1_case,
    )

    assert result.ok, result.diagnostic
    assert result.ttir
