from __future__ import annotations

import json
from pathlib import Path

from intentir.backend.common import load_high_performance_registry
from intentir.backend.selection import DEFAULT_PASSES


_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = _ROOT / "spec/lowering/high-performance-kernels.json"


def _test_path(node_id: str) -> Path:
    return _ROOT / node_id.split("::", 1)[0]


def test_high_performance_registry_preserves_28_kernels_and_29_scenarios() -> None:
    registry = load_high_performance_registry()

    assert len(registry.kernels) == 28
    assert registry.scenario_29["counted_as_direct_kernel"] is False
    assert registry.scenario_29["component_kernel_ordinals"] == [19, 22, 21]
    assert registry.summary() == {
        "exact_capture_alignment": 20,
        "device_execution_proven": 28,
        "automatic_selection_executed": 20,
        "explicit_pass_execution": 8,
        "exact_variant_gaps": 0,
        "performance": {
            "archived_formal_pass": 16,
            "archived_formal_below_threshold": 4,
            "runtime_parity_guard": 6,
            "correctness_only": 2,
            "none": 0,
        },
    }


def test_registry_evidence_paths_and_cuda_fast_path_ids_are_real() -> None:
    registry = load_high_performance_registry(_REGISTRY)
    known_fast_paths = {descriptor.fast_path_id for descriptor in DEFAULT_PASSES}

    for kernel in registry.kernels:
        for test in (*kernel.cuda_tests, *kernel.family_tests, *kernel.lit_tests):
            assert _test_path(test).is_file(), (
                f"{kernel.kernel_id} references missing evidence {test}"
            )
        if kernel.fast_path_id is not None:
            assert kernel.fast_path_id in known_fast_paths
        if kernel.candidate_fast_path_id is not None:
            assert kernel.candidate_fast_path_id in known_fast_paths


def test_registry_is_lowering_only_and_keeps_extraction_gaps_separate() -> None:
    payload = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True).lower()
    assert '"scope": "lowering_only"' in serialized
    assert "verdict" not in serialized
    assert "gate1" not in serialized
    assert "gate2" not in serialized

    registry = load_high_performance_registry(_REGISTRY)
    lowering_gaps = [
        kernel for kernel in registry.kernels
        if kernel.lowering_status == "exact_variant_gap"
    ]
    assert lowering_gaps == []

    one_col = registry.by_id[
        "high_performance:flash_attn.flash_attn_triton._bwd_kernel_one_col_block"
    ]
    assert one_col.lowering_status == "explicit_pass_executed"
    assert one_col.source_alignment == "upstream_compared_handwritten_mlir"
    assert one_col.cuda_tests == (
        "python/tests/test_flash_bwd_one_col_pk.py::"
        "test_flash_bwd_one_col_exact_symbol_lowering",
    )
    assert any("live extraction" in blocker for blocker in one_col.blockers)
