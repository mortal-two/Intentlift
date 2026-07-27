from __future__ import annotations

import copy

import pytest

from intentir.backend.dev_workload import (
    DevWorkloadConfig,
    DevWorkloadError,
    apply_dev_workload_config,
    prepare_dev_workload,
)
from intentir.ir.production_snapshot import content_digest
from intentir.verify.workload_adapters import TRITON_WORKLOAD_ADAPTERS


_CASE_ID = "high_performance:liger_rms_norm"


def _config(adapter_digest: str) -> DevWorkloadConfig:
    return DevWorkloadConfig(
        case_id=_CASE_ID,
        shape_config_id="llama3_8b_m2048_h4096",
        dtype_config_id="fp32_boundary_fp32_accum",
        base_adapter_digest=adapter_digest,
        concrete_extents={"x0": 4096, "x1": 2048},
        buffer_shapes={
            "Y_ptr": (2048, 4096),
            "X_ptr": (2048, 4096),
            "W_ptr": (4096,),
            "RSTD_ptr": (2048,),
        },
        buffer_dtypes={},
        scalar_values={
            "Y_row_stride": 4096,
            "X_row_stride": 4096,
            "W_row_stride": 4096,
            "n_cols": 4096,
        },
        semantic_constexprs={},
        c_values={"BLOCK_SIZE": 4096},
        grid=(2048,),
        grid_contract=("rows",),
        upstream_evidence=(
            "Dataset/Liger-Kernel/benchmark/scripts/benchmark_rms_norm.py:"
            "Llama-3-8B-M2048-H4096",
        ),
    )


def test_dev_workload_applies_natural_rmsnorm_shape_without_mutating_base() -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[_CASE_ID]
    before = copy.deepcopy(adapter.to_json())
    config = _config(content_digest(before))

    active = apply_dev_workload_config(adapter, config)

    assert adapter.to_json() == before
    assert {item.formal: item.shape for item in active.buffers} == {
        "Y_ptr": (2048, 4096),
        "X_ptr": (2048, 4096),
        "W_ptr": (4096,),
        "RSTD_ptr": (2048,),
    }
    assert dict((item.formal, item.value) for item in active.scalars)["n_cols"] == 4096
    assert dict(active.c_values)["BLOCK_SIZE"] == 4096
    assert active.grid == (2048,)


def test_dev_workload_round_trip_and_digest_pin_fail_closed() -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[_CASE_ID]
    config = _config(content_digest(adapter.to_json()))
    restored = DevWorkloadConfig.from_json(
        __import__("json").dumps(config.to_dict())
    )
    assert restored.to_dict() == config.to_dict()

    stale = _config("0" * 64)
    with pytest.raises(DevWorkloadError, match="base adapter changed"):
        apply_dev_workload_config(adapter, stale)


def test_dev_workload_rejects_unknown_binding_and_tampered_digest() -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[_CASE_ID]
    config = _config(content_digest(adapter.to_json()))
    payload = config.to_dict()
    payload["scalar_values"]["not_a_formal"] = 1
    payload["config_digest"] = content_digest(
        {key: value for key, value in payload.items() if key != "config_digest"}
    )
    parsed = DevWorkloadConfig.from_dict(payload)
    with pytest.raises(DevWorkloadError, match="unknown scalars"):
        apply_dev_workload_config(adapter, parsed)

    payload = config.to_dict()
    payload["grid"] = [1]
    with pytest.raises(DevWorkloadError, match="config_digest mismatch"):
        DevWorkloadConfig.from_dict(payload)


def test_dev_workload_prepares_the_real_source_with_natural_abi() -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[_CASE_ID]
    prepared = prepare_dev_workload(_config(content_digest(adapter.to_json())))

    assert prepared.adapter.case_id == _CASE_ID
    assert prepared.adapter.grid == (2048,)
    assert {item.formal: item.shape for item in prepared.adapter.buffers}[
        "X_ptr"
    ] == (2048, 4096)
    assert prepared.request.source.gate_source.function == "scrc_k0"
