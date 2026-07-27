import pytest

from intentir.backend.cuda.production_cases import (
    PRODUCTION_CASE_IDS,
    PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE,
    _apply_workload_profile,
)
from intentir.verify.workload_adapters import TRITON_WORKLOAD_ADAPTERS


def test_large_j_only_roofline_profile_is_a_128_mib_masked_copy_boundary():
    from intentir.backend.cuda import production_cases

    adapter = production_cases._CERTIFIED_CANARY_ADAPTERS[
        PRODUCTION_CASE_IDS["masked_copy"]
    ]
    profiled = _apply_workload_profile(
        "masked_copy", adapter, PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE
    )
    shapes = {item.formal: item.shape for item in profiled.buffers}
    extent = shapes["X"][0]
    assert shapes["Y"] == (extent,)
    assert 2 * extent * 4 == 128 * 1024 * 1024
    assert dict((item.formal, item.value) for item in profiled.scalars)["N"] == extent
    assert profiled.grid == (extent // 1024,)


def test_large_roofline_profile_rejects_unregistered_kernel():
    adapter = TRITON_WORKLOAD_ADAPTERS[PRODUCTION_CASE_IDS["dyt"]]
    with pytest.raises(ValueError, match="not registered"):
        _apply_workload_profile(
            "dyt", adapter, PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE
        )
