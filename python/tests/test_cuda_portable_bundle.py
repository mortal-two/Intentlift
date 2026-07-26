from __future__ import annotations

import numpy as np
import pytest

from intentir.backend.cuda.benchmark import (
    ExtractedSingleLaunch,
    PreparedLaunchRecord,
)
from intentir.backend.cuda.portable_bundle import (
    cuda_compute_capability_for_chip,
    decode_array,
    decode_triton_spec,
    encode_array,
    encode_triton_spec,
)
from intentir.verify.gate2.triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
)


def test_portable_array_preserves_rank_zero_and_exact_bytes() -> None:
    scalar = np.asarray(127, dtype=np.int32)
    restored = decode_array(encode_array(scalar))

    assert restored.shape == ()
    assert restored.dtype == scalar.dtype
    assert restored.tobytes() == scalar.tobytes()


def test_extracted_launch_json_roundtrip_is_content_exact() -> None:
    launch = ExtractedSingleLaunch(
        cubin=b"\x7fELFtest-cubin",
        kernel_name="kernel",
        binary_name="binary",
        plan=(("ptr", 0), ("i32", 7), ("f32", 0.5)),
        record=PreparedLaunchRecord((2, 1, 1), (128, 1, 1), 0, 3, 1),
        source_mlir_sha256="a" * 64,
    )

    restored = ExtractedSingleLaunch.from_dict(launch.to_dict())

    assert restored == launch
    assert restored.cubin_sha256 == launch.cubin_sha256


def test_triton_execution_spec_roundtrip_keeps_complete_abi() -> None:
    spec = TritonExecutionSpec(
        arguments=(
            TritonArgument(
                "x",
                ArgumentRole.INPUT,
                value_key="input_x",
                dtype="f32",
            ),
            TritonArgument(
                "out",
                ArgumentRole.OUTPUT,
                dtype="f32",
                shape=(2, 3),
                output_key="result",
            ),
            TritonArgument("n", ArgumentRole.SCALAR, value=3),
        ),
        grid=(2,),
        schedule={"BLOCK": 32},
    )

    assert decode_triton_spec(encode_triton_spec(spec)) == spec


def test_cuda_cubin_target_decodes_exact_runtime_capability() -> None:
    assert cuda_compute_capability_for_chip("sm_90a") == (9, 0)
    assert cuda_compute_capability_for_chip("sm_120a") == (12, 0)
    with pytest.raises(ValueError, match="unsupported CUDA chip"):
        cuda_compute_capability_for_chip("compute_90")
