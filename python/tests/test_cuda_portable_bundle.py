from __future__ import annotations

import hashlib

import numpy as np
import pytest

from intentir.backend.cuda.benchmark import (
    ExtractedGraphKernel,
    ExtractedGraphLaunch,
    ExtractedGraphLaunchRecord,
    ExtractedSingleLaunch,
    PreparedLaunchRecord,
    portable_launch_from_dict,
)
from intentir.backend.cuda.portable_bundle import (
    cuda_compute_capability_for_chip,
    decode_array,
    decode_triton_spec,
    encode_array,
    encode_triton_spec,
    logical_runtime_array,
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


def test_logical_runtime_array_preserves_j_rank_zero_abi() -> None:
    scalar = logical_runtime_array(
        np.asarray([3.25], dtype=np.float32), rank=0, name="scale"
    )

    assert scalar.shape == ()
    assert scalar.dtype == np.float32
    assert float(scalar) == pytest.approx(3.25)


def test_logical_runtime_array_rejects_non_scalar_rank_zero_value() -> None:
    with pytest.raises(ValueError, match="rank-0 logical input"):
        logical_runtime_array(
            np.asarray([1, 2], dtype=np.int32), rank=0, name="extent"
        )


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
    assert portable_launch_from_dict(launch.to_dict()) == launch


def test_extracted_graph_launch_json_roundtrip_is_content_exact() -> None:
    source = "module { func.func @main() }\n"
    graph = ExtractedGraphLaunch(
        kernels=(
            ExtractedGraphKernel(b"\x7fELFfirst", "first", "binary0"),
            ExtractedGraphKernel(b"\x7fELFsecond", "second", "binary1"),
        ),
        source_mlir=source,
        binary_mlir="module { gpu.launch_func @binary0::@first }\n",
        record=ExtractedGraphLaunchRecord(
            kernel_count=2,
            launch_count=3,
            intermediate_buffer_count=1,
        ),
        source_mlir_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )

    restored = ExtractedGraphLaunch.from_dict(graph.to_dict())

    assert restored == graph
    assert restored.executable_sha256 == graph.executable_sha256
    assert restored.binary_size_bytes == len(b"\x7fELFfirst\x7fELFsecond")
    assert portable_launch_from_dict(graph.to_dict()) == graph


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
