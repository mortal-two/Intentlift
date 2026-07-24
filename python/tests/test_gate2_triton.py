from __future__ import annotations

import numpy as np
import pytest

from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate2 import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
    TritonRunner,
)
from intentir.verify.gate2.triton import _launch_by_name


class _FakeKernel:
    arg_names = (
        "input_ptr",
        "MODE",
        "output_ptr",
        "extent",
        "BLOCK",
    )

    def __init__(self) -> None:
        self.grid: tuple[int, ...] | None = None
        self.kwargs: dict[str, object] | None = None

    def __getitem__(self, grid: tuple[int, ...]):
        self.grid = grid

        def launch(**kwargs: object) -> None:
            self.kwargs = kwargs

        return launch


def test_name_keyed_launch_binds_interleaved_constexpr_without_collision() -> None:
    kernel = _FakeKernel()
    input_value = object()
    output_value = object()

    _launch_by_name(
        kernel,
        (7,),
        {
            "input_ptr": input_value,
            "output_ptr": output_value,
            "extent": 65,
        },
        {"MODE": 3, "BLOCK": 128, "num_warps": 4},
    )

    assert kernel.grid == (7,)
    assert kernel.kwargs == {
        "input_ptr": input_value,
        "output_ptr": output_value,
        "extent": 65,
        "MODE": 3,
        "BLOCK": 128,
        "num_warps": 4,
    }


def test_name_keyed_launch_rejects_incomplete_or_reordered_runtime_abi() -> None:
    kernel = _FakeKernel()
    with pytest.raises(ValueError, match="missing Triton replay interface"):
        _launch_by_name(
            kernel,
            (1,),
            {"input_ptr": object(), "output_ptr": object(), "extent": 8},
            {"BLOCK": 8},
        )

    with pytest.raises(ValueError, match="runtime argument order differs"):
        _launch_by_name(
            kernel,
            (1,),
            {
                "output_ptr": object(),
                "input_ptr": object(),
                "extent": 8,
            },
            {"MODE": 0, "BLOCK": 8},
        )


def test_triton_execution_spec_rejects_argument_schedule_overlap() -> None:
    with pytest.raises(ValueError, match="arguments and schedule overlap"):
        TritonExecutionSpec(
            (
                TritonArgument(
                    "extent",
                    ArgumentRole.SCALAR,
                    value=8,
                ),
            ),
            (1,),
            {"extent": 8},
        )


def test_none_argument_is_closed_and_name_keyed() -> None:
    argument = TritonArgument("optional", ArgumentRole.NONE)
    assert argument.role is ArgumentRole.NONE
    with pytest.raises(ValueError, match="none argument forbids"):
        TritonArgument(
            "optional", ArgumentRole.NONE, value_key="dummy"
        )

    class OptionalKernel:
        arg_names = ("input_ptr", "optional", "output_ptr")

        def __init__(self):
            self.kwargs = None

        def __getitem__(self, _grid):
            def launch(**kwargs):
                self.kwargs = kwargs

            return launch

    kernel = OptionalKernel()
    _launch_by_name(
        kernel,
        (1,),
        {"input_ptr": object(), "optional": None, "output_ptr": object()},
        {},
    )
    assert kernel.kwargs is not None
    assert kernel.kwargs["optional"] is None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA is unavailable")
def test_real_triton_launch_supports_interleaved_constexpr_binding() -> None:
    source = KernelSource.from_code(
        """
import triton
import triton.language as tl

@triton.jit
def interleaved_kernel(
    input_ptr,
    MODE: tl.constexpr,
    output_ptr,
    extent,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < extent
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    if MODE == 1:
        values = values + 1.0
    tl.store(output_ptr + offsets, values, mask=mask)
""",
        "interleaved_kernel",
    )
    spec = TritonExecutionSpec(
        (
            TritonArgument(
                "input_ptr", ArgumentRole.INPUT, value_key="X", dtype="f32"
            ),
            TritonArgument(
                "output_ptr",
                ArgumentRole.OUTPUT,
                dtype="f32",
                shape=(9,),
                output_key="Y",
            ),
            TritonArgument(
                "extent", ArgumentRole.SCALAR, value_key="N"
            ),
        ),
        (1,),
        {"MODE": 1, "BLOCK": 16, "num_warps": 4},
    )

    result = TritonRunner().run(
        "replay",
        source,
        spec,
        {
            "X": np.arange(9, dtype=np.float32),
            "N": 9,
        },
    )

    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["Y"], np.arange(9, dtype=np.float32) + 1.0
    )
    assert result.metadata["binding_protocol"] == (
        "triton_name_keyed_complete_abi_v1"
    )
