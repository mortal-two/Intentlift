from __future__ import annotations

import json

from intentir.extraction.lifters.scrc import (
    SCRCCompilerArgument,
    SCRCCompilerView,
    anonymize_cuda_source,
    prepare_scrc_lift,
)


CUDA_SOURCE = r'''
#include <stdint.h>

extern "C" __global__ void vec_add(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N);

// Parameter names inside comments are not executable identifiers: A B C N.
extern "C" __global__ void vec_add(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {
  int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < N) C[index] = A[index] + B[index];
}
'''


def _compiler_view() -> SCRCCompilerView:
    return SCRCCompilerView(
        {},
        arguments=(
            SCRCCompilerArgument("A", "tensor", "f32", (32,), mode="input"),
            SCRCCompilerArgument("B", "tensor", "f32", (32,), mode="input"),
            SCRCCompilerArgument("C", "tensor", "f32", (32,), mode="output"),
            SCRCCompilerArgument("N", "scalar", "i32", value=32, mode="value"),
        ),
        grid=(1, 1, 1),
        grid_contract=("ceil_div(N,32)", "1", "1"),
        parameter_order=("A", "B", "C", "N"),
    )


def test_cuda_lift_anonymizer_selects_definition_and_runtime_abi() -> None:
    anonymous, binding, mapping = anonymize_cuda_source(
        CUDA_SOURCE,
        source_symbol="vec_add",
    )

    assert binding.source_function == "vec_add"
    assert mapping == {
        "vec_add": "scrc_k0",
        "A": "p0",
        "B": "p1",
        "C": "p2",
        "N": "p3",
    }
    assert anonymous.count("void scrc_k0(") == 2
    assert "if (index < p3) p2[index] = p0[index] + p1[index];" in anonymous
    assert "comments are not executable identifiers: A B C N" in anonymous


def test_cuda_lift_parameter_renaming_is_scoped_to_the_selected_entry() -> None:
    source = r'''
__device__ int helper(int N) { return N + 1; }
extern "C" __global__ void selected(const float* X, float* Y, int SIZE);
extern "C" __global__ void selected(const float* A, float* B, int N) {
  int index = helper(N);
  B[index] = A[index];
}
'''
    anonymous, _binding, mapping = anonymize_cuda_source(
        source, source_symbol="selected"
    )

    assert mapping == {
        "selected": "scrc_k0",
        "A": "p0",
        "B": "p1",
        "N": "p2",
    }
    assert "helper(int N) { return N + 1; }" in anonymous
    assert "scrc_k0(const float* p0, float* p1, int p2)" in anonymous
    assert "scrc_k0(const float* p0, float* p1, int p2);" in anonymous


def test_cuda_frontend_reaches_the_canonical_lift_packet() -> None:
    request = prepare_scrc_lift(
        CUDA_SOURCE,
        compiler_view=_compiler_view(),
        source_symbol="vec_add",
        frontend="cuda",
    )
    payload = json.loads(request.user_input)

    assert "void scrc_k0(" in payload["anonymous_source"]
    assert [
        item["name"] for item in payload["harness_compiler_view"]["arguments"]
    ] == ["p0", "p1", "p2", "p3"]
    assert payload["harness_compiler_view"]["parameter_order"] == [
        "p0",
        "p1",
        "p2",
        "p3",
    ]
    assert payload["harness_compiler_view"]["grid_contract"] == [
        "ceil_div(p3,32)",
        "1",
        "1",
    ]
