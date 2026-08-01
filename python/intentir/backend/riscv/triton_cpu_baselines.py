"""Adapters for precompiled, same-source Triton-CPU kernels on RVV.

The adapter contains only an ABI bridge from the benchmark memref boundary to
the raw function exported by Triton-CPU.  The target computation remains in the
shared object produced from the original Triton Source.
"""

from __future__ import annotations

from pathlib import Path

from intentir.backend.riscv.native_baselines import (
    EXTERNAL_RVV_BASELINE_PROTOCOL,
    NATIVE_RVV_SYMBOL,
)


TRITON_CPU_RVV_BASELINE_ID = "triton_cpu_rvv"


def swiglu_triton_cpu_rvv_record(
    shared_object: Path | str,
    *,
    columns: int,
    source_identity: str,
) -> dict[str, object]:
    """Return the strict SwiGLU adapter for one specialized Triton object."""

    if columns <= 0:
        raise ValueError("Triton-CPU SwiGLU columns must be positive")
    library = Path(shared_object)
    source = rf"""
#include <stdint.h>
#include <stdlib.h>

typedef struct {{ void *allocated, *aligned; int64_t offset; }} M0;
typedef struct {{
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
}} M2;

#define DATA(D, T) (((T *)((D)->aligned)) + (D)->offset)

extern void _swiglu_forward_kernel(
    float *, float *, float *, int64_t, float,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

void {NATIVE_RVV_SYMBOL}(M2 *a_d, M2 *b_d, M0 *mult_d, M2 *out_d) {{
  const int64_t rows = a_d->sizes[0];
  const int64_t cols = a_d->sizes[1];
  if (cols != {columns} || b_d->sizes[0] != rows || b_d->sizes[1] != cols ||
      out_d->sizes[0] != rows || out_d->sizes[1] != cols ||
      a_d->strides[1] != 1 || b_d->strides[1] != 1 || out_d->strides[1] != 1 ||
      a_d->strides[0] != cols || b_d->strides[0] != cols ||
      out_d->strides[0] != cols || rows > UINT32_MAX) abort();
  float *a = DATA(a_d, float);
  float *b = DATA(b_d, float);
  float *out = DATA(out_d, float);
  const float multiplier = DATA(mult_d, float)[0];
  for (uint32_t row = 0; row < (uint32_t)rows; ++row)
    _swiglu_forward_kernel(
        a, b, out, cols, multiplier, row, 0, 0, (uint32_t)rows, 1, 1);
}}
"""
    return {
        "protocol": EXTERNAL_RVV_BASELINE_PROTOCOL,
        "baseline_id": TRITON_CPU_RVV_BASELINE_ID,
        "display_name": "Triton-CPU/RVV",
        "kind": "source_portability",
        "symbol": NATIVE_RVV_SYMBOL,
        "source": source,
        "language": "c",
        "build_profile": "triton-cpu-rvv-prebuilt-strict",
        "implementation": "same-source-triton-cpu-rvv",
        "strict_match": True,
        "prebuilt_library": str(library),
        "source_identity": source_identity,
        "specialized_columns": columns,
    }

