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


def _record(
    shared_object: Path | str,
    *,
    source: str,
    source_identity: str,
    specialization: dict[str, object],
) -> dict[str, object]:
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
        "prebuilt_library": str(Path(shared_object)),
        "source_identity": source_identity,
        "specialization": specialization,
    }


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
    return _record(
        library,
        source=source,
        source_identity=source_identity,
        specialization={"columns": columns},
    )


def softmax_triton_cpu_rvv_record(
    shared_object: Path | str,
    *,
    columns: int,
    source_identity: str,
) -> dict[str, object]:
    """Bridge the accepted single-block Softmax Source to its J boundary."""

    if columns <= 0:
        raise ValueError("Triton-CPU Softmax columns must be positive")
    source = rf"""
#include <stdint.h>
#include <stdlib.h>

typedef struct {{ void *allocated, *aligned; int64_t offset; }} M0;
typedef struct {{
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
}} M2;
#define DATA(D, T) (((T *)((D)->aligned)) + (D)->offset)

extern void _softmax_single_block_forward_kernel(
    float *, int64_t, float *, int64_t, int32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

void {NATIVE_RVV_SYMBOL}(M2 *x_d, M0 *n_d, M2 *y_d) {{
  const int64_t rows = x_d->sizes[0], cols = x_d->sizes[1];
  if (rows > UINT32_MAX || cols != {columns} || cols > INT32_MAX ||
      DATA(n_d, int32_t)[0] != cols || y_d->sizes[0] != rows ||
      y_d->sizes[1] != cols || x_d->strides[1] != 1 ||
      y_d->strides[1] != 1) abort();
  float *x = DATA(x_d, float), *y = DATA(y_d, float);
  for (uint32_t row = 0; row < (uint32_t)rows; ++row)
    _softmax_single_block_forward_kernel(
        y, y_d->strides[0], x, x_d->strides[0], (int32_t)cols,
        row, 0, 0, (uint32_t)rows, 1, 1);
}}
"""
    return _record(
        shared_object,
        source=source,
        source_identity=source_identity,
        specialization={"columns": columns, "block_size": 1 << (columns - 1).bit_length()},
    )


def kldiv_triton_cpu_rvv_record(
    shared_object: Path | str,
    *,
    columns: int,
    epsilon: float,
    source_identity: str,
) -> dict[str, object]:
    """Bridge accepted log-target batchmean-row KLDiv to its J boundary."""

    if columns <= 0 or epsilon <= 0:
        raise ValueError("Triton-CPU KLDiv specialization is invalid")
    source = rf"""
#include <stdint.h>
#include <stdlib.h>

typedef struct {{
  void *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
}} M1;
typedef struct {{
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
}} M2;
#define DATA(D, T) (((T *)((D)->aligned)) + (D)->offset)

extern void _kldiv_kernel_forward(
    float *, int64_t, float *, int64_t, float *, int64_t, int32_t, float,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

void {NATIVE_RVV_SYMBOL}(M2 *y_d, M2 *gt_d, M1 *loss_d) {{
  const int64_t rows = y_d->sizes[0], cols = y_d->sizes[1];
  if (rows > UINT32_MAX || cols != {columns} || cols > INT32_MAX ||
      gt_d->sizes[0] != rows || gt_d->sizes[1] != cols ||
      loss_d->sizes[0] != rows || y_d->strides[1] != 1 ||
      gt_d->strides[1] != 1 || loss_d->strides[0] != 1) abort();
  float *y = DATA(y_d, float), *gt = DATA(gt_d, float);
  float *loss = DATA(loss_d, float);
  for (uint32_t row = 0; row < (uint32_t)rows; ++row)
    _kldiv_kernel_forward(
        y, y_d->strides[0], gt, gt_d->strides[0], loss,
        loss_d->strides[0], (int32_t)cols, {epsilon}f,
        row, 0, 0, (uint32_t)rows, 1, 1);
}}
"""
    return _record(
        shared_object,
        source=source,
        source_identity=source_identity,
        specialization={
            "columns": columns,
            "block_size": 64,
            "epsilon": epsilon,
            "log_target": True,
            "reduction": 3,
        },
    )


def flash_preprocess_triton_cpu_rvv_record(
    shared_object: Path | str,
    *,
    block_m: int,
    block_head_dim: int,
    source_identity: str,
) -> dict[str, object]:
    """Bridge the accepted Flash backward preprocessing Source."""

    if block_m <= 0 or block_head_dim <= 0:
        raise ValueError("Triton-CPU Flash preprocess blocks must be positive")
    source = rf"""
#include <stdint.h>
#include <stdlib.h>

typedef struct {{
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
}} M3;
typedef struct {{
  void *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
}} M4;
#define DATA(D, T) (((T *)((D)->aligned)) + (D)->offset)

extern void _bwd_preprocess_do_o_dot(
    uint16_t *, uint16_t *, float *,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int32_t, int32_t, int32_t, int32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

void {NATIVE_RVV_SYMBOL}(M4 *out_d, M4 *do_d, M3 *delta_d) {{
  const int64_t batch = out_d->sizes[0], heads = out_d->sizes[1];
  const int64_t sequence = out_d->sizes[2], head_dim = out_d->sizes[3];
  const int64_t rounded = delta_d->sizes[2];
  const int64_t grid_m = (sequence + {block_m} - 1) / {block_m};
  if (batch * heads > UINT32_MAX || grid_m > UINT32_MAX ||
      heads > INT32_MAX || sequence > INT32_MAX || rounded < sequence ||
      rounded > INT32_MAX || head_dim > {block_head_dim} ||
      do_d->sizes[0] != batch || do_d->sizes[1] != heads ||
      do_d->sizes[2] != sequence || do_d->sizes[3] != head_dim ||
      delta_d->sizes[0] != batch || delta_d->sizes[1] != heads) abort();
  uint16_t *out = DATA(out_d, uint16_t), *dout = DATA(do_d, uint16_t);
  float *delta = DATA(delta_d, float);
  for (uint32_t hb = 0; hb < (uint32_t)(batch * heads); ++hb)
    for (uint32_t start = 0; start < (uint32_t)grid_m; ++start)
      _bwd_preprocess_do_o_dot(
          out, dout, delta,
          out_d->strides[0], out_d->strides[1], out_d->strides[2],
          do_d->strides[0], do_d->strides[1], do_d->strides[2],
          (int32_t)heads, (int32_t)sequence, (int32_t)rounded,
          (int32_t)head_dim, start, hb, 0, (uint32_t)grid_m,
          (uint32_t)(batch * heads), 1);
}}
"""
    return _record(
        shared_object,
        source=source,
        source_identity=source_identity,
        specialization={"block_m": block_m, "block_head_dim": block_head_dim},
    )
