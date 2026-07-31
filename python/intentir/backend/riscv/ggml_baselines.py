"""Strictly matched ggml/llama.cpp CPU-RVV external baseline adapters.

The adapters build a ggml graph and CPU compute plan on the first untimed call,
then update only external tensor pointers and execute the prebuilt plan.  They do
not consume IntentIR or reuse target objects.  A record exists only when the
ggml graph has the same public memref boundary as the accepted J workload.
"""

from __future__ import annotations

from intentir.backend.riscv.native_baselines import (
    EXTERNAL_RVV_BASELINE_PROTOCOL,
    NATIVE_RVV_SYMBOL,
)


_PREAMBLE = r"""
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include "ggml.h"
#include "ggml-cpu.h"

typedef struct { void *allocated, *aligned; int64_t offset; } M0;
typedef struct { void *allocated, *aligned; int64_t offset, sizes[1], strides[1]; } M1;
typedef struct { void *allocated, *aligned; int64_t offset, sizes[2], strides[2]; } M2;
#define DATA(D, T) (((T *)((D)->aligned)) + (D)->offset)

static void * checked_alloc(size_t bytes) {
  void *result = NULL;
  if (posix_memalign(&result, 64, bytes ? bytes : 64) != 0 || result == NULL)
    abort();
  return result;
}

static void run_plan(struct ggml_cgraph *graph, struct ggml_cplan *plan) {
  if (ggml_graph_compute(graph, plan) != GGML_STATUS_SUCCESS)
    abort();
}
"""


def _softmax() -> str:
    return (
        _PREAMBLE
        + r"""
static struct ggml_context *context;
static struct ggml_cgraph *graph;
static struct ggml_cplan plan;
static struct ggml_tensor *input_tensor;
static struct ggml_tensor *output_tensor;
static int64_t configured_rows;
static int64_t configured_width;

static void initialize(int64_t rows, int64_t width) {
  struct ggml_init_params params = {
    .mem_size = 16 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true,
  };
  context = ggml_init(params);
  if (context == NULL) abort();
  input_tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, rows);
  output_tensor = ggml_soft_max(context, input_tensor);
  graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, output_tensor);
  plan = ggml_graph_plan(graph, 1, NULL);
  if (plan.work_size) plan.work_data = (uint8_t *)checked_alloc(plan.work_size);
  configured_rows = rows;
  configured_width = width;
}

extern "C" void intentir_native_baseline(M2 *x_d, M0 *n_d, M2 *y_d) {
  int64_t rows = x_d->sizes[0], width = x_d->sizes[1];
  if ((int64_t)DATA(n_d, int32_t)[0] != width) abort();
  if (context == NULL) initialize(rows, width);
  if (rows != configured_rows || width != configured_width) abort();
  input_tensor->data = DATA(x_d, float);
  output_tensor->data = DATA(y_d, float);
  run_plan(graph, &plan);
}
"""
    )


def _dyt() -> str:
    return (
        _PREAMBLE
        + r"""
static struct ggml_context *context;
static struct ggml_cgraph *graph;
static struct ggml_cplan plan;
static struct ggml_tensor *input_tensor;
static struct ggml_tensor *gamma_tensor;
static struct ggml_tensor *beta_tensor;
static struct ggml_tensor *output_tensor;
static int64_t configured_rows;
static int64_t configured_width;
static float configured_alpha;

static void initialize(int64_t rows, int64_t width, float alpha) {
  struct ggml_init_params params = {
    .mem_size = 16 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true,
  };
  context = ggml_init(params);
  if (context == NULL) abort();
  input_tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, rows);
  gamma_tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
  beta_tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
  struct ggml_tensor *scaled = ggml_scale(context, input_tensor, alpha);
  struct ggml_tensor *activated = ggml_tanh(context, scaled);
  struct ggml_tensor *weighted = ggml_mul(context, activated, gamma_tensor);
  output_tensor = ggml_add(context, weighted, beta_tensor);
  size_t bytes = (size_t)rows * (size_t)width * sizeof(float);
  scaled->data = checked_alloc(bytes);
  activated->data = checked_alloc(bytes);
  weighted->data = checked_alloc(bytes);
  graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, output_tensor);
  plan = ggml_graph_plan(graph, 1, NULL);
  if (plan.work_size) plan.work_data = (uint8_t *)checked_alloc(plan.work_size);
  configured_rows = rows;
  configured_width = width;
  configured_alpha = alpha;
}

extern "C" void intentir_native_baseline(
    M2 *x_d, M1 *alpha_d, M1 *gamma_d, M1 *beta_d, M2 *y_d) {
  int64_t rows = x_d->sizes[0], width = x_d->sizes[1];
  float alpha = DATA(alpha_d, float)[0];
  if (gamma_d->sizes[0] != width || beta_d->sizes[0] != width) abort();
  if (context == NULL) initialize(rows, width, alpha);
  if (rows != configured_rows || width != configured_width ||
      alpha != configured_alpha) abort();
  input_tensor->data = DATA(x_d, float);
  gamma_tensor->data = DATA(gamma_d, float);
  beta_tensor->data = DATA(beta_d, float);
  output_tensor->data = DATA(y_d, float);
  run_plan(graph, &plan);
}
"""
    )


def _swiglu() -> str:
    return (
        _PREAMBLE
        + r"""
static struct ggml_context *context;
static struct ggml_cgraph *graph;
static struct ggml_cplan plan;
static struct ggml_tensor *a_tensor;
static struct ggml_tensor *b_tensor;
static struct ggml_tensor *output_tensor;
static int64_t configured_rows;
static int64_t configured_width;
static float configured_multiplier;

static void initialize(int64_t rows, int64_t width, float multiplier) {
  struct ggml_init_params params = {
    .mem_size = 16 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true,
  };
  context = ggml_init(params);
  if (context == NULL) abort();
  a_tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, rows);
  b_tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, rows);
  struct ggml_tensor *scaled = ggml_scale(context, a_tensor, multiplier);
  struct ggml_tensor *activated = ggml_silu(context, scaled);
  output_tensor = ggml_mul(context, activated, b_tensor);
  size_t bytes = (size_t)rows * (size_t)width * sizeof(float);
  scaled->data = checked_alloc(bytes);
  activated->data = checked_alloc(bytes);
  graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, output_tensor);
  plan = ggml_graph_plan(graph, 1, NULL);
  if (plan.work_size) plan.work_data = (uint8_t *)checked_alloc(plan.work_size);
  configured_rows = rows;
  configured_width = width;
  configured_multiplier = multiplier;
}

extern "C" void intentir_native_baseline(
    M2 *a_d, M2 *b_d, M0 *multiplier_d, M2 *output_d) {
  int64_t rows = a_d->sizes[0], width = a_d->sizes[1];
  float multiplier = DATA(multiplier_d, float)[0];
  if (b_d->sizes[0] != rows || b_d->sizes[1] != width) abort();
  if (context == NULL) initialize(rows, width, multiplier);
  if (rows != configured_rows || width != configured_width ||
      multiplier != configured_multiplier) abort();
  a_tensor->data = DATA(a_d, float);
  b_tensor->data = DATA(b_d, float);
  output_tensor->data = DATA(output_d, float);
  run_plan(graph, &plan);
}
"""
    )


def _xformers_rms_norm() -> str:
    return (
        _PREAMBLE
        + r"""
static struct ggml_context *context;
static struct ggml_cgraph *graph;
static struct ggml_cplan plan;
static struct ggml_tensor *input_tensor;
static struct ggml_tensor *weight_tensor;
static struct ggml_tensor *norm_tensor;
static struct ggml_tensor *output_tensor;
static void *norm_data;
static int64_t configured_rows;
static int64_t configured_width;
static float configured_eps;

static void initialize(int64_t rows, int64_t width, float eps) {
  struct ggml_init_params params = {
    .mem_size = 16 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true,
  };
  context = ggml_init(params);
  if (context == NULL) abort();
  input_tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, rows);
  weight_tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
  norm_tensor = ggml_rms_norm(context, input_tensor, eps);
  output_tensor = ggml_mul(context, norm_tensor, weight_tensor);
  norm_data = checked_alloc((size_t)rows * (size_t)width * sizeof(float));
  norm_tensor->data = norm_data;
  graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, output_tensor);
  plan = ggml_graph_plan(graph, 1, NULL);
  if (plan.work_size) plan.work_data = (uint8_t *)checked_alloc(plan.work_size);
  configured_rows = rows;
  configured_width = width;
  configured_eps = eps;
}

extern "C" void intentir_native_baseline(M2 *x_d, M1 *w_d, M0 *eps_d, M2 *y_d) {
  int64_t rows = x_d->sizes[0], width = x_d->sizes[1];
  float eps = DATA(eps_d, float)[0];
  if (context == NULL) initialize(rows, width, eps);
  if (rows != configured_rows || width != configured_width || eps != configured_eps)
    abort();
  input_tensor->data = DATA(x_d, float);
  weight_tensor->data = DATA(w_d, float);
  output_tensor->data = DATA(y_d, float);
  run_plan(graph, &plan);
}
"""
    )


_SOURCES = {
    "dyt": _dyt,
    "softmax": _softmax,
    "swiglu": _swiglu,
    "xformers_rms_norm": _xformers_rms_norm,
}


def ggml_rvv_baseline_record(kernel: str) -> dict[str, object]:
    try:
        source = _SOURCES[kernel]()
    except KeyError as exc:
        raise ValueError(f"no strictly matched ggml RVV baseline for {kernel!r}") from exc
    return {
        "protocol": EXTERNAL_RVV_BASELINE_PROTOCOL,
        "baseline_id": "ggml_rvv",
        "display_name": "ggml CPU/RVV",
        "kind": "mature_library",
        "symbol": NATIVE_RVV_SYMBOL,
        "source": source,
        "language": "c++",
        "build_profile": "ggml-rvv-strict",
        "implementation": "ggml-cpu-rvv-prebuilt-graph-plan",
        "strict_match": True,
    }


def maybe_ggml_rvv_baseline_records(kernel: str) -> tuple[dict[str, object], ...]:
    if kernel not in _SOURCES:
        return ()
    return (ggml_rvv_baseline_record(kernel),)


__all__ = ["ggml_rvv_baseline_record", "maybe_ggml_rvv_baseline_records"]
