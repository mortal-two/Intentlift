#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const int* __restrict__ col_idx, const float* __restrict__ inp, float* __restrict__ out, const int* __restrict__ row_idx, int M);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const int* __restrict__ col_idx, const float* __restrict__ inp, float* __restrict__ out, const int* __restrict__ row_idx, int M) {
  int r = row_idx[((int)blockIdx.x)];
  int c = col_idx[((int)blockIdx.x)];
  float condval;
  if (((((0 <= c) && (c < 64)) && (0 <= r)) && (r < M))) {
    condval = inp[((((int64_t)r) * (int64_t)64) + ((int64_t)c))];
  } else {
    condval = 0x0p+0f/*0.000000e+00*/;
  }
  out[((int)blockIdx.x)] = condval;
}

