#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ hi, const float* __restrict__ inp, const float* __restrict__ lo, float* __restrict__ out, int M);
extern "C" __global__ void __launch_bounds__(16, 1) main_kernel(const float* __restrict__ hi, const float* __restrict__ inp, const float* __restrict__ lo, float* __restrict__ out, int M) {
  float lo_s = lo[0];
  float hi_s = hi[0];
  float y0_1 = max(inp[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((int64_t)((int)threadIdx.x)))], lo_s);
  float y1_1 = min(y0_1, hi_s);
  out[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((int64_t)((int)threadIdx.x)))] = y1_1;
}

