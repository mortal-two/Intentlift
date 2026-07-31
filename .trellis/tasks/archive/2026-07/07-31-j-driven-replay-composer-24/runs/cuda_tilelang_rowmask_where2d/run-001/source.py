#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ inp, float* __restrict__ out, const signed char* __restrict__ row_mask, int M);
extern "C" __global__ void __launch_bounds__(64, 1) main_kernel(const float* __restrict__ inp, float* __restrict__ out, const signed char* __restrict__ row_mask, int M) {
  bool m = ((bool)row_mask[((int64_t)((int)blockIdx.x))]);
  float mf = ((float)m);
  out[((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((int64_t)((int)threadIdx.x)))] = (inp[((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((int64_t)((int)threadIdx.x)))] * mf);
}

