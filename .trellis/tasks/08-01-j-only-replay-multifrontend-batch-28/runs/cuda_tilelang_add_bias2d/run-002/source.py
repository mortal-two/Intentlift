#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ bias, const float* __restrict__ inp, float* __restrict__ out, int M);
extern "C" __global__ void __launch_bounds__(16, 1) main_kernel(const float* __restrict__ bias, const float* __restrict__ inp, float* __restrict__ out, int M) {
  out[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((int64_t)((int)threadIdx.x)))] = (inp[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((int64_t)((int)threadIdx.x)))] + bias[((int)threadIdx.x)]);
}

