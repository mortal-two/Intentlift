#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ I, float* __restrict__ O);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ I, float* __restrict__ O) {
  O[((((int)blockIdx.x) * 8) + ((int)blockIdx.y))] = I[(((((int)blockIdx.x) >> 1) * 4) + (((int)blockIdx.y) >> 1))];
}

