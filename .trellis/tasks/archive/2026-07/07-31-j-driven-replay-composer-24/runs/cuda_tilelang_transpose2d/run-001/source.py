#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ inp, float* __restrict__ out, int M);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ inp, float* __restrict__ out, int M) {
  float row[16];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(ulonglong4*)(row + (i * 8)) = tl::load_global_256(&(*(ulonglong4*)(inp + ((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)i) * (int64_t)8)))));
  }
  for (int r = 0; r < 16; ++r) {
    out[((((int64_t)r) * ((int64_t)M)) + ((int64_t)((int)blockIdx.x)))] = row[r];
  }
}

