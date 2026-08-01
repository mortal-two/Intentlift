#include <math_constants.h>
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
  float row[1];
  float mx[1];
  row[0] = inp[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) & (int64_t)15))];
  mx[0] = -CUDART_INF_F;
  mx[0] = max(mx[0], row[0]);
  mx[0] = tl::AllReduce<tl::MaxOp, 16, 1, 0, tl::NamedBarrier<128>>::run(mx[0]);
  out[((int64_t)((int)blockIdx.x))] = mx[0];
}

