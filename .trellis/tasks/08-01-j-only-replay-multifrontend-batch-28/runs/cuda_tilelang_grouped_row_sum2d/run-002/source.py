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
  float vals[1];
  float sm[1];
  vals[0] = inp[(((((int64_t)((int)blockIdx.y)) * (int64_t)64) + (((int64_t)((int)blockIdx.x)) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) & (int64_t)3))];
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  sm[0] = (sm[0] + vals[0]);
  sm[0] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(sm[0]);
  out[((((int64_t)((int)blockIdx.y)) * (int64_t)16) + ((int64_t)((int)blockIdx.x)))] = sm[0];
}

