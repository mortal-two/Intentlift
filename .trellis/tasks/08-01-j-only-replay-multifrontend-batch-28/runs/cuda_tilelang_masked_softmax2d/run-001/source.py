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

extern "C" __global__ void main_kernel(const float* __restrict__ inp, const signed char* __restrict__ mask, float* __restrict__ out, int M);
extern "C" __global__ void __launch_bounds__(64, 1) main_kernel(const float* __restrict__ inp, const signed char* __restrict__ mask, float* __restrict__ out, int M) {
  float row[64];
  float masked_row[64];
  float mx[1];
  float exp_row[64];
  float sm[1];
  float out_row[64];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(ulonglong4*)(row + (i * 8)) = tl::load_global_256(&(*(ulonglong4*)(inp + ((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)i) * (int64_t)8)))));
  }
  for (int r0 = 0; r0 < 64; ++r0) {
    float mf = ((float)((bool)mask[r0]));
    masked_row[r0] = ((row[r0] * mf) + (-0x1.dcd65p+29f/*-1.000000e+09*/ * (0x1p+0f/*1.000000e+00*/ - mf)));
  }
  mx[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 64; ++rv) {
    mx[0] = max(mx[0], masked_row[rv]);
  }
  for (int r1 = 0; r1 < 64; ++r1) {
    exp_row[r1] = expf((masked_row[r1] - mx[0]));
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 64; ++rv_1) {
    sm[0] = (sm[0] + exp_row[rv_1]);
  }
  for (int r2 = 0; r2 < 64; ++r2) {
    out_row[r2] = (exp_row[r2] / sm[0]);
  }
  out[((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((int64_t)((int)threadIdx.x)))] = out_row[((int)threadIdx.x)];
}

