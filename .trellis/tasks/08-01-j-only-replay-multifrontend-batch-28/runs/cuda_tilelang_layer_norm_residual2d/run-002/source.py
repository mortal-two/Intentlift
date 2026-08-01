#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ bias, const float* __restrict__ inp, float* __restrict__ mean, float* __restrict__ out, const float* __restrict__ residual, float* __restrict__ rstd, const float* __restrict__ weight, int M);
extern "C" __global__ void __launch_bounds__(64, 1) main_kernel(const float* __restrict__ bias, const float* __restrict__ inp, float* __restrict__ mean, float* __restrict__ out, const float* __restrict__ residual, float* __restrict__ rstd, const float* __restrict__ weight, int M) {
  float x[64];
  float r[64];
  float w[64];
  float b[64];
  float z[64];
  float sm[1];
  float sq[64];
  float sm2[1];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(ulonglong4*)(x + (i * 8)) = tl::load_global_256(&(*(ulonglong4*)(inp + ((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)i) * (int64_t)8)))));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    *(ulonglong4*)(r + (i_1 * 8)) = tl::load_global_256(&(*(ulonglong4*)(residual + ((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)i_1) * (int64_t)8)))));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    *(ulonglong4*)(w + (i_2 * 8)) = tl::load_global_256(&(*(ulonglong4*)(weight + (i_2 * 8))));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    *(ulonglong4*)(b + (i_3 * 8)) = tl::load_global_256(&(*(ulonglong4*)(bias + (i_3 * 8))));
  }
  for (int i0 = 0; i0 < 64; ++i0) {
    z[i0] = (x[i0] + r[i0]);
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 64; ++rv) {
    sm[0] = (sm[0] + z[rv]);
  }
  float m0 = (sm[0] * 0x1p-6f/*1.562500e-02*/);
  mean[((int64_t)((int)blockIdx.x))] = m0;
  for (int i1 = 0; i1 < 64; ++i1) {
    float d = (z[i1] - m0);
    sq[i1] = (d * d);
  }
  sm2[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 64; ++rv_1) {
    sm2[0] = (sm2[0] + sq[rv_1]);
  }
  float var = (sm2[0] * 0x1p-6f/*1.562500e-02*/);
  float r0 = rsqrtf((var + 0x1.4f8b588e368f1p-17f/*1.000000e-05*/));
  rstd[((int64_t)((int)blockIdx.x))] = r0;
  for (int i2 = 0; i2 < 64; ++i2) {
    out[((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((int64_t)i2))] = ((((z[i2] - m0) * r0) * w[i2]) + b[i2]);
  }
}

