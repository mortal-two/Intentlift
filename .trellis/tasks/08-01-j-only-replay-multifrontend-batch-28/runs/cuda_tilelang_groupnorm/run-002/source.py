#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ B, float* __restrict__ Mean, float* __restrict__ Rstd, const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y, int N);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ B, float* __restrict__ Mean, float* __restrict__ Rstd, const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y, int N) {
  float x_tile[256];
  float w_tile[16];
  float b_tile[16];
  float sum_hw[16];
  float sum_all[1];
  float x2_tile[256];
  float sum2_hw[16];
  float sum2_all[1];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    *(ulonglong4*)(x_tile + (i * 8)) = tl::load_global_256(&(*(ulonglong4*)(X + (((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)blockIdx.y)) * (int64_t)256)) + (((int64_t)i) * (int64_t)8)))));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(ulonglong4*)(w_tile + (i_1 * 8)) = tl::load_global_256(&(*(ulonglong4*)(W + ((((int)blockIdx.y) * 16) + (i_1 * 8)))));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(ulonglong4*)(b_tile + (i_2 * 8)) = tl::load_global_256(&(*(ulonglong4*)(B + ((((int)blockIdx.y) * 16) + (i_2 * 8)))));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    sum_hw[i_3] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv = 0; rv < 16; ++rv) {
      sum_hw[i_3] = (sum_hw[i_3] + x_tile[((i_3 * 16) + rv)]);
    }
  }
  sum_all[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
    sum_all[0] = (sum_all[0] + sum_hw[rv_1]);
  }
  float mean = (sum_all[0] / 0x1p+8f/*2.560000e+02*/);
  for (int ci0 = 0; ci0 < 16; ++ci0) {
    for (int hi0 = 0; hi0 < 16; ++hi0) {
      float v = x_tile[((ci0 * 16) + hi0)];
      x2_tile[((ci0 * 16) + hi0)] = (v * v);
    }
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 16; ++i_4) {
    sum2_hw[i_4] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      sum2_hw[i_4] = (sum2_hw[i_4] + x2_tile[((i_4 * 16) + rv_2)]);
    }
  }
  sum2_all[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
    sum2_all[0] = (sum2_all[0] + sum2_hw[rv_3]);
  }
  float mean2 = (sum2_all[0] / 0x1p+8f/*2.560000e+02*/);
  float var = (mean2 - (mean * mean));
  float rstd = rsqrtf((var + 0x1.4f8b588e368f1p-17f/*1.000000e-05*/));
  Mean[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + ((int64_t)((int)blockIdx.y)))] = mean;
  Rstd[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + ((int64_t)((int)blockIdx.y)))] = rstd;
  for (int ci1 = 0; ci1 < 16; ++ci1) {
    for (int hi1 = 0; hi1 < 16; ++hi1) {
      Y[((((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)blockIdx.y)) * (int64_t)256)) + (((int64_t)ci1) * (int64_t)16)) + ((int64_t)hi1))] = ((((x_tile[((ci1 * 16) + hi1)] - mean) * rstd) * w_tile[ci1]) + b_tile[ci1]);
    }
  }
}

