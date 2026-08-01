#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ bias_ptr, const float* __restrict__ in_ptr, float* __restrict__ out_mean_ptr, float* __restrict__ out_ptr, float* __restrict__ out_rstd_ptr, const float* __restrict__ weight_ptr, int M);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ bias_ptr, const float* __restrict__ in_ptr, float* __restrict__ out_mean_ptr, float* __restrict__ out_ptr, float* __restrict__ out_rstd_ptr, const float* __restrict__ weight_ptr, int M) {
  float row[16];
  float w[16];
  float b[16];
  float s[1];
  float diff[16];
  float sq[16];
  float ss[1];
  float out_row[16];
  for (int r = 0; r < 16; ++r) {
    row[r] = in_ptr[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((int64_t)r))];
  }
  for (int r_1 = 0; r_1 < 16; ++r_1) {
    w[r_1] = weight_ptr[r_1];
  }
  for (int r_2 = 0; r_2 < 16; ++r_2) {
    b[r_2] = bias_ptr[r_2];
  }
  s[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 16; ++rv) {
    s[0] = (s[0] + row[rv]);
  }
  float mean = (s[0] / 0x1p+4f/*1.600000e+01*/);
  out_mean_ptr[((int64_t)((int)blockIdx.x))] = mean;
  for (int r_3 = 0; r_3 < 16; ++r_3) {
    float d = (row[r_3] - mean);
    diff[r_3] = d;
    sq[r_3] = (d * d);
  }
  ss[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
    ss[0] = (ss[0] + sq[rv_1]);
  }
  float var = (ss[0] / 0x1p+4f/*1.600000e+01*/);
  float rstd = rsqrtf((var + 0x1.4f8b588e368f1p-17f/*1.000000e-05*/));
  out_rstd_ptr[((int64_t)((int)blockIdx.x))] = rstd;
  for (int r_4 = 0; r_4 < 16; ++r_4) {
    out_row[r_4] = (((diff[r_4] * rstd) * w[r_4]) + b[r_4]);
  }
  for (int r_5 = 0; r_5 < 16; ++r_5) {
    out_ptr[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((int64_t)r_5))] = out_row[r_5];
  }
}

