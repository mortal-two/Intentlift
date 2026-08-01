#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ inp, float* __restrict__ out, float* __restrict__ rstd, const float* __restrict__ weight, int M);
extern "C" __global__ void __launch_bounds__(64, 1) main_kernel(const float* __restrict__ inp, float* __restrict__ out, float* __restrict__ rstd, const float* __restrict__ weight, int M) {
  float row[64];
  float w[64];
  float sq[64];
  float sm[1];
  float out_row[64];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(ulonglong4*)(row + (i * 8)) = tl::load_global_256(&(*(ulonglong4*)(inp + ((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)i) * (int64_t)8)))));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    *(ulonglong4*)(w + (i_1 * 8)) = tl::load_global_256(&(*(ulonglong4*)(weight + (i_1 * 8))));
  }
  for (int r0 = 0; r0 < 64; ++r0) {
    sq[r0] = (row[r0] * row[r0]);
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 64; ++rv) {
    sm[0] = (sm[0] + sq[rv]);
  }
  float mean = (sm[0] * 0x1p-6f/*1.562500e-02*/);
  float r0_1 = rsqrtf((mean + 0x1.4f8b588e368f1p-17f/*1.000000e-05*/));
  rstd[((int64_t)((int)blockIdx.x))] = r0_1;
  for (int r1 = 0; r1 < 64; ++r1) {
    out_row[r1] = ((row[r1] * r0_1) * w[r1]);
  }
  out[((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((int64_t)((int)threadIdx.x)))] = out_row[((int)threadIdx.x)];
}

