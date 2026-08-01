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

extern "C" __global__ void main_kernel(const float* __restrict__ input_ptr, float* __restrict__ output_ptr, float* __restrict__ row_max_ptr, float* __restrict__ row_sum_ptr, int M);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ input_ptr, float* __restrict__ output_ptr, float* __restrict__ row_max_ptr, float* __restrict__ row_sum_ptr, int M) {
  float row[16];
  float mx[1];
  float exp_row[16];
  float sm[1];
  float out_row[16];
  for (int r = 0; r < 16; ++r) {
    row[r] = input_ptr[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((int64_t)r))];
  }
  mx[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 16; ++rv) {
    mx[0] = max(mx[0], row[rv]);
  }
  row_max_ptr[((int64_t)((int)blockIdx.x))] = mx[0];
  for (int r_1 = 0; r_1 < 16; ++r_1) {
    exp_row[r_1] = expf((row[r_1] - mx[0]));
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
    sm[0] = (sm[0] + exp_row[rv_1]);
  }
  row_sum_ptr[((int64_t)((int)blockIdx.x))] = sm[0];
  for (int r_2 = 0; r_2 < 16; ++r_2) {
    out_row[r_2] = (exp_row[r_2] / sm[0]);
  }
  for (int r_3 = 0; r_3 < 16; ++r_3) {
    output_ptr[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((int64_t)r_3))] = out_row[r_3];
  }
}

