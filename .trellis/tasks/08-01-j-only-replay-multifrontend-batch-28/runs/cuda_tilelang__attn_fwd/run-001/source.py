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

extern "C" __global__ void main_kernel(const float* __restrict__ K, float* __restrict__ Out, const float* __restrict__ Q, const float* __restrict__ V, const float* __restrict__ attn_mask, const float* __restrict__ sm_scale);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ K, float* __restrict__ Out, const float* __restrict__ Q, const float* __restrict__ V, const float* __restrict__ attn_mask, const float* __restrict__ sm_scale) {
  float q_row[16];
  float scores[16];
  float mx[1];
  float exp_scores[16];
  float sm[1];
  float probs[16];
  float out_row[16];
  float tmp[16];
  float acc[1];
  float tmp2[16];
  float acc2[1];
  for (int k = 0; k < 16; ++k) {
    q_row[k] = Q[((((int)blockIdx.x) * 16) + k)];
  }
  float scale = sm_scale[0];
  for (int j = 0; j < 16; ++j) {
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      tmp[k_1] = (q_row[k_1] * K[((j * 16) + k_1)]);
    }
    acc[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv = 0; rv < 16; ++rv) {
      acc[0] = (acc[0] + tmp[rv]);
    }
    scores[j] = ((acc[0] + attn_mask[((((int)blockIdx.x) * 16) + j)]) * scale);
  }
  mx[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
    mx[0] = max(mx[0], scores[rv_1]);
  }
  for (int j_1 = 0; j_1 < 16; ++j_1) {
    exp_scores[j_1] = expf((scores[j_1] - mx[0]));
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
    sm[0] = (sm[0] + exp_scores[rv_2]);
  }
  for (int j_2 = 0; j_2 < 16; ++j_2) {
    probs[j_2] = (exp_scores[j_2] / sm[0]);
  }
  for (int d = 0; d < 16; ++d) {
    for (int j_3 = 0; j_3 < 16; ++j_3) {
      tmp2[j_3] = (probs[j_3] * V[((j_3 * 16) + d)]);
    }
    acc2[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      acc2[0] = (acc2[0] + tmp2[rv_3]);
    }
    out_row[d] = acc2[0];
  }
  if (((int)threadIdx.x) == 0) {
    for (int d_1 = 0; d_1 < 16; ++d_1) {
      Out[((((int)blockIdx.x) * 16) + d_1)] = out_row[d_1];
    }
  }
}

