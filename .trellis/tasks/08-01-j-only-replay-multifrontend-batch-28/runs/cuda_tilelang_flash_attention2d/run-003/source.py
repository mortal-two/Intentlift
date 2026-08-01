#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ K, float* __restrict__ Out, const float* __restrict__ Q, const float* __restrict__ V, const float* __restrict__ sm_scale);
extern "C" __global__ void main_kernel(const float* __restrict__ K, float* __restrict__ Out, const float* __restrict__ Q, const float* __restrict__ V, const float* __restrict__ sm_scale) {
  float m_i[1];
  float l_i[1];
  float acc[64];
  float scores[16];
  float dot_acc[1];
  float mx[1];
  float sm[1];
  float exp_scores[16];
  float value_acc[1];
  float scale = sm_scale[0];
  m_i[0] = -0x1.dcd65p+29f/*-1.000000e+09*/;
  l_i[0] = 0x0p+0f/*0.000000e+00*/;
  for (int d0 = 0; d0 < 64; ++d0) {
    acc[d0] = 0x0p+0f/*0.000000e+00*/;
  }
  for (int blk0 = 0; blk0 < 4; ++blk0) {
    for (int j0_1 = 0; j0_1 < 16; ++j0_1) {
      dot_acc[0] = 0x0p+0f/*0.000000e+00*/;
      for (int d1 = 0; d1 < 64; ++d1) {
        dot_acc[0] = (dot_acc[0] + (Q[((((int)blockIdx.x) * 64) + d1)] * K[(((blk0 * 1024) + (j0_1 * 64)) + d1)]));
      }
      float s = (dot_acc[0] * scale);
      float mf = ((float)(((blk0 * 16) + j0_1) <= ((int)blockIdx.x)));
      scores[j0_1] = ((s * mf) + (-0x1.dcd65p+29f/*-1.000000e+09*/ * (0x1p+0f/*1.000000e+00*/ - mf)));
    }
    mx[0] = -0x1.dcd65p+29f/*-1.000000e+09*/;
    for (int j1_1 = 0; j1_1 < 16; ++j1_1) {
      mx[0] = max(mx[0], scores[j1_1]);
    }
    float m_new = max(m_i[0], mx[0]);
    float alpha = expf((m_i[0] - m_new));
    sm[0] = 0x0p+0f/*0.000000e+00*/;
    for (int j2 = 0; j2 < 16; ++j2) {
      exp_scores[j2] = expf((scores[j2] - m_new));
      sm[0] = (sm[0] + exp_scores[j2]);
    }
    l_i[0] = ((l_i[0] * alpha) + sm[0]);
    for (int d2 = 0; d2 < 64; ++d2) {
      value_acc[0] = 0x0p+0f/*0.000000e+00*/;
      for (int j3 = 0; j3 < 16; ++j3) {
        value_acc[0] = (value_acc[0] + (exp_scores[j3] * V[(((blk0 * 1024) + (j3 * 64)) + d2)]));
      }
      acc[d2] = ((acc[d2] * alpha) + value_acc[0]);
    }
    m_i[0] = m_new;
  }
  for (int d3 = 0; d3 < 64; ++d3) {
    Out[((((int)blockIdx.x) * 64) + d3)] = (acc[d3] / l_i[0]);
  }
}

