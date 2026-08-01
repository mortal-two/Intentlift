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
  float scores[16];
  float mx[1];
  float sm[1];
  float exp_scores[16];
  float dot_acc[1];
  float out_acc[1];
  float scale = sm_scale[0];
  for (int j0_1 = 0; j0_1 < 16; ++j0_1) {
    dot_acc[0] = 0x0p+0f/*0.000000e+00*/;
    for (int k0 = 0; k0 < 16; ++k0) {
      dot_acc[0] = (dot_acc[0] + (Q[((((int)blockIdx.x) * 16) + k0)] * K[((j0_1 * 16) + k0)]));
    }
    float s = (dot_acc[0] * scale);
    float condval;
    if ((((int)blockIdx.x) < j0_1)) {
      condval = -0x1.dcd65p+29f/*-1.000000e+09*/;
    } else {
      condval = s;
    }
    scores[j0_1] = condval;
  }
  mx[0] = -0x1.dcd65p+29f/*-1.000000e+09*/;
  for (int j1_1 = 0; j1_1 < 16; ++j1_1) {
    mx[0] = max(mx[0], scores[j1_1]);
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  for (int j2 = 0; j2 < 16; ++j2) {
    exp_scores[j2] = expf((scores[j2] - mx[0]));
    sm[0] = (sm[0] + exp_scores[j2]);
  }
  for (int d0 = 0; d0 < 16; ++d0) {
    out_acc[0] = 0x0p+0f/*0.000000e+00*/;
    for (int j3 = 0; j3 < 16; ++j3) {
      float prob = (exp_scores[j3] / sm[0]);
      out_acc[0] = (out_acc[0] + (prob * V[((j3 * 16) + d0)]));
    }
    Out[((((int)blockIdx.x) * 16) + d0)] = out_acc[0];
  }
}

