#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, float* __restrict__ C, const float* __restrict__ bias, int K, int M, int N);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, float* __restrict__ C, const float* __restrict__ bias, int K, int M, int N) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[32];
  float bias_local[8];
  float A_local[8];
  float B_local[8];
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int ko = 0; ko < ((K + 31) >> 5); ++ko) {
      mbarrier[((ko & 1) + 2)].wait((((ko & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(ko & 1)].expect_transaction(8192);
        tl::fence_proxy_async();
        tl::tma_load(A_desc, mbarrier[(ko & 1)], (&(((float*)buf_dyn_shmem)[((ko & 1) * 2048)])), (ko * 32), (((int)blockIdx.y) * 64));
        mbarrier[(ko & 1)].expect_transaction(8192);
        tl::fence_proxy_async();
        tl::tma_load(B_desc, mbarrier[(ko & 1)], (&(((float*)buf_dyn_shmem)[(((ko & 1) * 2048) + 4096)])), (((int)blockIdx.x) * 64), (ko * 32));
        tl::tma_load(B_desc, mbarrier[(ko & 1)], (&(((float*)buf_dyn_shmem)[(((ko & 1) * 2048) + 5120)])), ((((int)blockIdx.x) * 64) + 32), (ko * 32));
      }
      mbarrier[(ko & 1)].arrive();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int ko_1 = 0; ko_1 < ((K + 31) >> 5); ++ko_1) {
      mbarrier[(ko_1 & 1)].wait(((ko_1 & 3) >> 1));
      for (int ki = 0; ki < 4; ++ki) {
        for (int i_1 = 0; i_1 < 2; ++i_1) {
          tl::ptx_ldmatrix_x4((&(((float*)buf_dyn_shmem)[((((((ko_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (i_1 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])) + 0, A_local + (i_1 * 4));
        }
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          for (int j = 0; j < 4; ++j) {
            B_local[((i_2 * 4) + j)] = ((float*)buf_dyn_shmem)[(((((((((((ko_1 & 1) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (ki * 256)) + ((j & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + (((i_2 + (j & 1)) & 1) * 16)) + ((((j >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2)) + 4096)];
          }
        }
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          for (int j_1 = 0; j_1 < 2; ++j_1) {
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + ((i_3 * 16) + (j_1 * 8))), reinterpret_cast<const uint32_t*>(A_local + (i_3 * 4)), reinterpret_cast<const uint32_t*>(B_local + (j_1 * 4)));
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + (((i_3 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const uint32_t*>(A_local + (i_3 * 4)), reinterpret_cast<const uint32_t*>(B_local + ((j_1 * 4) + 2)));
          }
        }
      }
      mbarrier[((ko_1 & 1) + 2)].arrive();
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      for (int vec_s = 0; vec_s < 2; ++vec_s) {
        float condval;
        if (((((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 6) * 32)) + (i_4 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) < N)) {
          condval = bias[(((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((int64_t)i_4) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + ((int64_t)vec_s))];
        } else {
          condval = 0x0p+0f/*0.000000e+00*/;
        }
        bias_local[((i_4 * 2) + vec_s)] = condval;
      }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 32; ++i_5) {
      C_local[i_5] = max((C_local[i_5] + bias_local[((((i_5 & 15) >> 2) * 2) + (i_5 & 1))]), 0x0p+0f/*0.000000e+00*/);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 32; ++i_6) {
      if (((((((((int)blockIdx.y) * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_6 >> 4) * 16)) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < M) && ((((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_6 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) < N)) {
        C[((((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_6) & (int64_t)15) >> (int64_t)2) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)64) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)32)) + ((((int64_t)i_6) >> (int64_t)4) * (int64_t)16)) + (((((int64_t)i_6) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) * ((int64_t)N))) + (((int64_t)i_6) & (int64_t)1))] = C_local[i_6];
      }
    }
  }
}

