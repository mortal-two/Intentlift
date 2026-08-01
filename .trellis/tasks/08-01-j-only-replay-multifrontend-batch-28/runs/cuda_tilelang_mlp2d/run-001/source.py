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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, float* __restrict__ C, __grid_constant__ const CUtensorMap W1_desc, __grid_constant__ const CUtensorMap W2_desc, const float* __restrict__ b1, const float* __restrict__ b2, int H, int K, int M, int N);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, float* __restrict__ C, __grid_constant__ const CUtensorMap W1_desc, __grid_constant__ const CUtensorMap W2_desc, const float* __restrict__ b1, const float* __restrict__ b2, int H, int K, int M, int N) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[32];
  float H_local[32];
  float b1_local[8];
  float b2_local[8];
  float A_local[8];
  float B_local[8];
  float A_local_1[8];
  float B_local_1[8];
  __shared__ __align__(16) uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(W1_desc);
    tl::prefetch_tma_descriptor(W2_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
    mbarrier[4].init(1);
    mbarrier[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int ho = 0; ho < ((H + 63) >> 6); ++ho) {
      for (int ko = 0; ko < ((K + 31) >> 5); ++ko) {
        mbarrier[((((ho * ((K + 31) >> 5)) + ko) & 1) + 2)].wait(((((((ho * ((K + 31) >> 5)) + ko) >> 1) + ((ho & 1) * ((K + 31) >> 5))) & 1) ^ 1));
        if (tl::tl_shuffle_elect<128>()) {
          mbarrier[(((ho * ((K + 31) >> 5)) + ko) & 1)].expect_transaction(8192);
          tl::fence_proxy_async();
          tl::tma_load(A_desc, mbarrier[(((ho * ((K + 31) >> 5)) + ko) & 1)], (&(((float*)buf_dyn_shmem)[((((ho * ((K + 31) >> 5)) + ko) & 1) * 2048)])), (ko * 32), (((int)blockIdx.y) * 64));
          mbarrier[(((ho * ((K + 31) >> 5)) + ko) & 1)].expect_transaction(8192);
          tl::fence_proxy_async();
          tl::tma_load(W1_desc, mbarrier[(((ho * ((K + 31) >> 5)) + ko) & 1)], (&(((float*)buf_dyn_shmem)[(((((ho * ((K + 31) >> 5)) + ko) & 1) * 2048) + 4096)])), (ho * 64), (ko * 32));
          tl::tma_load(W1_desc, mbarrier[(((ho * ((K + 31) >> 5)) + ko) & 1)], (&(((float*)buf_dyn_shmem)[(((((ho * ((K + 31) >> 5)) + ko) & 1) * 2048) + 5120)])), ((ho * 64) + 32), (ko * 32));
        }
        mbarrier[(((ho * ((K + 31) >> 5)) + ko) & 1)].arrive();
      }
      mbarrier[5].wait(((ho & 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[4].arrive_and_expect_tx(16384);
        tl::fence_proxy_async();
        tl::tma_load(W2_desc, mbarrier[4], (&(((float*)buf_dyn_shmem)[8192])), (((int)blockIdx.x) * 64), (ho * 64));
        tl::tma_load(W2_desc, mbarrier[4], (&(((float*)buf_dyn_shmem)[10240])), ((((int)blockIdx.x) * 64) + 32), (ho * 64));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int ho_1 = 0; ho_1 < ((H + 63) >> 6); ++ho_1) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(H_local + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
      }
      for (int ko_1 = 0; ko_1 < ((K + 31) >> 5); ++ko_1) {
        mbarrier[(((ho_1 * ((K + 31) >> 5)) + ko_1) & 1)].wait((((((ho_1 * ((K + 31) >> 5)) + ko_1) >> 1) + ((ho_1 & 1) * ((K + 31) >> 5))) & 1));
        for (int ki = 0; ki < 4; ++ki) {
          for (int i_2 = 0; i_2 < 2; ++i_2) {
            tl::ptx_ldmatrix_x4((&(((float*)buf_dyn_shmem)[((((((((ho_1 * ((K + 31) >> 5)) + ko_1) & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (i_2 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])) + 0, A_local + (i_2 * 4));
          }
          for (int i_3 = 0; i_3 < 2; ++i_3) {
            for (int j = 0; j < 4; ++j) {
              B_local[((i_3 * 4) + j)] = ((float*)buf_dyn_shmem)[(((((((((((((ho_1 * ((K + 31) >> 5)) + ko_1) & 1) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (ki * 256)) + ((j & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + (((i_3 + (j & 1)) & 1) * 16)) + ((((j >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2)) + 4096)];
            }
          }
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            for (int j_1 = 0; j_1 < 2; ++j_1) {
              tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(H_local + ((i_4 * 16) + (j_1 * 8))), reinterpret_cast<const uint32_t*>(A_local + (i_4 * 4)), reinterpret_cast<const uint32_t*>(B_local + (j_1 * 4)));
              tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(H_local + (((i_4 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const uint32_t*>(A_local + (i_4 * 4)), reinterpret_cast<const uint32_t*>(B_local + ((j_1 * 4) + 2)));
            }
          }
        }
        mbarrier[((((ho_1 * ((K + 31) >> 5)) + ko_1) & 1) + 2)].arrive();
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        for (int vec_s = 0; vec_s < 2; ++vec_s) {
          float condval;
          if (((((((ho_1 * 64) + ((((int)threadIdx.x) >> 6) * 32)) + (i_5 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) < H)) {
            condval = b1[(((((((int64_t)ho_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((int64_t)i_5) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + ((int64_t)vec_s))];
          } else {
            condval = 0x0p+0f/*0.000000e+00*/;
          }
          b1_local[((i_5 * 2) + vec_s)] = condval;
        }
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 32; ++i_6) {
        H_local[i_6] = max((H_local[i_6] + b1_local[((((i_6 & 15) >> 2) * 2) + (i_6 & 1))]), 0x0p+0f/*0.000000e+00*/);
      }
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_7 = 0; i_7 < 16; ++i_7) {
        *(float2*)(((float*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_7 >> 3) * 512)) + ((i_7 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_7 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_7 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 12288)) = *(float2*)(H_local + (i_7 * 2));
      }
      mbarrier[4].wait((ho_1 & 1));
      tl::__sync_thread_partial<3, 128>();
      for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          tl::ptx_ldmatrix_x4((&(((float*)buf_dyn_shmem)[(((((((ki_1 >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (i_8 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255)) + 12288)])) + 0, A_local_1 + (i_8 * 4));
        }
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          for (int j_2 = 0; j_2 < 4; ++j_2) {
            B_local_1[((i_9 * 4) + j_2)] = ((float*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 256)) + ((j_2 & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + (((i_9 + (j_2 & 1)) & 1) * 16)) + ((((j_2 >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2)) + 8192)];
          }
        }
        for (int i_10 = 0; i_10 < 2; ++i_10) {
          for (int j_3 = 0; j_3 < 2; ++j_3) {
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + ((i_10 * 16) + (j_3 * 8))), reinterpret_cast<const uint32_t*>(A_local_1 + (i_10 * 4)), reinterpret_cast<const uint32_t*>(B_local_1 + (j_3 * 4)));
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + (((i_10 * 16) + (j_3 * 8)) + 4)), reinterpret_cast<const uint32_t*>(A_local_1 + (i_10 * 4)), reinterpret_cast<const uint32_t*>(B_local_1 + ((j_3 * 4) + 2)));
          }
        }
      }
      mbarrier[5].arrive();
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      for (int vec_s_1 = 0; vec_s_1 < 2; ++vec_s_1) {
        float condval_1;
        if (((((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 6) * 32)) + (i_11 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_1) < N)) {
          condval_1 = b2[(((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + ((int64_t)vec_s_1))];
        } else {
          condval_1 = 0x0p+0f/*0.000000e+00*/;
        }
        b2_local[((i_11 * 2) + vec_s_1)] = condval_1;
      }
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 32; ++i_12) {
      C_local[i_12] = (C_local[i_12] + b2_local[((((i_12 & 15) >> 2) * 2) + (i_12 & 1))]);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 32; ++i_13) {
      if (((((((((int)blockIdx.y) * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_13 >> 4) * 16)) + (((i_13 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < M) && ((((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_13 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_13 & 1)) < N)) {
        C[((((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_13) & (int64_t)15) >> (int64_t)2) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)64) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)32)) + ((((int64_t)i_13) >> (int64_t)4) * (int64_t)16)) + (((((int64_t)i_13) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) * ((int64_t)N))) + (((int64_t)i_13) & (int64_t)1))] = C_local[i_13];
      }
    }
  }
}

