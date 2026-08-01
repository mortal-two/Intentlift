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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C, int K, int M, int N);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C, int K, int M, int N) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  half_t A_local[32];
  half_t B_local[32];
  __shared__ __align__(16) uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int ko = 0; ko < ((K + 31) >> 5); ++ko) {
      mbarrier[((ko % 3) + 3)].wait((((ko % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(ko % 3)].expect_transaction(8192);
        tl::fence_proxy_async();
        tl::tma_load(A_desc, mbarrier[(ko % 3)], (&(((half_t*)buf_dyn_shmem)[((ko % 3) * 4096)])), (ko * 32), (((int)blockIdx.y) * 128));
        mbarrier[(ko % 3)].expect_transaction(8192);
        tl::fence_proxy_async();
        tl::tma_load(B_desc, mbarrier[(ko % 3)], (&(((half_t*)buf_dyn_shmem)[(((ko % 3) * 4096) + 12288)])), (((int)blockIdx.x) * 128), (ko * 32));
        tl::tma_load(B_desc, mbarrier[(ko % 3)], (&(((half_t*)buf_dyn_shmem)[(((ko % 3) * 4096) + 14336)])), ((((int)blockIdx.x) * 128) + 64), (ko * 32));
      }
      mbarrier[(ko % 3)].arrive();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int ko_1 = 0; ko_1 < ((K + 31) >> 5); ++ko_1) {
      mbarrier[(ko_1 % 3)].wait(((ko_1 % 6) / 3));
      for (int ki = 0; ki < 2; ++ki) {
        for (int i_1 = 0; i_1 < 4; ++i_1) {
          tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((ko_1 % 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_1 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])) + 0, A_local + (i_1 * 8));
        }
        for (int i_2 = 0; i_2 < 4; ++i_2) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((ko_1 % 3) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, B_local + (i_2 * 8));
        }
        for (int i_3 = 0; i_3 < 4; ++i_3) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_3 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_3 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
      mbarrier[((ko_1 % 3) + 3)].arrive();
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 128; ++i_4) {
      C_local[i_4] = max(C_local[i_4], 0x0p+0f/*0.000000e+00*/);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 128; ++i_5) {
      if (((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + ((i_5 >> 5) * 16)) + (((i_5 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < M) && ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_5 & 31) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) < N)) {
        C[((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + (((((int64_t)i_5) & (int64_t)31) >> (int64_t)2) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)64)) + ((((int64_t)i_5) >> (int64_t)5) * (int64_t)16)) + (((((int64_t)i_5) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) * ((int64_t)N))) + (((int64_t)i_5) & (int64_t)1))] = ((half_t)C_local[i_5]);
      }
    }
  }
}

