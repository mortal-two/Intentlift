// RUN: intent-opt -convert-intent-contraction-to-gpu %s | FileCheck %s

// A single intent.contraction{"mk,nk->mn"} (A[M,K] f16, W[N,K] f16 -> Y[M,N]
// f32) lowered to ONE tiled-K tensor-core gpu.launch: BM=BN=64, BK=16, a 4x4
// grid of m16n16k16 WMMA tiles over 4 warps. The per-warp fp32 accumulator
// fragments are carried across the K loop as scf.for iter_args; the A/W sub-tiles
// are staged through gpu.dynamic_shared_memory; the contraction is realized by
// gpu.subgroup_mma_compute (-> wmma.mma.sync on sm_120a), NOT a linalg.generic
// scalar MAC. The B operand is loaded transposed so "mk,nk->mn" is exact.
// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:    %{{.*}}: memref<128x64xf16>, %{{.*}}: memref<256x64xf16>, %{{.*}}: memref<128x256xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         gpu.subgroup_mma_constant_matrix
// CHECK:         scf.for
// CHECK-SAME:      iter_args
// CHECK:           gpu.barrier
// CHECK:           gpu.subgroup_mma_load_matrix
// CHECK:           gpu.subgroup_mma_compute
// CHECK:           scf.yield
// CHECK:         gpu.subgroup_mma_store_matrix
// CHECK:         gpu.terminator
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     intent.
func.func @matmul(%A: tensor<128x64xf16>, %W: tensor<256x64xf16>) -> tensor<128x256xf32> {
  %Y = intent.contraction %A, %W {subscripts = "mk,nk->mn"}
      : (tensor<128x64xf16>, tensor<256x64xf16>) -> tensor<128x256xf32>
  return %Y : tensor<128x256xf32>
}
