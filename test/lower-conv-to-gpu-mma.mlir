// RUN: intent-opt -convert-intent-conv-to-gpu-mma %s | FileCheck %s
// RUN: intent-opt -convert-intent-conv-to-gpu-mma='tf32=true' %s | FileCheck %s --check-prefix=TF32

// The tensor-core sibling of -convert-intent-conv-to-gpu. Lowers a forward
// conv2d as an IMPLICIT GEMM (M=N*OH*OW, N_gemm=Cout, K=Cin*kH*kW) onto the
// nvgpu.ldmatrix + nvgpu.mma.sync (m16n8k16) contraction: the im2col matrix is
// never materialized -- the OOB-guarded conv address math is emitted as the f16
// shared tile-fill (scf.if pad-zero predicate), the register-level contraction
// (ldmatrix A vector<4x2xf16> / W vector<2x2xf16> + mma.sync -> vector<2x2xf32>
// accumulators as scf.for iter_args) is reused verbatim, and the store scatters
// each accumulator element back to the NCHW output with a bias add.
// CHECK-LABEL: func.func @conv2d(
// CHECK-SAME:    %{{.*}}: memref<2x4x16x16xf32>, %{{.*}}: memref<8x4x3x3xf32>, %{{.*}}: memref<8xf32>, %{{.*}}: memref<2x8x14x14xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         scf.for
// CHECK-SAME:      iter_args
// CHECK:           scf.if
// CHECK:             arith.truncf
// CHECK:           memref.store
// CHECK:           gpu.barrier
// CHECK:           nvgpu.ldmatrix
// CHECK-SAME:        numTiles = 4
// CHECK:           nvgpu.ldmatrix
// CHECK-SAME:        numTiles = 2
// CHECK:           nvgpu.mma.sync
// CHECK-SAME:        mmaShape = [16, 8, 16]
// CHECK:           gpu.barrier
// CHECK:           scf.yield
// CHECK:         vector.extract
// CHECK:         arith.addf
// CHECK:         memref.store
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.

// TF32 (precise) input mode: operands stay f32 in smem (NO arith.truncf), the
// fragments are gathered by explicit memref.load + vector.insert (NO ldmatrix --
// it is 16-bit only), and the contraction is the m16n8k8 tf32Enabled mma.sync.
// TF32-LABEL: func.func @conv2d(
// TF32:         gpu.launch
// TF32:         scf.for
// TF32-SAME:      iter_args
// TF32:           scf.if
// TF32-NOT:         arith.truncf
// TF32:           memref.store
// TF32:           gpu.barrier
// TF32-NOT:       nvgpu.ldmatrix
// TF32:           vector.insert
// TF32:           nvgpu.mma.sync
// TF32-SAME:        mmaShape = [16, 8, 8]
// TF32-SAME:        tf32Enabled
// TF32:           gpu.barrier
// TF32:           scf.yield
func.func @conv2d(%x: tensor<2x4x16x16xf32>, %w: tensor<8x4x3x3xf32>,
                  %b: tensor<8xf32>) -> tensor<2x8x14x14xf32> {
  %o = intent.conv %x, %w, %b {stride = [1, 1], padding = [0, 0], dilation = [1, 1], groups = 1 : i64}
      : (tensor<2x4x16x16xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<2x8x14x14xf32>
  return %o : tensor<2x8x14x14xf32>
}

// -----

// Grouped conv (groups != 1) is declined by this pass (the naive
// -convert-intent-conv-to-gpu fallback handles it) -- the intent.conv is left
// untouched.
// CHECK-LABEL: func.func @conv2d_g2(
// CHECK:         intent.conv
func.func @conv2d_g2(%x: tensor<2x4x16x16xf32>, %w: tensor<8x2x3x3xf32>)
    -> tensor<2x8x14x14xf32> {
  %o = intent.conv %x, %w {stride = [1, 1], padding = [0, 0], dilation = [1, 1], groups = 2 : i64}
      : (tensor<2x4x16x16xf32>, tensor<8x2x3x3xf32>) -> tensor<2x8x14x14xf32>
  return %o : tensor<2x8x14x14xf32>
}
