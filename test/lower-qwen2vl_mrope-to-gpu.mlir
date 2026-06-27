// RUN: intent-opt -convert-intent-elementwise-to-gpu=block-threads=4 %s | FileCheck %s

// Qwen2-VL M-RoPE = the SAME rope rotation lowered by the SAME pass with NO
// per-kernel branch. The only addition is that cos/sin are each assembled from
// THREE position planes (temporal / height / width) selected by a per-column
// section mask: cos_row = tc*tm + hc*hm + wc*wm (the masks partition [0,hd/2),
// so the masked sum equals liger's `tl.load(..., mask, other=0)` plane sum).
// tc/hc/wc/ts/hs/ws are broadcast rows (one per token, strided<[hd,1]> half-view
// of the [.,.,hd] plane); tm/hm/wm are Col[N] {0,1} masks. Then the rotation is
// the identical half-view elementwise op as rope. M=8, hd=8, N=4, T=4.

// In-place (intent.inplace = [0,1]): y_lo/y_hi alias q_lo/q_hi; the 11 inputs are
// the only args (no fresh outputs) and the rotation is stored back through q.
// CHECK-LABEL: func.func @qwen2vl_mrope(
// CHECK-SAME:    %[[QLO:.*]]: memref<8x4xf32, strided<[8, 1]>>, %[[QHI:.*]]: memref<8x4xf32, strided<[8, 1]>>,
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     gpu.all_reduce
// CHECK-NOT:     scf.for
// q half-views + broadcast cos/sin planes (strided<[8,1]>) + Col masks.
// CHECK:         memref.load %[[QLO]][%{{.*}}, %{{.*}}] : memref<8x4xf32, strided<[8, 1]>>
// CHECK:         arith.divui %{{.*}}, %{{.*}} : index
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x4xf32, strided<[8, 1]>>
// CHECK:         memref.load %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK:         arith.subf
// CHECK:         memref.store %{{.*}}, %[[QLO]][%{{.*}}, %{{.*}}] : memref<8x4xf32, strided<[8, 1]>>
// CHECK:         gpu.terminator
func.func @qwen2vl_mrope(
    %q_lo: tensor<8x4xf32> {intent.strides = array<i64: 8, 1>},
    %q_hi: tensor<8x4xf32> {intent.strides = array<i64: 8, 1>},
    %tc: tensor<4x4xf32> {intent.strides = array<i64: 8, 1>},
    %hc: tensor<4x4xf32> {intent.strides = array<i64: 8, 1>},
    %wc: tensor<4x4xf32> {intent.strides = array<i64: 8, 1>},
    %ts: tensor<4x4xf32> {intent.strides = array<i64: 8, 1>},
    %hs: tensor<4x4xf32> {intent.strides = array<i64: 8, 1>},
    %ws: tensor<4x4xf32> {intent.strides = array<i64: 8, 1>},
    %tm: tensor<4xf32>, %hm: tensor<4xf32>, %wm: tensor<4xf32>)
    -> (tensor<8x4xf32>, tensor<8x4xf32>)
    attributes {intent.inplace = array<i64: 0, 1>} {
  %tcm = intent.map_binary %tc, %tm {fn = "mul"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %hcm = intent.map_binary %hc, %hm {fn = "mul"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %wcm = intent.map_binary %wc, %wm {fn = "mul"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %co1 = intent.map_binary %tcm, %hcm {fn = "add"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %cos = intent.map_binary %co1, %wcm {fn = "add"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %tsm = intent.map_binary %ts, %tm {fn = "mul"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %hsm = intent.map_binary %hs, %hm {fn = "mul"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %wsm = intent.map_binary %ws, %wm {fn = "mul"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %si1 = intent.map_binary %tsm, %hsm {fn = "add"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %sin = intent.map_binary %si1, %wsm {fn = "add"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %a = intent.map_binary %q_lo, %cos {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %b = intent.map_binary %q_hi, %sin {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %ylo = intent.map_binary %a, %b {fn = "sub"} : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
  %c = intent.map_binary %q_hi, %cos {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %d = intent.map_binary %q_lo, %sin {fn = "mul"} : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  %yhi = intent.map_binary %c, %d {fn = "add"} : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
  return %ylo, %yhi : tensor<8x4xf32>, tensor<8x4xf32>
}

// CHECK-NOT:     intent.
