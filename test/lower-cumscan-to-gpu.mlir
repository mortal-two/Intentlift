// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// C6 cummax/cummin: intent.scan fn=max/min reuses the SAME Hillis-Steele block
// scan as fn=sum (C5/M4), only the combine changes (arith.maxnumf / minnumf and
// the -inf/+inf identity). A general monoid is not invertible, so the exclusive
// prefix is read as thread (tx-1)'s inclusive value (identity for tx==0) instead
// of the sum-only incl-segTotal subtraction. M=2, N=4, T=2 (K=2).

// CHECK-LABEL: func.func @cummax_row(
// CHECK-SAME:    %{{.*}}: memref<2x4xf32>, %{{.*}}: memref<2x4xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK-NOT:     gpu.all_reduce
// CHECK:         arith.maxnumf
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]{{.*}} : memref<2x4xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @cummax_row(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %y = intent.scan %x {fn = "max", axis = 1} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %y : tensor<2x4xf32>
}

// cummin uses the minnumf combine.
// CHECK-LABEL: func.func @cummin_row(
// CHECK:         arith.minnumf
func.func @cummin_row(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %y = intent.scan %x {fn = "min", axis = 1} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %y : tensor<2x4xf32>
}
