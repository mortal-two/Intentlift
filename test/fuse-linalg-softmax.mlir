// RUN: intent-opt '-convert-intent-to-linalg=reassociate-float-reductions=true' %s | intent-opt '-fuse-linalg-softmax=vector-width=4 scalable-vector=true require-fusion=true' -canonicalize -cse | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: intent.rvv.fused_softmax_count = 1 : i64
// CHECK-SAME: intent.rvv.fused_softmax_full_row_traversal_count = 3 : i64
// CHECK-SAME: intent.rvv.fused_softmax_input_traversal_count = 2 : i64
// CHECK-SAME: intent.rvv.fused_softmax_removed_materialization_count = 4 : i64
// CHECK-SAME: intent.rvv.fused_softmax_scalable_count = 1 : i64
// CHECK-SAME: intent.rvv.fused_softmax_scalar_reduction_count = 1 : i64
// CHECK-SAME: intent.rvv.fused_softmax_vector_reduction_count = 1 : i64
// CHECK-LABEL: func.func @softmax
// CHECK: scf.for
// CHECK: vector.transfer_read
// CHECK: vector.reduction <maxnumf>
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: tensor.insert
// CHECK: vector.transfer_read
// CHECK: arith.divf
// CHECK: vector.transfer_write
// CHECK-NOT: linalg.generic
// CHECK: return
func.func @softmax(%x: tensor<4x10xf32>) -> tensor<4x10xf32> {
  %maximum = intent.reduce %x {
    fn = "max", axes = [1], assoc = "associative"
  } : (tensor<4x10xf32>) -> tensor<4x1xf32>
  %shifted = intent.map_binary %x, %maximum {
    fn = "sub"
  } : (tensor<4x10xf32>, tensor<4x1xf32>) -> tensor<4x10xf32>
  %exponent = intent.map_unary %shifted {
    fn = "exp"
  } : (tensor<4x10xf32>) -> tensor<4x10xf32>
  %denominator = intent.reduce %exponent {
    fn = "sum", axes = [1], assoc = "float_nondeterministic"
  } : (tensor<4x10xf32>) -> tensor<4x1xf32>
  %output = intent.map_binary %exponent, %denominator {
    fn = "div"
  } : (tensor<4x10xf32>, tensor<4x1xf32>) -> tensor<4x10xf32>
  return %output : tensor<4x10xf32>
}
