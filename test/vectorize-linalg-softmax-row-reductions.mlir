// RUN: intent-opt '-convert-intent-to-linalg=reassociate-float-reductions=true' %s | mlir-opt --linalg-fuse-elementwise-ops --canonicalize | intent-opt '-vectorize-linalg-row-reduction=vector-width=4 scalable-vector=true require-vectorization=true' -canonicalize -cse | FileCheck %s

// A canonical stable softmax has two different row monoids.  Full must lower
// both the associative max and the reassociable sum to real scalable-vector
// accumulators; leaving max as a scalar/materialized reduction hides the main
// target topology from search.
// CHECK: module attributes {
// CHECK-SAME: intent.rvv.explicit_vector_max_reduction_count = 1 : i64
// CHECK-SAME: intent.rvv.explicit_vector_reduction_count = 2 : i64
// CHECK-SAME: intent.rvv.explicit_vector_reduction_scalable_count = 2 : i64
// CHECK-LABEL: func.func @softmax
// CHECK: vector.transfer_read
// CHECK: vector.reduction <maxnumf>
// CHECK: vector.transfer_read
// CHECK: vector.reduction <add>
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
