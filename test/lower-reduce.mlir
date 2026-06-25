// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

// reduce{max} over axis 1 with keepdims (result rank == input rank): a
// linalg.generic reduction into a rank-1 temp, then tensor.expand_shape back
// to [4x1] so downstream row-broadcasts line up.
// CHECK-LABEL: func.func @rmax
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK: arith.maximumf
// CHECK: tensor.expand_shape
func.func @rmax(%x: tensor<4x8xf32>) -> tensor<4x1xf32> {
  %m = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  return %m : tensor<4x1xf32>
}

// -----

// reduce{sum} over axis 1, no keepdims (result rank < input rank): no expand.
// CHECK-LABEL: func.func @rsum
// CHECK: arith.addf
// CHECK-NOT: tensor.expand_shape
func.func @rsum(%x: tensor<4x8xf32>) -> tensor<4xf32> {
  %s = intent.reduce %x {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4xf32>
  return %s : tensor<4xf32>
}
