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

// -----

// reduce{max} over ALL axes with keepdims (result [1x1], every dim reduced):
// a single all-reduction linalg.generic writing a constant-0 output index — no
// surviving dim, no expand_shape. This is the multi-axis / full-reduction case
// (amax / mse_loss / vector_norm / std / var_mean / count_nonzero).
// CHECK-LABEL: func.func @rall
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["reduction", "reduction"]
// CHECK: arith.maximumf
// CHECK-NOT: tensor.expand_shape
func.func @rall(%x: tensor<4x8xf32>) -> tensor<1x1xf32> {
  %m = intent.reduce %x {fn = "max", axes = [0, 1], assoc = "associative"} : (tensor<4x8xf32>) -> tensor<1x1xf32>
  return %m : tensor<1x1xf32>
}

// -----

// reduce{mean} over ALL axes with keepdims: the all-reduction generic followed
// by the 1/count elementwise scale, all over the [1x1] result.
// CHECK-LABEL: func.func @rmeanall
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["reduction", "reduction"]
// CHECK: arith.addf
// CHECK: arith.mulf
func.func @rmeanall(%x: tensor<4x8xf32>) -> tensor<1x1xf32> {
  %m = intent.reduce %x {fn = "mean", axes = [0, 1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<1x1xf32>
  return %m : tensor<1x1xf32>
}
