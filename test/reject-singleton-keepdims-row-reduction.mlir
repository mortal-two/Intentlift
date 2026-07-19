// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// Reducing a per-row [M,1] uniform while keeping [M,1] is an identity.  Until
// the uniform table represents that rank-2 view explicitly, fail closed rather
// than summing the same value N times.
// CHECK-LABEL: func.func @singleton_keepdims_identity
// CHECK: intent.reduce
// CHECK-NOT: gpu.launch
func.func @singleton_keepdims_identity(%X: tensor<2x4xf32>, %S: tensor<2xf32>)
    -> (tensor<2x4xf32>, tensor<2x1xf32>) {
  %one = arith.constant dense<1.0> : tensor<1xf32>
  %keep = intent.contraction %S, %one {subscripts = "i,j->ij"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  %sum = intent.reduce %keep {fn = "sum", axes = [1], assoc = "associative"} : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %Y = intent.map_binary %X, %S {fn = "add"} : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xf32>
  return %Y, %sum : tensor<2x4xf32>, tensor<2x1xf32>
}
