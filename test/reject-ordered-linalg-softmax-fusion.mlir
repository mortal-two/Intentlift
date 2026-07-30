// RUN: not intent-opt '-convert-intent-to-linalg=reassociate-float-reductions=true' '-fuse-linalg-softmax=vector-width=4 scalable-vector=true require-fusion=true' %s 2>&1 | FileCheck %s

// CHECK: required fused Softmax matched no canonical stable graph
func.func @ordered_softmax(%x: tensor<4x10xf32>) -> tensor<4x10xf32> {
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
    fn = "sum", axes = [1], assoc = "associative"
  } : (tensor<4x10xf32>) -> tensor<4x1xf32>
  %output = intent.map_binary %exponent, %denominator {
    fn = "div"
  } : (tensor<4x10xf32>, tensor<4x1xf32>) -> tensor<4x10xf32>
  return %output : tensor<4x10xf32>
}
