// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// B == G makes a length-2 iota ambiguous by extent.  Its einsum label proves
// that this one is the unsupported batch coordinate, not group = row % G.
// CHECK-LABEL: func.func @ambiguous_grouped_batch_iota
// CHECK: intent.iota
// CHECK-NOT: gpu.launch
func.func @ambiguous_grouped_batch_iota(%X: tensor<2x2x4xf32>)
    -> (tensor<2x2x4xf32>, tensor<2x2xf32>) {
  %sum = intent.reduce %X {fn = "sum", axes = [2], assoc = "float_nondeterministic"} : (tensor<2x2x4xf32>) -> tensor<2x2xf32>
  %batch = intent.iota : tensor<2xi32>
  %ones = arith.constant dense<1> : tensor<2x4xi32>
  %grid = intent.contraction %batch, %ones {subscripts = "b,gh->bgh"} : (tensor<2xi32>, tensor<2x4xi32>) -> tensor<2x2x4xi32>
  %gridf = intent.cast %grid {to_dtype = "f32"} : (tensor<2x2x4xi32>) -> tensor<2x2x4xf32>
  %Y = intent.map_binary %X, %gridf {fn = "add"} : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %Y, %sum : tensor<2x2x4xf32>, tensor<2x2xf32>
}
