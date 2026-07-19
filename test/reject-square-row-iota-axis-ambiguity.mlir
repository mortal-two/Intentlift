// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// With M == N, extent alone cannot distinguish a row-coordinate iota from a
// column-coordinate iota.  The row fast path must fail closed until IntentIR
// carries an explicit axis role.
// CHECK-LABEL: func.func @ambiguous_square_row_iota(
// CHECK: intent.iota
// CHECK-NOT: gpu.launch
func.func @ambiguous_square_row_iota(%X: tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x1xf32>) {
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x4xf32>) -> tensor<4x1xf32>
  %row = intent.iota : tensor<4xi32>
  %ones = arith.constant dense<1> : tensor<4xi32>
  %grid = intent.contraction %row, %ones {subscripts = "i,j->ij"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4x4xi32>
  %gridf = intent.cast %grid {to_dtype = "f32"} : (tensor<4x4xi32>) -> tensor<4x4xf32>
  %Y = intent.map_binary %X, %gridf {fn = "add"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %Y, %sum : tensor<4x4xf32>, tensor<4x1xf32>
}
