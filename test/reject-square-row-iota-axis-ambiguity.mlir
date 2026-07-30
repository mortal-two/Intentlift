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

// A same-extent iota is not intrinsically ambiguous when its compiler-visible
// use proves a right-aligned predicate broadcast into [M,N].  This is the
// natural square LayerNorm mask: columns -> compare -> select(row, scalar).
// CHECK-LABEL: func.func @square_column_iota_select(
// CHECK: gpu.launch
// CHECK-NOT: intent.
func.func @square_column_iota_select(%X: tensor<4x4xf32>, %N: tensor<i32>)
    -> (tensor<4x4xf32>, tensor<4x1xf32>)
    attributes {intent.scalar_equalities = array<i64: 1, 4>} {
  %columns = intent.iota : tensor<4xi32>
  %valid = intent.map_binary %columns, %N {fn = "lt"}
      : (tensor<4xi32>, tensor<i32>) -> tensor<4xi1>
  %zero = arith.constant dense<0.0> : tensor<f32>
  %masked = intent.select %valid, %X, %zero
      : (tensor<4xi1>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  %sum = intent.reduce %masked
      {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<4x4xf32>) -> tensor<4x1xf32>
  return %masked, %sum : tensor<4x4xf32>, tensor<4x1xf32>
}
