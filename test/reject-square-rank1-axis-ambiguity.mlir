// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// With M == N, a rank-1 f32 argument cannot be classified as per-row or
// per-column from shape alone.  Treating this row vector as a column vector
// would silently load R[col] instead of R[row].
// CHECK-LABEL: func.func @ambiguous_square_rank1(
// CHECK: intent.contraction
// CHECK-NOT: gpu.launch
func.func @ambiguous_square_rank1(%X: tensor<4x4xf32>, %R: tensor<4xf32>)
    -> (tensor<4x4xf32>, tensor<4x1xf32>) {
  %ones = arith.constant dense<1.0> : tensor<4xf32>
  %grid = intent.contraction %R, %ones {subscripts = "i,j->ij"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %Y = intent.map_binary %X, %grid {fn = "add"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x4xf32>) -> tensor<4x1xf32>
  return %Y, %sum : tensor<4x4xf32>, tensor<4x1xf32>
}

// A direct rank-1 operand of a rank-2 map follows Intent's right-aligned
// broadcast semantics, so its compiler-visible use proves Col[N] even when
// M == N. This is the square RMSNorm/LayerNorm weight pattern.
// CHECK-LABEL: func.func @square_trailing_broadcast(
// CHECK: gpu.launch
// CHECK-NOT: intent.
func.func @square_trailing_broadcast(%X: tensor<4x4xf32>,
                                     %W: tensor<4xf32>)
    -> tensor<4x4xf32> {
  %sq = intent.map_binary %X, %X {fn = "mul"}
      : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %mean = intent.reduce %sq
      {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<4x4xf32>) -> tensor<4x1xf32>
  %scaled = intent.map_binary %X, %mean {fn = "mul"}
      : (tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
  %Y = intent.map_binary %scaled, %W {fn = "mul"}
      : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  return %Y : tensor<4x4xf32>
}
