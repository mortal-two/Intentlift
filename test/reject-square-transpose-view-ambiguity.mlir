// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// A square transpose has an unchanged type.  It may be real data movement,
// not the paired non-square LayerNorm reduction view that the row path can
// erase, so shape-only matching must fail closed.
// CHECK-LABEL: func.func @ambiguous_square_transpose(
// CHECK: intent.transpose
// CHECK-NOT: gpu.launch
func.func @ambiguous_square_transpose(%X: tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x1xf32>) {
  %Y = intent.transpose %X {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x4xf32>) -> tensor<4x1xf32>
  return %Y, %sum : tensor<4x4xf32>, tensor<4x1xf32>
}
