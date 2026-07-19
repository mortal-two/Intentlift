// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// `ii,j->ij` reads a diagonal.  The scalar row emitter cannot replace it with
// X[row,col], even though first-appearance label order happens to be `ij`.
// CHECK-LABEL: func.func @repeated_label_diagonal
// CHECK: intent.contraction
// CHECK-NOT: gpu.launch
func.func @repeated_label_diagonal(%X: tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x1xf32>) {
  %ones = arith.constant dense<1.0> : tensor<4xf32>
  %Y = intent.contraction %X, %ones {subscripts = "ii,j->ij"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x4xf32>) -> tensor<4x1xf32>
  return %Y, %sum : tensor<4x4xf32>, tensor<4x1xf32>
}
