// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// The scalar row emitter evaluates a contraction at the current row/column.
// A no-reduction einsum that permutes output labels is therefore outside this
// fast path even when equal extents make the type look compatible.
// CHECK-LABEL: func.func @permuting_contraction(
// CHECK: intent.contraction
// CHECK-NOT: gpu.launch
func.func @permuting_contraction(%X: tensor<4x4xf32>, %scale: tensor<f32>)
    -> (tensor<4x4xf32>, tensor<4x1xf32>) {
  %Y = intent.contraction %X, %scale {subscripts = "ij,->ji"} : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x4xf32>) -> tensor<4x1xf32>
  return %Y, %sum : tensor<4x4xf32>, tensor<4x1xf32>
}
