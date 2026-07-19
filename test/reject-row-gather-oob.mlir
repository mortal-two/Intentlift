// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// Axis-1 gather index 4 is outside a width-4 row.  Constants are checked
// mechanically; the row path must not generate X[row,4].
// CHECK-LABEL: func.func @row_gather_oob
// CHECK: intent.gather
// CHECK-NOT: gpu.launch
func.func @row_gather_oob(%X: tensor<2x4xf32>)
    -> (tensor<2x1xf32>, tensor<2x1xf32>) {
  %index = arith.constant dense<4> : tensor<2x1xi32>
  %gathered = intent.gather %X, %index {axis = 1 : i64, mode = "along_axis"} : (tensor<2x4xf32>, tensor<2x1xi32>) -> tensor<2x1xf32>
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  return %gathered, %sum : tensor<2x1xf32>, tensor<2x1xf32>
}
