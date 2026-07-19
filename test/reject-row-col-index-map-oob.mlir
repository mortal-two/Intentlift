// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// The legacy ColSeg compatibility map must be in bounds over the complete
// group/column rectangle.  This map reaches 301 for a length-4 parameter.
// CHECK-LABEL: func.func @col_index_map_oob
// CHECK: intent.map_binary
// CHECK-NOT: gpu.launch
func.func @col_index_map_oob(
    %X: tensor<4x4xf32>,
    %W: tensor<4xf32> {intent.col_index_map = affine_map<(d0, d1) -> (d0 + d1 * 100)>})
    -> (tensor<4x4xf32>, tensor<4x1xf32>) {
  %Y = intent.map_binary %X, %W {fn = "mul"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x4xf32>) -> tensor<4x1xf32>
  return %Y, %sum : tensor<4x4xf32>, tensor<4x1xf32>
}
