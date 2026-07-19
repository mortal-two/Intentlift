// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// A parameter-vector take gather with a statically out-of-bounds index must
// fail closed.  Producing one launch here would turn malformed extracted JSON
// into an unchecked global load.
// CHECK-LABEL: func.func @oob_parameter_gather(
// CHECK: intent.gather
// CHECK-NOT: gpu.launch
func.func @oob_parameter_gather(%X: tensor<2x4xf32>, %W: tensor<3xf32>)
    -> (tensor<2x4xf32>, tensor<2x1xf32>) {
  %r = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %index = arith.constant dense<3> : tensor<i32>
  %w = intent.gather %W, %index {axis = 0 : i64, mode = "take"} : (tensor<3xf32>, tensor<i32>) -> tensor<f32>
  %Y = intent.map_binary %X, %w {fn = "mul"} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
  return %Y, %r : tensor<2x4xf32>, tensor<2x1xf32>
}
