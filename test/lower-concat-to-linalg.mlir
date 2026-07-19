// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

// Generic concat remains device-lowerable linalg work.  Using
// tensor.insert_slice here bufferizes to host memrefCopy calls, which are not
// part of the persisted CUDA launch graph.
// CHECK-LABEL: func.func @concat_columns(
// CHECK-NOT: intent.concat
// CHECK: tensor.extract_slice
// CHECK: linalg.generic
// CHECK: tensor.insert_slice
// CHECK: tensor.extract_slice
// CHECK: linalg.generic
// CHECK: tensor.insert_slice
func.func @concat_columns(%a: tensor<2x1xf32>, %b: tensor<2x2xf32>)
    -> tensor<2x3xf32> {
  %out = intent.concat %a, %b {axis = 1 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<2x3xf32>
  return %out : tensor<2x3xf32>
}
