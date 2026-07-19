// RUN: intent-opt -convert-intent-row-reduction-to-gpu="block-threads=2 tile-cols=1" %s | FileCheck %s

// The tiled rewrite does not implement compact [M,P] result storage.  It must
// fail closed instead of changing the ABI to memref<M> and omitting stores.
// CHECK-LABEL: func.func @tiled_vector_result(
// CHECK-SAME: tensor<2x4xf32>
// CHECK: intent.reduce
// CHECK: intent.concat
// CHECK-NOT: gpu.launch
func.func @tiled_vector_result(%X: tensor<2x4xf32>)
    -> (tensor<2x4xf32>, tensor<2x2xf32>) {
  %r = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %stats = intent.concat %r, %r {axis = 1 : i64} : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  return %X, %stats : tensor<2x4xf32>, tensor<2x2xf32>
}
