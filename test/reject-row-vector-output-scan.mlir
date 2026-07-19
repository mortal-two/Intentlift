// RUN: intent-opt -convert-intent-row-reduction-to-gpu="block-threads=2 tile-cols=8" %s | FileCheck %s

// The scan rewrite currently owns full row outputs only.  A compact [M,P]
// side-result must keep the original graph for another path rather than being
// silently dropped from the generated kernel.
// CHECK-LABEL: func.func @scan_vector_result(
// CHECK-SAME: tensor<2x4xf32>
// CHECK: intent.scan
// CHECK: intent.concat
// CHECK-NOT: gpu.launch
func.func @scan_vector_result(%X: tensor<2x4xf32>)
    -> (tensor<2x4xf32>, tensor<2x2xf32>) {
  %scan = intent.scan %X {fn = "sum", axis = 1} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %r = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  %stats = intent.concat %r, %r {axis = 1 : i64} : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  return %scan, %stats : tensor<2x4xf32>, tensor<2x2xf32>
}
