// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// With G == N, extent alone cannot distinguish group from hidden-column.  This
// direct broadcast has no einsum-label proof, so the row path must fail closed.
// CHECK-LABEL: func.func @ambiguous_group_iota(
// CHECK: intent.iota
// CHECK-NOT: gpu.launch
func.func @ambiguous_group_iota(%X: tensor<1x2x2xf32>)
    -> (tensor<1x2x2xf32>, tensor<1x2xf32>) {
  %sum = intent.reduce %X {fn = "sum", axes = [2], assoc = "float_nondeterministic"} : (tensor<1x2x2xf32>) -> tensor<1x2xf32>
  %group = intent.iota : tensor<2xi32>
  %groupf = intent.cast %group {to_dtype = "f32"} : (tensor<2xi32>) -> tensor<2xf32>
  %Y = intent.map_binary %X, %groupf {fn = "add"} : (tensor<1x2x2xf32>, tensor<2xf32>) -> tensor<1x2x2xf32>
  return %Y, %sum : tensor<1x2x2xf32>, tensor<1x2xf32>
}
