// RUN: intent-opt -convert-intent-elementwise-to-gpu=block-threads=4 %s | FileCheck %s --check-prefix=ELT
// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s --check-prefix=RR
// RUN: intent-opt -convert-intent-gather-to-gpu=block-threads=4 %s | FileCheck %s --check-prefix=GS

// The backward / gradient cluster (07-06 P4) — each *_backward is a composition
// of mechanisms IntentIR already lowers, NOT a new pass. Three groups, one
// representative each, proving the existing pass lowers the authored gradient:
//
//   1. pointwise activation backward = pure map      -> elementwise pass
//   2. softmax / norm backward = row-reduction + map -> row-reduction pass
//   3. scatter-grad backward = C3 scatter reduce=add -> gather pass

// ---------------------------------------------------------------------------- //
// Group 1: sigmoid_backward = grad * y*(1-y) — one map graph, no reduce/scatter.
// ELT-LABEL: func.func @sigmoid_backward(
// ELT:         gpu.launch
// ELT-NOT:     gpu.all_reduce
// ELT:         gpu.terminator
func.func @sigmoid_backward(%g: tensor<4x8xf32>, %y: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %yy = intent.map_binary %y, %y {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %d = intent.map_binary %y, %yy {fn = "sub"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %o = intent.map_binary %g, %d {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %o : tensor<4x8xf32>
}

// ---------------------------------------------------------------------------- //
// Group 2: softmax_backward = y*(g - sum(g*y)) — a row-reduction + map.
// RR-LABEL: func.func @softmax_backward(
// RR:         gpu.launch
// RR:         gpu.all_reduce
// RR:         gpu.terminator
func.func @softmax_backward(%g: tensor<4x8xf32>, %y: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %gy = intent.map_binary %g, %y {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %s = intent.reduce %gy {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %d = intent.map_binary %g, %s {fn = "sub"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %o = intent.map_binary %y, %d {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %o : tensor<4x8xf32>
}

// ---------------------------------------------------------------------------- //
// Group 3: gather_backward = zeros.scatter_add_(1, index, grad) — C3 scatter add.
// GS-LABEL: func.func @gather_backward(
// GS:         gpu.launch
// GS-NOT:     gpu.all_reduce
// GS:         memref.atomic_rmw addf
// GS:         gpu.terminator
func.func @gather_backward(%gself: tensor<4x8xf32>, %index: tensor<4x8xi64>, %grad: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %o = intent.scatter %gself, %index, %grad {axis = 1 : i64, mode = "along_axis", reduce = "add"} : (tensor<4x8xf32>, tensor<4x8xi64>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %o : tensor<4x8xf32>
}
