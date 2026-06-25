// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @add
// CHECK-NOT: intent.map_binary
// CHECK: %[[E:.*]] = tensor.empty() : tensor<8x16xf32>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME: ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<8x16xf32>)
// CHECK-SAME: outs(%[[E]] : tensor<8x16xf32>)
// CHECK: arith.addf
// CHECK: linalg.yield
func.func @add(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %c = intent.map_binary %a, %b {fn = "add"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %c : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @mul
// CHECK: arith.mulf
func.func @mul(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %c = intent.map_binary %a, %b {fn = "mul"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %c : tensor<4xf32>
}

// -----

// Row broadcast now lowers: the [4x1] operand pins its last dim to constant 0.
// CHECK-LABEL: func.func @row_broadcast
// CHECK-NOT: intent.map_binary
// CHECK: arith.subf
func.func @row_broadcast(%a: tensor<4x8xf32>, %b: tensor<4x1xf32>) -> tensor<4x8xf32> {
  %c = intent.map_binary %a, %b {fn = "sub"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  return %c : tensor<4x8xf32>
}
