// RUN: intent-opt %s | FileCheck %s
// A contraction (matmul qd,kd->qk) at the intent level — lowers to a reduction
// kernel that executes on the 5090D (see test_backend_execute.py).
// CHECK-LABEL: func.func @mm
// CHECK: intent.contraction
func.func @mm(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>) -> tensor<8x8xf32> {
  %c = intent.contraction %a, %b {subscripts = "qd,kd->qk"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x8xf32>
  return %c : tensor<8x8xf32>
}
