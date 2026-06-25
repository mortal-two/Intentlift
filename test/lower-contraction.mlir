// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @mm
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
func.func @mm(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>) -> tensor<8x8xf32> {
  %c = intent.contraction %a, %b {subscripts = "qd,kd->qk"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x8xf32>
  return %c : tensor<8x8xf32>
}

// -----

// Batched contraction -> two batch parallel dims + one reduction.
// CHECK-LABEL: func.func @bmm
// CHECK: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
func.func @bmm(%a: tensor<2x3x8x16xf32>, %b: tensor<2x3x8x16xf32>) -> tensor<2x3x8x8xf32> {
  %c = intent.contraction %a, %b {subscripts = "bhqd,bhkd->bhqk"} : (tensor<2x3x8x16xf32>, tensor<2x3x8x16xf32>) -> tensor<2x3x8x8xf32>
  return %c : tensor<2x3x8x8xf32>
}
