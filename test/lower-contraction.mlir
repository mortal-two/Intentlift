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

// -----

// Mixed-precision contraction keeps f16 storage but accumulates in f32.  The
// generic linalg body must extend both operands before multiply/add; otherwise
// the emitted arith.addf combines an f16 product with an f32 accumulator.
// CHECK-LABEL: func.func @mm_f16_acc_f32
// CHECK: linalg.generic
// CHECK: ^bb0(%[[A:.*]]: f16, %[[B:.*]]: f16, %[[ACC:.*]]: f32):
// CHECK: %[[AF:.*]] = arith.extf %[[A]] : f16 to f32
// CHECK: %[[BF:.*]] = arith.extf %[[B]] : f16 to f32
// CHECK: %[[PROD:.*]] = arith.mulf %[[AF]], %[[BF]] : f32
// CHECK: %[[SUM:.*]] = arith.addf %[[ACC]], %[[PROD]] : f32
// CHECK: linalg.yield %[[SUM]] : f32
func.func @mm_f16_acc_f32(%a: tensor<8x16xf16>, %b: tensor<8x16xf16>) -> tensor<8x8xf32> {
  %c = intent.contraction %a, %b {subscripts = "qd,kd->qk"} : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x8xf32>
  return %c : tensor<8x8xf32>
}

// -----

// No contracted label means direct broadcast multiplication.  A fill(+0) and
// add would change -0 to +0 and is not denotationally equivalent.
// CHECK-LABEL: func.func @row_scale
// CHECK-NOT: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: %[[PROD:.*]] = arith.mulf
// CHECK-NOT: arith.addf
// CHECK: linalg.yield %[[PROD]] : f32
func.func @row_scale(%x: tensor<2x3xf32>, %scale: tensor<2xf32>) -> tensor<2x3xf32> {
  %y = intent.contraction %x, %scale {subscripts = "mn,m->mn"} : (tensor<2x3xf32>, tensor<2xf32>) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

// Integer outer products are also legal no-reduction contractions and are
// needed by live extraction to materialize gather indices.
// CHECK-LABEL: func.func @index_column
// CHECK-NOT: linalg.fill
// CHECK: linalg.generic
// CHECK: %[[IPROD:.*]] = arith.muli
// CHECK-NOT: arith.addi
// CHECK: linalg.yield %[[IPROD]] : i64
func.func @index_column(%index: tensor<4xi64>, %one: tensor<1xi64>) -> tensor<4x1xi64> {
  %column = intent.contraction %index, %one {subscripts = "i,j->ij"} : (tensor<4xi64>, tensor<1xi64>) -> tensor<4x1xi64>
  return %column : tensor<4x1xi64>
}
