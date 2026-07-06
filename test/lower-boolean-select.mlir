// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

// C2c i1-boolean composability: the finiteness predicates + logical family
// (map_unary/map_binary -> i1), intent.select on the general linalg leg, and the
// all/any boolean reductions all lower to zero residual intent.* so the
// map+cmp+select composites (masked_fill / threshold / logical_*) reach a cubin.

// map_unary isinf/isfinite/logical_not -> i1 result via arith.cmpf/xori.
// CHECK-LABEL: func.func @finiteness
// CHECK: linalg.generic
// CHECK: arith.cmpf oeq
// CHECK: linalg.generic
// CHECK: arith.cmpf one
// CHECK: linalg.generic
// CHECK: arith.xori
// CHECK-NOT: intent.
func.func @finiteness(%x: tensor<4x8xf32>) -> tensor<4x8xi1> {
  %inf = intent.map_unary %x {fn = "isinf"} : (tensor<4x8xf32>) -> tensor<4x8xi1>
  %fin = intent.map_unary %x {fn = "isfinite"} : (tensor<4x8xf32>) -> tensor<4x8xi1>
  %not = intent.map_unary %fin {fn = "logical_not"} : (tensor<4x8xi1>) -> tensor<4x8xi1>
  %y = intent.map_binary %inf, %not {fn = "logical_or"} : (tensor<4x8xi1>, tensor<4x8xi1>) -> tensor<4x8xi1>
  return %y : tensor<4x8xi1>
}

// -----

// intent.select on the linalg leg: an i1 condition picks between two f32 tensors
// (a rank-0 scalar branch broadcasts). This is masked_fill = select(mask, v, x).
// CHECK-LABEL: func.func @masked_fill
// CHECK: linalg.generic
// CHECK: arith.select
// CHECK-NOT: intent.
func.func @masked_fill(%x: tensor<4x8xf32>, %m: tensor<4x8xi1>, %v: tensor<f32>)
    -> tensor<4x8xf32> {
  %y = intent.select %m, %v, %x : (tensor<4x8xi1>, tensor<f32>, tensor<4x8xf32>)
      -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}

// -----

// isclose -> i1 (|a-b| <= atol + rtol*|b|), and logical_and over i1.
// CHECK-LABEL: func.func @isclose_and
// CHECK: linalg.generic
// CHECK: arith.cmpf ole
// CHECK: linalg.generic
// CHECK: arith.andi
// CHECK-NOT: intent.
func.func @isclose_and(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>,
                       %m: tensor<4x8xi1>) -> tensor<4x8xi1> {
  %c = intent.map_binary %a, %b {fn = "isclose"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
  %y = intent.map_binary %c, %m {fn = "logical_and"} : (tensor<4x8xi1>, tensor<4x8xi1>) -> tensor<4x8xi1>
  return %y : tensor<4x8xi1>
}

// -----

// boolean reduce all/any over i1: and/or combine with a true/false identity.
// CHECK-LABEL: func.func @all_any
// CHECK-DAG: arith.constant true
// CHECK-DAG: arith.constant false
// CHECK: linalg.fill ins(%true
// CHECK: arith.andi
// CHECK: linalg.fill ins(%false
// CHECK: arith.ori
// CHECK-NOT: intent.
func.func @all_any(%m: tensor<4x8xi1>) -> (tensor<1x1xi1>, tensor<1x1xi1>) {
  %a = intent.reduce %m {fn = "all", axes = [0, 1], assoc = "associative"} : (tensor<4x8xi1>) -> tensor<1x1xi1>
  %o = intent.reduce %m {fn = "any", axes = [0, 1], assoc = "associative"} : (tensor<4x8xi1>) -> tensor<1x1xi1>
  return %a, %o : tensor<1x1xi1>, tensor<1x1xi1>
}
