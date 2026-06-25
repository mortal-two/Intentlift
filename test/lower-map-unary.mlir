// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

// map_unary{exp} -> a single-input linalg.generic with math.exp in the body.
// CHECK-LABEL: func.func @ue
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: math.exp
func.func @ue(%a: tensor<256xf32>) -> tensor<256xf32> {
  %r = intent.map_unary %a {fn = "exp"} : (tensor<256xf32>) -> tensor<256xf32>
  return %r : tensor<256xf32>
}

// -----

// map_unary{neg} lowers to arith.negf (no transcendental).
// CHECK-LABEL: func.func @un
// CHECK: arith.negf
func.func @un(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %r = intent.map_unary %a {fn = "neg"} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}
