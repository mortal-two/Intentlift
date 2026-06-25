// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

// Broadcasting map_binary{sub}: [4x8] - [4x1]. The [4x1] operand broadcasts on
// the last dim, so its indexing map pins that dim to constant 0 while the
// result/lhs maps are the identity.
// CHECK: affine_map<(d0, d1) -> (d0, 0)>
// CHECK-LABEL: func.func @bsub
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: arith.subf
func.func @bsub(%a: tensor<4x8xf32>, %m: tensor<4x1xf32>) -> tensor<4x8xf32> {
  %r = intent.map_binary %a, %m {fn = "sub"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// -----

// Equal-shape map_binary stays fully elementwise (identity maps, no constant 0).
// CHECK-LABEL: func.func @eqadd
// CHECK-NOT: -> (d0, 0)
// CHECK: arith.addf
func.func @eqadd(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %r = intent.map_binary %a, %b {fn = "add"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %r : tensor<8x16xf32>
}
