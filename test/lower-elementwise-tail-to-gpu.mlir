// RUN: intent-opt -convert-intent-elementwise-to-gpu=block-threads=128 %s | FileCheck %s

// A non-power-of-two production row keeps one launch, redirects inactive loads
// to a safe in-bounds column, and guards the destination store.  The inactive
// lane cannot issue an out-of-bounds Col/Row access.

// CHECK-LABEL: func.func @tail_mul(
// CHECK-SAME: memref<2x127xf32>
// CHECK: %[[ACTIVE:.*]] = arith.cmpi ult
// CHECK: %[[SAFE:.*]] = arith.select %[[ACTIVE]]
// CHECK-NOT: scf.if
// CHECK: memref.load {{.*}}[%{{.*}}, %[[SAFE]]]
// CHECK: scf.if %[[ACTIVE]]
// CHECK: memref.store
// CHECK-NOT: intent.
func.func @tail_mul(%a: tensor<2x127xf32>, %b: tensor<2x127xf32>)
    -> tensor<2x127xf32> {
  %out = intent.map_binary %a, %b {fn = "mul"}
    : (tensor<2x127xf32>, tensor<2x127xf32>) -> tensor<2x127xf32>
  return %out : tensor<2x127xf32>
}
