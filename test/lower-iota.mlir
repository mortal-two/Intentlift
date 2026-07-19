// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

// Backend-neutral iota materializes its sole semantic axis with linalg.index,
// then casts that index to the declared integer width.
// CHECK-LABEL: func.func @iota_i32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<7xi32>
// CHECK: %[[IOTA:.*]] = linalg.generic
// CHECK:   %[[INDEX:.*]] = linalg.index 0
// CHECK:   %[[VALUE:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK:   linalg.yield %[[VALUE]] : i32
// CHECK-NOT: intent.iota
func.func @iota_i32() -> tensor<7xi32> {
  %i = intent.iota : tensor<7xi32>
  return %i : tensor<7xi32>
}

// CHECK-LABEL: func.func @iota_i64
// CHECK: linalg.index 0
// CHECK: arith.index_cast {{.*}} : index to i64
// CHECK-NOT: intent.iota
func.func @iota_i64() -> tensor<5xi64> {
  %i = intent.iota : tensor<5xi64>
  return %i : tensor<5xi64>
}
