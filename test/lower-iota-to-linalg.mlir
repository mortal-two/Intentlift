// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

func.func @iota_i32() -> tensor<17xi32> {
  %i = intent.iota : tensor<17xi32>
  return %i : tensor<17xi32>
}

// CHECK-LABEL: func.func @iota_i32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<17xi32>
// CHECK: %[[IOTA:.*]] = linalg.generic
// CHECK:   %[[INDEX:.*]] = linalg.index 0
// CHECK:   %[[VALUE:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK:   linalg.yield %[[VALUE]] : i32
// CHECK-NOT: intent.iota

// -----

func.func @iota_index() -> tensor<9xindex> {
  %i = intent.iota : tensor<9xindex>
  return %i : tensor<9xindex>
}

// CHECK-LABEL: func.func @iota_index
// CHECK: linalg.generic
// CHECK:   %[[INDEX2:.*]] = linalg.index 0
// CHECK-NEXT:   linalg.yield %[[INDEX2]] : index
// CHECK-NOT: intent.iota
