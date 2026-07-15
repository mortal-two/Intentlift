// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

func.func @embedding(%weight: tensor<97x13xf32>, %index: tensor<17xi64>)
    -> tensor<17x13xf32> {
  %g = intent.gather %weight, %index {axis = 0 : i64, mode = "take"}
      : (tensor<97x13xf32>, tensor<17xi64>) -> tensor<17x13xf32>
  return %g : tensor<17x13xf32>
}

// CHECK-LABEL: func.func @embedding
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<17x13xf32>
// CHECK: %[[G:.*]] = linalg.generic
// CHECK:   %[[ROW:.*]] = linalg.index 0
// CHECK:   %[[COL:.*]] = linalg.index 1
// CHECK:   %[[RAW:.*]] = tensor.extract %arg1[%[[ROW]]] : tensor<17xi64>
// CHECK:   %[[IDX:.*]] = arith.index_cast %[[RAW]] : i64 to index
// CHECK:   %[[VALUE:.*]] = tensor.extract %arg0[%[[IDX]], %[[COL]]] : tensor<97x13xf32>
// CHECK:   linalg.yield %[[VALUE]] : f32
// CHECK-NOT: intent.gather

func.func @along_axis(%data: tensor<4x7xf32>, %index: tensor<4x3xi32>)
    -> tensor<4x3xf32> {
  %g = intent.gather %data, %index {axis = 1 : i64, mode = "along_axis"}
      : (tensor<4x7xf32>, tensor<4x3xi32>) -> tensor<4x3xf32>
  return %g : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @along_axis
// CHECK: linalg.generic
// CHECK:   %[[R:.*]] = linalg.index 0
// CHECK:   %[[C:.*]] = linalg.index 1
// CHECK:   %[[RAW2:.*]] = tensor.extract %arg1[%[[R]], %[[C]]] : tensor<4x3xi32>
// CHECK:   %[[IDX2:.*]] = arith.index_cast %[[RAW2]] : i32 to index
// CHECK:   %[[VALUE2:.*]] = tensor.extract %arg0[%[[R]], %[[IDX2]]] : tensor<4x7xf32>
// CHECK:   linalg.yield %[[VALUE2]] : f32
// CHECK-NOT: intent.gather
