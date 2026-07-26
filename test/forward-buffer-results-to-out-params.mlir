// RUN: intent-opt -forward-buffer-results-to-out-params %s | FileCheck %s

// CHECK-LABEL: func.func @result_buffer(
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: memref<8xf32>,
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<8xf32>)
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: memref.store {{.*}}, %[[OUT]]
func.func @result_buffer(%input: memref<8xf32>, %output: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %temporary = memref.alloc() : memref<8xf32>
  %value = memref.load %input[%c0] : memref<8xf32>
  memref.store %value, %temporary[%c0] : memref<8xf32>
  memref.copy %temporary, %output : memref<8xf32> to memref<8xf32>
  return
}
