// RUN: intent-opt -lower-disjoint-memref-copy-to-loops %s | FileCheck %s

// CHECK-LABEL: func.func @copy_to_subview(
// CHECK-NOT: memref.copy
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
func.func @copy_to_subview(%output: memref<2x8xf32>) {
  %c0 = arith.constant 0 : index
  %source = memref.alloc() : memref<2x4xf32>
  %target = memref.subview %output[0, 0] [2, 4] [1, 1]
    : memref<2x8xf32> to memref<2x4xf32, strided<[8, 1]>>
  memref.copy %source, %target
    : memref<2x4xf32> to memref<2x4xf32, strided<[8, 1]>>
  memref.dealloc %source : memref<2x4xf32>
  return
}
