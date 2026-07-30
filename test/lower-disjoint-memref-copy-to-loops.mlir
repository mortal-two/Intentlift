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

// A bufferized tensor identity/copy has no local allocation: both source and
// destination are ABI arguments with dynamic strided layouts.  It must still
// become self-contained loops rather than the external memrefCopy runtime.
// CHECK-LABEL: func.func @copy_between_arguments(
// CHECK-NOT: memref.copy
// CHECK: memref.dim
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
func.func @copy_between_arguments(
    %source: memref<?xf32, strided<[?], offset: ?>>,
    %target: memref<?xf32, strided<[?], offset: ?>>) {
  memref.copy %source, %target
    : memref<?xf32, strided<[?], offset: ?>>
      to memref<?xf32, strided<[?], offset: ?>>
  return
}
