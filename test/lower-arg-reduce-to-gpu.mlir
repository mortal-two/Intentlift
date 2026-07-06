// RUN: intent-opt -convert-intent-arg-reduce-to-gpu=block-threads=4 %s | FileCheck %s

// C6 order-statistics arg-reduce: intent.arg_reduce (argmax/argmin) over a [M,N]
// row (axis 1) -> one block-cooperative gpu.launch returning the extremum's
// INTEGER INDEX. The block reduction carries a (val,idx) PAIR: a per-thread
// local best over K contiguous columns, then a power-of-two shared-memory tree
// keeping the extremum (ties -> lower index, torch's first-index rule), then
// thread 0 stores the winning i32 index. Two dynamic-shared buffers (val f32 +
// idx i32). M=4, N=8, T=4 (K=2).

// CHECK-LABEL: func.func @argmax_row(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4xi32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         memref.view %{{.*}} : {{.*}} to memref<4xf32
// CHECK:         memref.view %{{.*}} : {{.*}} to memref<4xi32
// CHECK:         arith.cmpf ogt
// CHECK:         arith.cmpi slt
// CHECK:         arith.select
// CHECK:         gpu.barrier
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xi32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @argmax_row(%x: tensor<4x8xf32>) -> tensor<4xi32> {
  %i = intent.arg_reduce %x {fn = "argmax", axis = 1 : i64} : (tensor<4x8xf32>) -> tensor<4xi32>
  return %i : tensor<4xi32>
}

// argmin uses the strict OLT compare (same pair-tree, opposite extremum).
// CHECK-LABEL: func.func @argmin_row(
// CHECK:         arith.cmpf olt
func.func @argmin_row(%x: tensor<4x8xf32>) -> tensor<4xi32> {
  %i = intent.arg_reduce %x {fn = "argmin", axis = 1 : i64} : (tensor<4x8xf32>) -> tensor<4xi32>
  return %i : tensor<4xi32>
}
