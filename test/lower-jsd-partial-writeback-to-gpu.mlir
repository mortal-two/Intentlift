// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// A validated JSD-style partial writeback is expressed through a transposed
// row-domain select. The false branch preserves loss_init, so the lowered ABI
// reuses that input as output 0 and guards only its stores. dX remains a fresh
// output. Both outputs share the post-reduction `%shifted` DAG; computing the
// first output inside scf.if would make cached SSA values fail to dominate the
// second output, so this also protects the multi-output epilogue invariant.
// CHECK-LABEL: func.func @jsd_partial_writeback(
// CHECK-SAME: %{{.*}}: memref<4x8xf32>, %[[LOSS:.*]]: memref<4x8xf32>, %{{.*}}: memref<4xi64>, %[[DX:.*]]: memref<4x8xf32>)
// CHECK: gpu.launch
// CHECK: arith.cmpi ne
// CHECK: gpu.all_reduce{{.*}}max
// CHECK: scf.if
// CHECK: memref.store %{{.*}}, %[[LOSS]][%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK: memref.store %{{.*}}, %[[DX]][%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK: gpu.terminator
// CHECK-NOT: intent.
func.func @jsd_partial_writeback(
    %X: tensor<4x8xf32>,
    %loss_init: tensor<4x8xf32>,
    %label: tensor<4xi64>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>)
    attributes {intent.inplace = array<i64: 1, -1>} {
  %ignore = arith.constant dense<-100> : tensor<i64>
  %active = intent.map_binary %label, %ignore {fn = "ne"} :
      (tensor<4xi64>, tensor<i64>) -> tensor<4xi1>
  %row_max = intent.reduce %X {fn = "max", axes = [1], assoc = "associative"} :
      (tensor<4x8xf32>) -> tensor<4x1xf32>
  %shifted = intent.map_binary %X, %row_max {fn = "sub"} :
      (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %shifted_t = intent.transpose %shifted {perm = [1, 0]} :
      (tensor<4x8xf32>) -> tensor<8x4xf32>
  %loss_init_t = intent.transpose %loss_init {perm = [1, 0]} :
      (tensor<4x8xf32>) -> tensor<8x4xf32>
  %loss_selected_t = intent.select %active, %shifted_t, %loss_init_t :
      (tensor<4xi1>, tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
  %loss = intent.transpose %loss_selected_t {perm = [1, 0]} :
      (tensor<8x4xf32>) -> tensor<4x8xf32>
  %zero = arith.constant dense<0.0> : tensor<f32>
  %dx_selected_t = intent.select %active, %shifted_t, %zero :
      (tensor<4xi1>, tensor<8x4xf32>, tensor<f32>) -> tensor<8x4xf32>
  %dx = intent.transpose %dx_selected_t {perm = [1, 0]} :
      (tensor<8x4xf32>) -> tensor<4x8xf32>
  return %loss, %dx : tensor<4x8xf32>, tensor<4x8xf32>
}
