// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=128 %s | FileCheck %s

// The counted KLDiv inner JIT returns one scalar per row.  Its sole reduction
// is consumed only by the block-leader store, so the compact hierarchy needs
// one publish barrier and no shared-memory rebroadcast barrier.
// CHECK-LABEL: func.func @kl_div_per_row(
// CHECK:         gpu.launch
// CHECK:         math.log
// CHECK-COUNT-1: gpu.barrier
// CHECK-NOT:     memref.atomic_rmw
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @kl_div_per_row(%y: tensor<4x4096xf32>,
                          %gt: tensor<4x4096xf32>,
                          %eps: tensor<f32>) -> tensor<4xf32> {
  %clamped = intent.map_binary %gt, %eps {fn = "max"} : (tensor<4x4096xf32>, tensor<f32>) -> tensor<4x4096xf32>
  %log_gt = intent.map_unary %clamped {fn = "log"} : (tensor<4x4096xf32>) -> tensor<4x4096xf32>
  %ratio = intent.map_binary %log_gt, %y {fn = "sub"} : (tensor<4x4096xf32>, tensor<4x4096xf32>) -> tensor<4x4096xf32>
  %terms = intent.map_binary %gt, %ratio {fn = "mul"} : (tensor<4x4096xf32>, tensor<4x4096xf32>) -> tensor<4x4096xf32>
  %loss = intent.reduce %terms {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x4096xf32>) -> tensor<4xf32>
  return %loss : tensor<4xf32>
}
