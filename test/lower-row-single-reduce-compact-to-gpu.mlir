// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=128 %s | FileCheck %s

// A single row reduction on a power-of-two multi-warp schedule uses a compact
// two-level hierarchy rather than gpu.all_reduce. Four warps need five
// intra-warp XOR shuffles plus two cross-warp shuffles, two barriers, and only
// four f32 shared slots. The condition is structural and independent of the
// function name or normalization epilogue.
// CHECK-LABEL: func.func @single_reduce_compact(
// CHECK-SAME: intent.launch_plan = {{.*}}smem = 16
// CHECK: gpu.dynamic_shared_memory
// CHECK: memref.view {{.*}} to memref<4xf32, #gpu.address_space<workgroup>>
// CHECK-NOT: gpu.all_reduce
// CHECK-COUNT-5: gpu.shuffle
// CHECK: gpu.barrier
// CHECK-COUNT-2: gpu.shuffle
// CHECK: gpu.barrier
// CHECK: memref.load {{.*}} : memref<4xf32, #gpu.address_space<workgroup>>
// CHECK: gpu.terminator
// CHECK-NOT: intent.
func.func @single_reduce_compact(
    %X: tensor<2x1024xf32>, %W: tensor<1024xf32>, %eps: tensor<f32>)
    -> tensor<2x1024xf32> {
  %square = intent.map_binary %X, %X {fn = "mul"}
      : (tensor<2x1024xf32>, tensor<2x1024xf32>) -> tensor<2x1024xf32>
  %mean = intent.reduce %square {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x1024xf32>) -> tensor<2x1xf32>
  %mean_eps = intent.map_binary %mean, %eps {fn = "add"}
      : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %rstd = intent.map_unary %mean_eps {fn = "rsqrt"}
      : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %scaled = intent.map_binary %X, %rstd {fn = "mul"}
      : (tensor<2x1024xf32>, tensor<2x1xf32>) -> tensor<2x1024xf32>
  %Y = intent.map_binary %scaled, %W {fn = "mul"}
      : (tensor<2x1024xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  return %Y : tensor<2x1024xf32>
}
