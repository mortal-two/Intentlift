// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=64 %s | FileCheck %s
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=4 tile-cols=2' %s | FileCheck %s --check-prefix=TILED
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=64 tile-cols=1' %s | FileCheck %s --check-prefix=WARP-TILED

// Two independent f32 reductions share one publication barrier. Each reduction
// has its own compact [numWarps] dynamic-shared slot, so neither a second
// publish barrier nor a standalone gpu.all_reduce buffer is required.

// CHECK-LABEL: func.func @independent_pair(
// CHECK-SAME: attributes {
// CHECK-SAME: intent.backend.folded_constexpr_select_count = 0 : i64
// CHECK-SAME: intent.backend.predicated_value_if_count = 0 : i64
// CHECK-SAME: intent.launch_plan = {{.*}}smem = 16 : i64
// CHECK: gpu.launch
// CHECK-SAME: dynamic_shared_memory_size
// CHECK: gpu.dynamic_shared_memory
// CHECK-COUNT-1: gpu.barrier
// CHECK-NOT: gpu.all_reduce
// CHECK: memref.store
// CHECK: gpu.terminator
// CHECK-NOT: intent.
// TILED-LABEL: func.func @independent_pair(
// TILED-SAME: intent.backend.reduction_topology = "tiled_shared_multi_reduce"
// TILED-SAME: intent.backend.transformation_trace = ["cuda.row_reduction.read_once.v1", "cuda.reduction.tiled_shared_multi_reduce.v1"]
// TILED: gpu.launch
// TILED: scf.for %{{.*}} iter_args(%{{.*}}, %{{.*}}) -> (f32, f32) {
// TILED: memref.store %{{.*}} : memref<128xf32, #gpu.address_space<workgroup>>
// TILED-NOT: gpu.all_reduce
// TILED: gpu.terminator
// TILED-NOT: intent.
// WARP-TILED-LABEL: func.func @independent_pair(
// WARP-TILED-SAME: intent.backend.barrier_count = 1
// WARP-TILED-SAME: intent.backend.reduction_topology = "tiled_frontier_warp_multi_reduce"
// WARP-TILED-SAME: intent.backend.shuffle_count = 20
// WARP-TILED-SAME: intent.backend.transformation_trace = ["cuda.row_reduction.read_once.v1", "cuda.reduction.tiled_frontier_warp_multi_reduce.v1"]
// WARP-TILED: memref.view {{.*}} to memref<2xf32, #gpu.address_space<workgroup>>
// WARP-TILED: gpu.shuffle
// WARP-TILED-COUNT-1: gpu.barrier
// WARP-TILED-NOT: gpu.all_reduce
// WARP-TILED: gpu.terminator
// WARP-TILED-NOT: intent.
func.func @independent_pair(%X: tensor<2x128xf32>)
    -> (tensor<2x128xf32>, tensor<2x1xf32>, tensor<2x1xf32>) {
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x128xf32>) -> tensor<2x1xf32>
  %square = intent.map_binary %X, %X {fn = "mul"}
      : (tensor<2x128xf32>, tensor<2x128xf32>) -> tensor<2x128xf32>
  %sum_square = intent.reduce %square {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x128xf32>) -> tensor<2x1xf32>
  %Y = intent.map_binary %X, %sum {fn = "sub"}
      : (tensor<2x128xf32>, tensor<2x1xf32>) -> tensor<2x128xf32>
  return %Y, %sum, %sum_square
      : tensor<2x128xf32>, tensor<2x1xf32>, tensor<2x1xf32>
}
