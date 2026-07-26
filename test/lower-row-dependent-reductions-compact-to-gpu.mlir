// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=64 %s | FileCheck %s
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=64 fuse-fma=true' %s | FileCheck %s --check-prefix=FMA

// Centered LayerNorm has two dependent row reductions: variance consumes the
// mean.  At production warp counts each dependency level must still use the
// compact per-warp scratch hierarchy instead of materializing two padded
// gpu.all_reduce buffers.  The graph and centered-variance arithmetic remain
// unchanged; only Layer-C synchronization/storage is retuned.
// CHECK-LABEL: func.func @dependent_centered_variance(
// CHECK-SAME: intent.backend.fma_count = 0 : i64
// CHECK-SAME: intent.backend.host_value_capture_count = 1 : i64
// CHECK-SAME: intent.backend.reduction_topology = "workgroup_hierarchical"
// CHECK-SAME: intent.backend.transformation_trace = ["cuda.row_reduction.read_once.v1", "cuda.reduction.workgroup_hierarchical.v1", "cuda.b.rank0_host_value_capture.v1"]
// CHECK-SAME: intent.launch_plan = {{.*}}smem = 16 : i64
// CHECK: gpu.launch
// CHECK-SAME: dynamic_shared_memory_size
// CHECK: gpu.dynamic_shared_memory
// CHECK-COUNT-2: gpu.barrier
// CHECK-NOT: gpu.all_reduce
// CHECK: memref.store
// CHECK: gpu.terminator
// CHECK-NOT: intent.
func.func @dependent_centered_variance(
    %X: tensor<2x128xf32>, %B: tensor<128xf32>, %eps: tensor<f32>)
    -> (tensor<2x128xf32>, tensor<2x1xf32>, tensor<2x1xf32>) {
  %mean = intent.reduce %X {fn = "mean", axes = [1],
      assoc = "float_nondeterministic"} :
      (tensor<2x128xf32>) -> tensor<2x1xf32>
  %centered = intent.map_binary %X, %mean {fn = "sub"} :
      (tensor<2x128xf32>, tensor<2x1xf32>) -> tensor<2x128xf32>
  %square = intent.map_binary %centered, %centered {fn = "mul"} :
      (tensor<2x128xf32>, tensor<2x128xf32>) -> tensor<2x128xf32>
  %variance = intent.reduce %square {fn = "mean", axes = [1],
      assoc = "float_nondeterministic"} :
      (tensor<2x128xf32>) -> tensor<2x1xf32>
  %variance_eps = intent.map_binary %variance, %eps {fn = "add"} :
      (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %rstd = intent.map_unary %variance_eps {fn = "rsqrt"} :
      (tensor<2x1xf32>) -> tensor<2x1xf32>
  %normalized = intent.map_binary %centered, %rstd {fn = "mul"} :
      (tensor<2x128xf32>, tensor<2x1xf32>) -> tensor<2x128xf32>
  %Y = intent.map_binary %normalized, %B {fn = "add"} :
      (tensor<2x128xf32>, tensor<128xf32>) -> tensor<2x128xf32>
  return %Y, %mean, %variance :
      tensor<2x128xf32>, tensor<2x1xf32>, tensor<2x1xf32>
}

// FMA-LABEL: func.func @dependent_centered_variance(
// FMA-SAME: intent.backend.fma_count = 4 : i64
// FMA-SAME: intent.backend.transformation_trace = ["cuda.row_reduction.read_once.v1", "cuda.reduction.workgroup_hierarchical.v1", "cuda.numeric.fma.f32.v1", "cuda.b.rank0_host_value_capture.v1"]
// FMA: math.fma
// FMA-NOT: intent.
