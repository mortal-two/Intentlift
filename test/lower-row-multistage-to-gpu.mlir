// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=128 vector-width=1 reduction-parts=4' %s | FileCheck %s

// One structural single-reduction matcher emits a compiler-owned launch graph:
// partials [M,P], per-row finalize, then a flat RowOut epilogue.  No source or
// kernel name participates in this decision.
// CHECK-LABEL: func.func @add_rms_multistage(
// CHECK-SAME: intent.backend.grid_blocks = 8 : i64
// CHECK-SAME: intent.backend.reduction_parts = 4 : i64
// CHECK-SAME: intent.backend.reduction_topology = "multi_stage_partial_finalize"
// CHECK-SAME: intent.backend.stage_count = 3 : i64
// CHECK-SAME: intent.launch_plan = {
// CHECK-SAME: split_factor = 4 : i64
// CHECK: %[[PARTIALS:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK: %[[STATS:.*]] = memref.alloc() : memref<2xf32>
// CHECK: gpu.launch
// CHECK: gpu.all_reduce add
// CHECK: memref.store {{.*}}, %[[PARTIALS]]
// CHECK: gpu.launch
// CHECK: memref.load %[[PARTIALS]]
// CHECK: gpu.all_reduce add
// CHECK: memref.store {{.*}}, %[[STATS]]
// CHECK: gpu.launch
// CHECK: memref.load %[[STATS]]
// CHECK: return
// CHECK-NOT: intent.
func.func @add_rms_multistage(
    %X: tensor<2x1024xf32>, %R: tensor<2x1024xf32>,
    %W: tensor<1024xf32>, %eps: tensor<f32>, %offset: tensor<f32>)
    -> (tensor<2x1024xf32>, tensor<2x1024xf32>, tensor<2xf32>) {
  %S = intent.map_binary %X, %R {fn = "add"}
      : (tensor<2x1024xf32>, tensor<2x1024xf32>) -> tensor<2x1024xf32>
  %square = intent.map_binary %S, %S {fn = "mul"}
      : (tensor<2x1024xf32>, tensor<2x1024xf32>) -> tensor<2x1024xf32>
  %mean = intent.reduce %square {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x1024xf32>) -> tensor<2xf32>
  %mean_eps = intent.map_binary %mean, %eps {fn = "add"}
      : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %rstd = intent.map_unary %mean_eps {fn = "rsqrt"}
      : (tensor<2xf32>) -> tensor<2xf32>
  %ones = arith.constant dense<1.0> : tensor<1024xf32>
  %rstd_broadcast = intent.contraction %rstd, %ones {subscripts = "i,j->ij"}
      : (tensor<2xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  %normalized = intent.map_binary %S, %rstd_broadcast {fn = "mul"}
      : (tensor<2x1024xf32>, tensor<2x1024xf32>) -> tensor<2x1024xf32>
  %shifted_weight = intent.map_binary %offset, %W {fn = "add"}
      : (tensor<f32>, tensor<1024xf32>) -> tensor<1024xf32>
  %Y = intent.map_binary %normalized, %shifted_weight {fn = "mul"}
      : (tensor<2x1024xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  return %Y, %S, %rstd
      : tensor<2x1024xf32>, tensor<2x1024xf32>, tensor<2xf32>
}

// A statistic-only graph needs no epilogue launch, but it uses the identical
// partial/finalize implementation and records the actual two-stage topology.
// CHECK-LABEL: func.func @row_sum_multistage(
// CHECK-SAME: intent.backend.reduction_parts = 4 : i64
// CHECK-SAME: intent.backend.stage_count = 2 : i64
// CHECK: memref.alloc() : memref<2x4xf32>
// CHECK: memref.alloc() : memref<2xf32>
// CHECK: gpu.launch
// CHECK: gpu.all_reduce add
// CHECK: gpu.launch
// CHECK: gpu.all_reduce add
// CHECK: return
// CHECK-NOT: intent.
func.func @row_sum_multistage(%X: tensor<2x1024xf32>) -> tensor<2xf32> {
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x1024xf32>) -> tensor<2xf32>
  return %sum : tensor<2xf32>
}
