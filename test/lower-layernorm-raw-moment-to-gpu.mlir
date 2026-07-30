// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=64 tile-cols=32 raw-moment-variance=false' %s | FileCheck %s --check-prefix=CONTROL
// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=64 tile-cols=32 raw-moment-variance=true' %s | FileCheck %s --check-prefix=RAW
// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=256 allow-execution-domain-sorted-prefix=true require-sorted-prefix=true' %s | FileCheck %s --check-prefix=PREFIX-REJECT

// The control follows the authored dependent mean -> centered-square reduction.
// The expanded state proves the exact typed pattern, reduces raw sum and raw
// sum-of-squares in one publication group, then reconstructs the centered sum.

// CONTROL-LABEL: func.func @layer_norm_graph(
// CONTROL-SAME: intent.backend.barrier_count = 2 : i64
// CONTROL-SAME: intent.backend.reduction_topology = "workgroup_hierarchical"

// RAW-LABEL: func.func @layer_norm_graph(
// RAW-SAME: intent.backend.barrier_count = 1 : i64
// RAW-SAME: intent.backend.reduction_topology = "raw_moment_hierarchical"
// RAW-SAME: "cuda.reduction.raw_moment_hierarchical.v1"
// RAW: arith.mulf
// RAW: arith.subf
// RAW-NOT: intent.

// PREFIX-REJECT-LABEL: func.func @layer_norm_graph(
// PREFIX-REJECT: intent.reduce
// PREFIX-REJECT-NOT: gpu.launch

func.func @layer_norm_graph(%X: tensor<64x128xf32>, %eps: tensor<f32>)
    -> (tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>) {
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<64x128xf32>) -> tensor<64xf32>
  %n = arith.constant dense<128.0> : tensor<f32>
  %mean = intent.map_binary %sum, %n {fn = "div"}
    : (tensor<64xf32>, tensor<f32>) -> tensor<64xf32>
  %mean2d = tensor.expand_shape %mean [[0, 1]] output_shape [64, 1]
    : tensor<64xf32> into tensor<64x1xf32>
  %centered = intent.map_binary %X, %mean2d {fn = "sub"}
    : (tensor<64x128xf32>, tensor<64x1xf32>) -> tensor<64x128xf32>
  %square = intent.map_binary %centered, %centered {fn = "mul"}
    : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %variance_sum = intent.reduce %square {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<64x128xf32>) -> tensor<64xf32>
  %variance = intent.map_binary %variance_sum, %n {fn = "div"}
    : (tensor<64xf32>, tensor<f32>) -> tensor<64xf32>
  %shifted = intent.map_binary %variance, %eps {fn = "add"}
    : (tensor<64xf32>, tensor<f32>) -> tensor<64xf32>
  %rstd = intent.map_unary %shifted {fn = "rsqrt"}
    : (tensor<64xf32>) -> tensor<64xf32>
  %Y = intent.contraction %centered, %rstd {subscripts = "ab,a->ab"}
    : (tensor<64x128xf32>, tensor<64xf32>) -> tensor<64x128xf32>
  return %Y, %mean, %rstd : tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>
}
