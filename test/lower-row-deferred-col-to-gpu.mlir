// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=128 %s | FileCheck %s
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=128 defer-epilogue-col-loads=false' %s | FileCheck %s --check-prefix=PRELOAD

// Centered LayerNorm uses W/B only in its pointwise epilogue.  Their vec4 loads
// must occur after both block collectives so the column tiles do not stay live
// across the mean and variance reductions.
// CHECK-LABEL: func.func @centered_layer_deferred_cols(
// CHECK: gpu.launch
// CHECK-NOT: memref.load %arg1
// CHECK-NOT: memref.load %arg2
// CHECK: gpu.barrier
// CHECK-NOT: memref.load %arg1
// CHECK-NOT: memref.load %arg2
// CHECK: gpu.barrier
// CHECK: memref.load %arg1
// CHECK: memref.load %arg2
// CHECK-NOT: intent.
// PRELOAD-LABEL: func.func @centered_layer_deferred_cols(
// PRELOAD: gpu.launch
// PRELOAD: memref.load %arg1
// PRELOAD: memref.load %arg2
// PRELOAD: gpu.barrier
func.func @centered_layer_deferred_cols(
    %X: tensor<2x1024xf32>, %W: tensor<1024xf32>,
    %B: tensor<1024xf32>, %eps: tensor<f32>)
    -> (tensor<2x1024xf32>, tensor<2xf32>, tensor<2xf32>) {
  %n = arith.constant dense<1024.0> : tensor<f32>
  %xt = intent.transpose %X {perm = [1, 0]} :
      (tensor<2x1024xf32>) -> tensor<1024x2xf32>
  %sum = intent.reduce %xt {fn = "sum", axes = [0],
      assoc = "float_nondeterministic"} :
      (tensor<1024x2xf32>) -> tensor<2xf32>
  %mean = intent.map_binary %sum, %n {fn = "div"} :
      (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %centered = intent.map_binary %xt, %mean {fn = "sub"} :
      (tensor<1024x2xf32>, tensor<2xf32>) -> tensor<1024x2xf32>
  %square = intent.map_binary %centered, %centered {fn = "mul"} :
      (tensor<1024x2xf32>, tensor<1024x2xf32>) -> tensor<1024x2xf32>
  %square_sum = intent.reduce %square {fn = "sum", axes = [0],
      assoc = "float_nondeterministic"} :
      (tensor<1024x2xf32>) -> tensor<2xf32>
  %var = intent.map_binary %square_sum, %n {fn = "div"} :
      (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %var_eps = intent.map_binary %var, %eps {fn = "add"} :
      (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %rstd = intent.map_unary %var_eps {fn = "rsqrt"} :
      (tensor<2xf32>) -> tensor<2xf32>
  %normalized_t = intent.map_binary %centered, %rstd {fn = "mul"} :
      (tensor<1024x2xf32>, tensor<2xf32>) -> tensor<1024x2xf32>
  %normalized = intent.transpose %normalized_t {perm = [1, 0]} :
      (tensor<1024x2xf32>) -> tensor<2x1024xf32>
  %scaled = intent.map_binary %normalized, %W {fn = "mul"} :
      (tensor<2x1024xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  %Y = intent.map_binary %scaled, %B {fn = "add"} :
      (tensor<2x1024xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  return %Y, %mean, %rstd :
      tensor<2x1024xf32>, tensor<2xf32>, tensor<2xf32>
}

// A Col value that feeds a reduction is not deferred: its vector load must be
// available before the first warp collective.  This guards the fail-closed
// side of epilogue-only column scheduling.
// CHECK-LABEL: func.func @col_reduction_dependency_preloads(
// CHECK: gpu.launch
// CHECK: memref.load %arg1
// CHECK: gpu.shuffle
// CHECK-NOT: intent.
func.func @col_reduction_dependency_preloads(
    %X: tensor<2x1024xf32>, %W: tensor<1024xf32>)
    -> (tensor<2x1024xf32>, tensor<2x1xf32>) {
  %weighted = intent.map_binary %X, %W {fn = "mul"} :
      (tensor<2x1024xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  %mean = intent.reduce %weighted {fn = "mean", axes = [1],
      assoc = "float_nondeterministic"} :
      (tensor<2x1024xf32>) -> tensor<2x1xf32>
  %Y = intent.map_binary %X, %mean {fn = "sub"} :
      (tensor<2x1024xf32>, tensor<2x1xf32>) -> tensor<2x1024xf32>
  return %Y, %mean : tensor<2x1024xf32>, tensor<2x1xf32>
}
