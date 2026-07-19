// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=128 %s | FileCheck %s

// A required RowOut that feeds both the single reduction and a later RowOut is
// stored before the compact block collective, then reloaded for the epilogue.
// The required output store therefore replaces a long cross-barrier SSA live
// range and is not emitted a second time at the end of the kernel.
// CHECK-LABEL: func.func @out_of_place_add_rms(
// CHECK-COUNT-2: memref.store {{.*}}, %arg6
// CHECK: gpu.shuffle
// CHECK: gpu.barrier
// CHECK: gpu.barrier
// CHECK-COUNT-2: memref.load %arg6
// CHECK-NOT: memref.store {{.*}}, %arg6
// CHECK: gpu.terminator
// CHECK-NOT: intent.
func.func @out_of_place_add_rms(
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

// The same transformation is safe for typed in-place output because every Row
// input is loaded before the early overwrite. The post-collective reload comes
// from the aliased input/output pointer, and no duplicate writeback follows.
// CHECK-LABEL: func.func @in_place_add_rms(
// CHECK-COUNT-2: memref.store {{.*}}, %arg0
// CHECK: gpu.shuffle
// CHECK: gpu.barrier
// CHECK: gpu.barrier
// CHECK-COUNT-2: memref.load %arg0
// CHECK-NOT: memref.store {{.*}}, %arg0
// CHECK: gpu.terminator
// CHECK-NOT: intent.
func.func @in_place_add_rms(
    %X: tensor<2x1024xf32>, %Y: tensor<2x1024xf32>,
    %W: tensor<1024xf32>, %eps: tensor<f32>)
    -> (tensor<2x1024xf32>, tensor<2x1024xf32>)
    attributes {intent.inplace = array<i64: 0, -1>} {
  %X_out = intent.map_binary %X, %Y {fn = "add"}
      : (tensor<2x1024xf32>, tensor<2x1024xf32>) -> tensor<2x1024xf32>
  %square = intent.map_binary %X_out, %X_out {fn = "mul"}
      : (tensor<2x1024xf32>, tensor<2x1024xf32>) -> tensor<2x1024xf32>
  %sum = intent.reduce %square {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x1024xf32>) -> tensor<2x1xf32>
  %n = arith.constant dense<1024.0> : tensor<f32>
  %mean = intent.map_binary %sum, %n {fn = "div"}
      : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %mean_eps = intent.map_binary %mean, %eps {fn = "add"}
      : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %rstd = intent.map_unary %mean_eps {fn = "rsqrt"}
      : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %normalized = intent.map_binary %X_out, %rstd {fn = "mul"}
      : (tensor<2x1024xf32>, tensor<2x1xf32>) -> tensor<2x1024xf32>
  %H1 = intent.map_binary %normalized, %W {fn = "mul"}
      : (tensor<2x1024xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  return %X_out, %H1 : tensor<2x1024xf32>, tensor<2x1024xf32>
}
