// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// A rank-0 specialization switch is uniform for the whole block.  It may
// choose between W and W+1 without changing the logical column mapping, so the
// Flash-LayerNorm shape must retain vec4 Row/Col loads and the vec4 Row store.
// CHECK-LABEL: func.func @layer_norm_uniform_select_vec4(
// CHECK-SAME: %{{[^,]+}}: memref<2x4xvector<4xf32>>
// CHECK-SAME: %{{[^,]+}}: memref<4xvector<4xf32>>
// CHECK-SAME: %{{[^,]+}}: memref<4xvector<4xf32>>
// CHECK-SAME: %{{[^,]+}}: memref<f32>
// CHECK-SAME: %{{[^,]+}}: memref<i32>
// CHECK-SAME: %{{[^,]+}}: memref<2x4xvector<4xf32>>
// CHECK: memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x4xvector<4xf32>>
// CHECK: gpu.all_reduce{{.*}}add
// CHECK: gpu.all_reduce{{.*}}add
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<4xvector<4xf32>>
// CHECK: arith.select
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x4xvector<4xf32>>
// CHECK-NOT: intent.
func.func @layer_norm_uniform_select_vec4(
    %X: tensor<2x16xf32>, %W: tensor<16xf32>, %B: tensor<16xf32>,
    %eps: tensor<f32>, %zero_centered_weight: tensor<i32>)
    -> (tensor<2x16xf32>, tensor<2xf32>, tensor<2xf32>) {
  %mean = intent.reduce %X {fn = "mean", axes = [1], assoc = "float_nondeterministic"} :
      (tensor<2x16xf32>) -> tensor<2xf32>
  %ones = arith.constant dense<1.0> : tensor<16xf32>
  %mean2d = intent.contraction %mean, %ones {subscripts = "i,j->ij"} :
      (tensor<2xf32>, tensor<16xf32>) -> tensor<2x16xf32>
  %centered = intent.map_binary %X, %mean2d {fn = "sub"} :
      (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
  %square = intent.map_binary %centered, %centered {fn = "mul"} :
      (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
  %variance = intent.reduce %square {fn = "mean", axes = [1], assoc = "float_nondeterministic"} :
      (tensor<2x16xf32>) -> tensor<2xf32>
  %variance_eps = intent.map_binary %variance, %eps {fn = "add"} :
      (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %rstd = intent.map_unary %variance_eps {fn = "rsqrt"} :
      (tensor<2xf32>) -> tensor<2xf32>
  %rstd2d = intent.contraction %rstd, %ones {subscripts = "i,j->ij"} :
      (tensor<2xf32>, tensor<16xf32>) -> tensor<2x16xf32>
  %xhat = intent.map_binary %centered, %rstd2d {fn = "mul"} :
      (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
  %zero = arith.constant dense<0> : tensor<i32>
  %use_shift = intent.map_binary %zero_centered_weight, %zero {fn = "ne"} :
      (tensor<i32>, tensor<i32>) -> tensor<i1>
  %one = arith.constant dense<1.0> : tensor<f32>
  %W1 = intent.map_binary %W, %one {fn = "add"} :
      (tensor<16xf32>, tensor<f32>) -> tensor<16xf32>
  %Weff = intent.select %use_shift, %W1, %W :
      (tensor<i1>, tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  %weighted = intent.map_binary %xhat, %Weff {fn = "mul"} :
      (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
  %Y = intent.map_binary %weighted, %B {fn = "add"} :
      (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
  return %Y, %mean, %rstd : tensor<2x16xf32>, tensor<2xf32>, tensor<2xf32>
}
