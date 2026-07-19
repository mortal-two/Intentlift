// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// Reduced-size, structure-preserving versions of the three live-provider
// batch-4 graphs.  These tests intentionally retain the mechanisms that were
// absent from the older hand-written normalization tests: a rank-3 grouped
// boundary with a take-gathered parameter vector, scalar take gathers plus a
// compact per-row concat result, and an integer `ne` condition controlling a
// vector epilogue select.

// CHECK-LABEL: func.func @live_group_norm_rank3
// CHECK-SAME:    (%{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<i32>, %{{.*}}: memref<i32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<4xf32>)
// CHECK:         gpu.launch
// CHECK:         arith.remui
// CHECK-COUNT-4: gpu.shuffle xor
// CHECK:         arith.divsi
// CHECK-COUNT-4: memref.load %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK-COUNT-2: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @live_group_norm_rank3(
    %X: tensor<2x2x8xf32>, %W: tensor<4xf32>, %B: tensor<4xf32>,
    %hidden_size: tensor<i32>, %channels_per_group: tensor<i32>,
    %eps: tensor<f32>)
    -> (tensor<2x2x8xf32>, tensor<2x2xf32>, tensor<2x2xf32>) {
  %sum_x = intent.reduce %X {fn = "sum", axes = [2], assoc = "float_nondeterministic"} : (tensor<2x2x8xf32>) -> tensor<2x2xf32>
  %x_squared = intent.map_binary %X, %X {fn = "mul"} : (tensor<2x2x8xf32>, tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
  %sum_x_squared = intent.reduce %x_squared {fn = "sum", axes = [2], assoc = "float_nondeterministic"} : (tensor<2x2x8xf32>) -> tensor<2x2xf32>
  %hidden_size_f32 = intent.cast %hidden_size {to_dtype = "f32"} : (tensor<i32>) -> tensor<f32>
  %mean = intent.map_binary %sum_x, %hidden_size_f32 {fn = "div"} : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  %second_moment = intent.map_binary %sum_x_squared, %hidden_size_f32 {fn = "div"} : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  %mean_squared = intent.map_binary %mean, %mean {fn = "mul"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %variance = intent.map_binary %second_moment, %mean_squared {fn = "sub"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %variance_eps = intent.map_binary %variance, %eps {fn = "add"} : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  %rstd = intent.map_unary %variance_eps {fn = "rsqrt"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %one_f32 = arith.constant dense<1.0> : tensor<1xf32>
  %mean_keep = intent.contraction %mean, %one_f32 {subscripts = "ng,k->ngk"} : (tensor<2x2xf32>, tensor<1xf32>) -> tensor<2x2x1xf32>
  %rstd_keep = intent.contraction %rstd, %one_f32 {subscripts = "ng,k->ngk"} : (tensor<2x2xf32>, tensor<1xf32>) -> tensor<2x2x1xf32>
  %centered = intent.map_binary %X, %mean_keep {fn = "sub"} : (tensor<2x2x8xf32>, tensor<2x2x1xf32>) -> tensor<2x2x8xf32>
  %normalized = intent.map_binary %centered, %rstd_keep {fn = "mul"} : (tensor<2x2x8xf32>, tensor<2x2x1xf32>) -> tensor<2x2x8xf32>
  %hidden_index = intent.iota : tensor<8xi32>
  %hidden_per_channel = intent.map_binary %hidden_size, %channels_per_group {fn = "div"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %local_channel = intent.map_binary %hidden_index, %hidden_per_channel {fn = "div"} : (tensor<8xi32>, tensor<i32>) -> tensor<8xi32>
  %group_index = intent.iota : tensor<2xi32>
  %group_base = intent.map_binary %group_index, %channels_per_group {fn = "mul"} : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  %ones_hidden = arith.constant dense<1> : tensor<8xi32>
  %group_grid = intent.contraction %group_base, %ones_hidden {subscripts = "g,h->gh"} : (tensor<2xi32>, tensor<8xi32>) -> tensor<2x8xi32>
  %ones_group = arith.constant dense<1> : tensor<2xi32>
  %local_grid = intent.contraction %ones_group, %local_channel {subscripts = "g,h->gh"} : (tensor<2xi32>, tensor<8xi32>) -> tensor<2x8xi32>
  %channel = intent.map_binary %group_grid, %local_grid {fn = "add"} : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi32>
  %weight = intent.gather %W, %channel {axis = 0 : i64, mode = "take", intent.index_bounds_proven = true} : (tensor<4xf32>, tensor<2x8xi32>) -> tensor<2x8xf32>
  %bias = intent.gather %B, %channel {axis = 0 : i64, mode = "take", intent.index_bounds_proven = true} : (tensor<4xf32>, tensor<2x8xi32>) -> tensor<2x8xf32>
  %scaled = intent.map_binary %normalized, %weight {fn = "mul"} : (tensor<2x2x8xf32>, tensor<2x8xf32>) -> tensor<2x2x8xf32>
  %Y = intent.map_binary %scaled, %bias {fn = "add"} : (tensor<2x2x8xf32>, tensor<2x8xf32>) -> tensor<2x2x8xf32>
  return %Y, %mean, %rstd : tensor<2x2x8xf32>, tensor<2x2xf32>, tensor<2x2xf32>
}

// CHECK-LABEL: func.func @live_poly_norm_vector_params
// CHECK-SAME:    (%{{.*}}: memref<4x8xf32>, %{{.*}}: memref<3xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x3xf32>)
// CHECK:         gpu.launch
// CHECK-COUNT-3: memref.load %{{.*}}[%{{.*}}] : memref<3xf32>
// CHECK-COUNT-6: gpu.shuffle xor
// CHECK-COUNT-3: math.rsqrt
// CHECK-COUNT-3: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x3xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @live_poly_norm_vector_params(
    %X: tensor<4x8xf32>, %W: tensor<3xf32>, %B: tensor<f32>,
    %eps: tensor<f32>) -> (tensor<4x8xf32>, tensor<4x3xf32>) {
  %i0 = arith.constant dense<0> : tensor<i32>
  %i1 = arith.constant dense<1> : tensor<i32>
  %i2 = arith.constant dense<2> : tensor<i32>
  %w0 = intent.gather %W, %i0 {axis = 0 : i64, mode = "take"} : (tensor<3xf32>, tensor<i32>) -> tensor<f32>
  %w1 = intent.gather %W, %i1 {axis = 0 : i64, mode = "take"} : (tensor<3xf32>, tensor<i32>) -> tensor<f32>
  %w2 = intent.gather %W, %i2 {axis = 0 : i64, mode = "take"} : (tensor<3xf32>, tensor<i32>) -> tensor<f32>
  %n = arith.constant dense<8.0> : tensor<f32>
  %x2 = intent.map_binary %X, %X {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %x3 = intent.map_binary %x2, %X {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %x3sq = intent.map_binary %x3, %x3 {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %s3 = intent.reduce %x3sq {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %m3 = intent.map_binary %s3, %n {fn = "div"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %e3 = intent.map_binary %m3, %eps {fn = "add"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %r3 = intent.map_unary %e3 {fn = "rsqrt"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %n3 = intent.map_binary %x3, %r3 {fn = "mul"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %x2sq = intent.map_binary %x2, %x2 {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %s2 = intent.reduce %x2sq {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %m2 = intent.map_binary %s2, %n {fn = "div"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %e2 = intent.map_binary %m2, %eps {fn = "add"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %r2 = intent.map_unary %e2 {fn = "rsqrt"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %n2 = intent.map_binary %x2, %r2 {fn = "mul"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %s1 = intent.reduce %x2 {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %m1 = intent.map_binary %s1, %n {fn = "div"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %e1 = intent.map_binary %m1, %eps {fn = "add"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %r1 = intent.map_unary %e1 {fn = "rsqrt"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %n1 = intent.map_binary %X, %r1 {fn = "mul"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %rstd = intent.concat %r3, %r2, %r1 {axis = 1 : i64} : (tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x3xf32>
  %t3 = intent.map_binary %w0, %n3 {fn = "mul"} : (tensor<f32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %t2 = intent.map_binary %w1, %n2 {fn = "mul"} : (tensor<f32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %t1 = intent.map_binary %w2, %n1 {fn = "mul"} : (tensor<f32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %s32 = intent.map_binary %t3, %t2 {fn = "add"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %s321 = intent.map_binary %s32, %t1 {fn = "add"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %Y = intent.map_binary %s321, %B {fn = "add"} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  return %Y, %rstd : tensor<4x8xf32>, tensor<4x3xf32>
}

// CHECK-LABEL: func.func @live_flash_layer_norm_zero_centered
// CHECK-SAME:    (%{{.*}}: memref<4x8xf32>, %{{.*}}: memref<8xf32>, %{{.*}}: memref<8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<i32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<4xf32>)
// CHECK:         gpu.launch
// CHECK:         arith.cmpi ne
// CHECK-COUNT-2: gpu.all_reduce{{.*}}add
// CHECK:         math.rsqrt
// CHECK-NOT:     math.sqrt
// The multi-output epilogue materializes shared SSA values before any guarded
// destination region, then performs the row stores. This keeps cached values
// dominating every output even when a sibling output is partial-writeback.
// CHECK-DAG:     memref.load %{{.*}}[%{{.*}}] : memref<8xf32>
// CHECK-DAG:     memref.load %{{.*}}[%{{.*}}] : memref<8xf32>
// CHECK-DAG:     memref.load %{{.*}}[%{{.*}}] : memref<8xf32>
// CHECK-DAG:     memref.load %{{.*}}[%{{.*}}] : memref<8xf32>
// CHECK-DAG:     arith.select
// CHECK-DAG:     arith.select
// CHECK-COUNT-2: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK-COUNT-2: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @live_flash_layer_norm_zero_centered(
    %X: tensor<4x8xf32>, %W: tensor<8xf32>, %B: tensor<8xf32>,
    %eps: tensor<f32>, %zero_centered_weight: tensor<i32>)
    -> (tensor<4x8xf32>, tensor<4xf32>, tensor<4xf32>) {
  %mean = intent.reduce %X {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4xf32>
  %ones = arith.constant dense<1.0> : tensor<8xf32>
  %mean2d = intent.contraction %mean, %ones {subscripts = "i,j->ij"} : (tensor<4xf32>, tensor<8xf32>) -> tensor<4x8xf32>
  %centered = intent.map_binary %X, %mean2d {fn = "sub"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %square = intent.map_binary %centered, %centered {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %variance = intent.reduce %square {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4xf32>
  %variance_eps = intent.map_binary %variance, %eps {fn = "add"} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %std = intent.map_unary %variance_eps {fn = "sqrt"} : (tensor<4xf32>) -> tensor<4xf32>
  %rstd = intent.map_unary %std {fn = "recip"} : (tensor<4xf32>) -> tensor<4xf32>
  %rstd2d = intent.contraction %rstd, %ones {subscripts = "i,j->ij"} : (tensor<4xf32>, tensor<8xf32>) -> tensor<4x8xf32>
  %xhat = intent.map_binary %centered, %rstd2d {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %zero = arith.constant dense<0> : tensor<i32>
  %use_shift = intent.map_binary %zero_centered_weight, %zero {fn = "ne"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %one = arith.constant dense<1.0> : tensor<f32>
  %W1 = intent.map_binary %W, %one {fn = "add"} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  %Weff = intent.select %use_shift, %W1, %W : (tensor<i1>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %weighted = intent.map_binary %xhat, %Weff {fn = "mul"} : (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4x8xf32>
  %Y = intent.map_binary %weighted, %B {fn = "add"} : (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4x8xf32>
  return %Y, %mean, %rstd : tensor<4x8xf32>, tensor<4xf32>, tensor<4xf32>
}
