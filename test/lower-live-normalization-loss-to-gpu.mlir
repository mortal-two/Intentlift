// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// These are reduced-size versions of the unchanged live-provider graphs used
// by the paper RMSNorm, fused-add RMSNorm, LayerNorm, Softmax and cross-entropy
// chains.  In particular, they retain
// the no-reduction contractions, singleton-axis reductions, target gather,
// integer per-row labels, and conditional per-row/result writes produced by
// extraction.  Both complete graphs must be consumed by one row kernel.

// CHECK-LABEL: func.func @live_rms_norm(
// CHECK:         gpu.launch
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         math.rsqrt
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @live_rms_norm(%x: tensor<2x8xf32>, %w: tensor<8xf32>,
                         %n_cols: tensor<i32>, %eps: tensor<f32>,
                         %offset: tensor<f32>)
    -> (tensor<2x8xf32>, tensor<2xf32>) {
  %n = intent.cast %n_cols {to_dtype = "f32"} : (tensor<i32>) -> tensor<f32>
  %sq = intent.map_binary %x, %x {fn = "mul"} : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  %sum = intent.reduce %sq {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x8xf32>) -> tensor<2xf32>
  %mean = intent.map_binary %sum, %n {fn = "div"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %regularized = intent.map_binary %mean, %eps {fn = "add"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %rstd = intent.map_unary %regularized {fn = "rsqrt"} : (tensor<2xf32>) -> tensor<2xf32>
  %scaled = intent.contraction %x, %rstd {subscripts = "mn,m->mn"} : (tensor<2x8xf32>, tensor<2xf32>) -> tensor<2x8xf32>
  %weight = intent.map_binary %offset, %w {fn = "add"} : (tensor<f32>, tensor<8xf32>) -> tensor<8xf32>
  %y = intent.map_binary %scaled, %weight {fn = "mul"} : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
  return %y, %rstd : tensor<2x8xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @live_fused_add_rms_norm(
// CHECK:         gpu.launch
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         math.rsqrt
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @live_fused_add_rms_norm(
    %x: tensor<2x8xf32>, %r: tensor<2x8xf32>, %w: tensor<8xf32>,
    %eps: tensor<f32>, %offset: tensor<f32>)
    -> (tensor<2x8xf32>, tensor<2x8xf32>, tensor<2xf32>) {
  %s = intent.map_binary %x, %r {fn = "add"} : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  %sq = intent.map_binary %s, %s {fn = "mul"} : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  %mean = intent.reduce %sq {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x8xf32>) -> tensor<2xf32>
  %regularized = intent.map_binary %mean, %eps {fn = "add"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %rstd = intent.map_unary %regularized {fn = "rsqrt"} : (tensor<2xf32>) -> tensor<2xf32>
  %ones = arith.constant dense<1.0> : tensor<8xf32>
  %rstd2 = intent.contraction %rstd, %ones {subscripts = "i,j->ij"} : (tensor<2xf32>, tensor<8xf32>) -> tensor<2x8xf32>
  %norm = intent.map_binary %s, %rstd2 {fn = "mul"} : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  %affine = intent.map_binary %offset, %w {fn = "add"} : (tensor<f32>, tensor<8xf32>) -> tensor<8xf32>
  %y = intent.map_binary %norm, %affine {fn = "mul"} : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
  return %y, %s, %rstd : tensor<2x8xf32>, tensor<2x8xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @live_layer_norm(
// CHECK:         gpu.launch
// CHECK-COUNT-2: gpu.all_reduce{{.*}}add
// CHECK:         math.rsqrt
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @live_layer_norm(
    %x: tensor<2x8xf32>, %w: tensor<8xf32>, %bias: tensor<8xf32>,
    %eps: tensor<f32>)
    -> (tensor<2x8xf32>, tensor<2xf32>, tensor<2xf32>) {
  %n = arith.constant dense<8.0> : tensor<f32>
  %xt = intent.transpose %x {perm = [1, 0]} : (tensor<2x8xf32>) -> tensor<8x2xf32>
  %sum = intent.reduce %xt {fn = "sum", axes = [0], assoc = "float_nondeterministic"} : (tensor<8x2xf32>) -> tensor<2xf32>
  %mean = intent.map_binary %sum, %n {fn = "div"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %centered = intent.map_binary %xt, %mean {fn = "sub"} : (tensor<8x2xf32>, tensor<2xf32>) -> tensor<8x2xf32>
  %square = intent.map_binary %centered, %centered {fn = "mul"} : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf32>
  %sum_square = intent.reduce %square {fn = "sum", axes = [0], assoc = "float_nondeterministic"} : (tensor<8x2xf32>) -> tensor<2xf32>
  %var = intent.map_binary %sum_square, %n {fn = "div"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %regularized = intent.map_binary %var, %eps {fn = "add"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %rstd = intent.map_unary %regularized {fn = "rsqrt"} : (tensor<2xf32>) -> tensor<2xf32>
  %normalized_t = intent.map_binary %centered, %rstd {fn = "mul"} : (tensor<8x2xf32>, tensor<2xf32>) -> tensor<8x2xf32>
  %normalized = intent.transpose %normalized_t {perm = [1, 0]} : (tensor<8x2xf32>) -> tensor<2x8xf32>
  %scaled = intent.map_binary %normalized, %w {fn = "mul"} : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
  %y = intent.map_binary %scaled, %bias {fn = "add"} : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
  return %y, %mean, %rstd : tensor<2x8xf32>, tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @live_softmax(
// CHECK:         gpu.launch
// CHECK:         gpu.all_reduce{{.*}}maxnumf
// CHECK:         math.exp
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @live_softmax(%x: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %maximum = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"} : (tensor<2x8xf32>) -> tensor<2x1xf32>
  %shifted = intent.map_binary %x, %maximum {fn = "sub"} : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
  %exponent = intent.map_unary %shifted {fn = "exp"} : (tensor<2x8xf32>) -> tensor<2x8xf32>
  %denominator = intent.reduce %exponent {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x8xf32>) -> tensor<2x1xf32>
  %y = intent.map_binary %exponent, %denominator {fn = "div"} : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
  return %y : tensor<2x8xf32>
}

// CHECK-LABEL: func.func @live_cross_entropy(
// CHECK:         gpu.launch
// CHECK:         memref.load {{.*}}[%{{.*}}, %{{.*}}] : memref<2x8xf32>
// CHECK:         arith.cmpi slt
// CHECK:         arith.select
// CHECK:         arith.cmpi sgt
// CHECK:         arith.select
// CHECK:         memref.load {{.*}}[%{{.*}}, %{{.*}}] : memref<2x8xf32>
// CHECK:         gpu.all_reduce{{.*}}maxnumf
// CHECK:         math.exp
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         arith.select
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @live_cross_entropy(%x: tensor<2x8xf32>, %target: tensor<2xi64>,
                              %loss_initial: tensor<2xf32>,
                              %n_non_ignore: tensor<i32>,
                              %ignore_index: tensor<i32>)
    -> (tensor<2x8xf32>, tensor<2xf32>)
    attributes {intent.inplace = array<i64: 0, 2>} {
  %ignore64 = intent.cast %ignore_index {to_dtype = "i64"} : (tensor<i32>) -> tensor<i64>
  %ignored = intent.map_binary %target, %ignore64 {fn = "eq"} : (tensor<2xi64>, tensor<i64>) -> tensor<2xi1>
  %zero_i64 = arith.constant dense<0> : tensor<i64>
  %safe = intent.select %ignored, %zero_i64, %target : (tensor<2xi1>, tensor<i64>, tensor<2xi64>) -> tensor<2xi64>
  %one_i64 = arith.constant dense<1> : tensor<1xi64>
  %safe_col = intent.contraction %safe, %one_i64 {subscripts = "i,j->ij"} : (tensor<2xi64>, tensor<1xi64>) -> tensor<2x1xi64>
  %row_max = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"} : (tensor<2x8xf32>) -> tensor<2x1xf32>
  %shift = intent.map_binary %x, %row_max {fn = "sub"} : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
  %exp = intent.map_unary %shift {fn = "exp"} : (tensor<2x8xf32>) -> tensor<2x8xf32>
  %exp_sum = intent.reduce %exp {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<2x8xf32>) -> tensor<2x1xf32>
  %log_sum = intent.map_unary %exp_sum {fn = "log"} : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %lse_keep = intent.map_binary %row_max, %log_sum {fn = "add"} : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %lse = intent.reduce %lse_keep {fn = "sum", axes = [1], assoc = "associative"} : (tensor<2x1xf32>) -> tensor<2xf32>
  %target_keep = intent.gather %x, %safe_col {axis = 1 : i64, mode = "along_axis"} : (tensor<2x8xf32>, tensor<2x1xi64>) -> tensor<2x1xf32>
  %target_logit = intent.reduce %target_keep {fn = "sum", axes = [1], assoc = "associative"} : (tensor<2x1xf32>) -> tensor<2xf32>
  %nll = intent.map_binary %lse, %target_logit {fn = "sub"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %count = intent.cast %n_non_ignore {to_dtype = "f32"} : (tensor<i32>) -> tensor<f32>
  %mean_nll = intent.map_binary %nll, %count {fn = "div"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %zero = arith.constant dense<0.0> : tensor<f32>
  %z0 = intent.map_binary %zero, %lse {fn = "mul"} : (tensor<f32>, tensor<2xf32>) -> tensor<2xf32>
  %z1 = intent.map_binary %z0, %lse {fn = "mul"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %z2 = intent.map_binary %z1, %count {fn = "div"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %candidate = intent.map_binary %mean_nll, %z2 {fn = "add"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %loss = intent.select %ignored, %loss_initial, %candidate : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %ignored_f32 = intent.cast %ignored {to_dtype = "f32"} : (tensor<2xi1>) -> tensor<2xf32>
  %ones = arith.constant dense<1.0> : tensor<8xf32>
  %mask_matrix = intent.contraction %ignored_f32, %ones {subscripts = "i,j->ij"} : (tensor<2xf32>, tensor<8xf32>) -> tensor<2x8xf32>
  %mask = intent.map_binary %mask_matrix, %zero {fn = "gt"} : (tensor<2x8xf32>, tensor<f32>) -> tensor<2x8xi1>
  %x_out = intent.select %mask, %zero, %x : (tensor<2x8xi1>, tensor<f32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %x_out, %loss : tensor<2x8xf32>, tensor<2xf32>
}
