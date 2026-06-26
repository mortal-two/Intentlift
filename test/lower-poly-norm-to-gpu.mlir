// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// poly_norm (liger): normalizes ONE x register tile at three powers and sums the
// weighted norms. norm(u) = u * rsqrt(mean(u*u) + eps); the row output is
//   Y = w0*norm(x^3) + w1*norm(x^2) + w2*norm(x) + b
// (w0/w1/w2/b/eps are rank-0 Scalar args). x, x^2, x^3 are all recomputed from
// the SAME read-once register tile, so this is THREE last-axis mean reduces
// (one per power's mean-square) feeding ONE read-once block-cooperative
// gpu.launch -> THREE gpu.all_reduce ...add + THREE math.rsqrt. Besides Y[M,N]
// (written from the register tile) it returns the three backward-cache rstd
// values as THREE separate per-row [M,1] outputs (rstd3, rstd2, rstd1, matching
// liger's RSTD[:,0:3] columns), each a per-row uniform written ONCE by thread 0
// (cmpi eq(tx,0) + scf.if) into a rank-1 memref<Mxf32>. Same pass / same DAG
// scheduler as rms_norm/layer_norm — no per-kernel branch, no new mechanism.
// CHECK-LABEL: func.func @poly_norm(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<4xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     scf.for
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         math.rsqrt
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         math.rsqrt
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         math.rsqrt
// CHECK:         arith.cmpi eq, %{{.*}}, %{{.*}} : index
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         scf.if
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         scf.if
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         scf.if
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @poly_norm(%X: tensor<4x8xf32>, %w0: tensor<f32>, %w1: tensor<f32>,
                     %w2: tensor<f32>, %b: tensor<f32>, %eps: tensor<f32>)
    -> (tensor<4x8xf32>, tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>) {
  %xp2 = intent.map_binary %X, %X {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %xp3 = intent.map_binary %xp2, %X {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  // power 3: norm(x^3) = x^3 * rsqrt(mean(x^3 * x^3) + eps)
  %sq3 = intent.map_binary %xp3, %xp3 {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %ms3 = intent.reduce %sq3 {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %ms3e = intent.map_binary %ms3, %eps {fn = "add"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rstd3 = intent.map_unary %ms3e {fn = "rsqrt"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %norm3 = intent.map_binary %xp3, %rstd3 {fn = "mul"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %t3 = intent.map_binary %norm3, %w0 {fn = "mul"} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // power 2: norm(x^2) = x^2 * rsqrt(mean(x^2 * x^2) + eps)
  %sq2 = intent.map_binary %xp2, %xp2 {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %ms2 = intent.reduce %sq2 {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %ms2e = intent.map_binary %ms2, %eps {fn = "add"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rstd2 = intent.map_unary %ms2e {fn = "rsqrt"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %norm2 = intent.map_binary %xp2, %rstd2 {fn = "mul"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %t2 = intent.map_binary %norm2, %w1 {fn = "mul"} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // power 1: norm(x) = x * rsqrt(mean(x * x) + eps); mean(x*x) reuses xp2
  %ms1 = intent.reduce %xp2 {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %ms1e = intent.map_binary %ms1, %eps {fn = "add"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rstd1 = intent.map_unary %ms1e {fn = "rsqrt"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %norm1 = intent.map_binary %X, %rstd1 {fn = "mul"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %t1 = intent.map_binary %norm1, %w2 {fn = "mul"} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // Y = w0*norm(x^3) + w1*norm(x^2) + w2*norm(x) + b
  %s01 = intent.map_binary %t3, %t2 {fn = "add"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %s012 = intent.map_binary %s01, %t1 {fn = "add"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %Y = intent.map_binary %s012, %b {fn = "add"} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  return %Y, %rstd3, %rstd2, %rstd1 : tensor<4x8xf32>, tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32>
}
