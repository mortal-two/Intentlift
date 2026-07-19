// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// fused_add_rms_norm (liger, casting_mode none): S = X + R (R = residual, a
// Row[M,N] input); then RMS over S: rstd = rsqrt(mean(S*S) + eps); the output is
//   Y = S * rstd * (offset + W)
// (W a Col[N] arg, eps/offset rank-0 Scalars). Both S (the new residual) and Y
// are full [M,N] RowOut outputs written from the SAME read-once register tile —
// S = X + R is an [M,N] intermediate that is ALSO the reduce input (emitAt
// recomputes it from the X and R register tiles). ONE last-axis mean reduce ->
// ONE gpu.all_reduce add + math.rsqrt; TWO register-tile stores, no thread-0
// guard (both are RowOut, not per-row stats). Separate S output buffer (no
// in-place aliasing in M5a). Same pass / same DAG scheduler as rms_norm — the
// only new shape is the second RowOut. No per-kernel branch, no new mechanism.
// CHECK-LABEL: func.func @fused_add_rms_norm(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     scf.for
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         math.rsqrt
// CHECK-NOT:     scf.if
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @fused_add_rms_norm(%X: tensor<4x8xf32>, %R: tensor<4x8xf32>,
                              %W: tensor<8xf32>, %eps: tensor<f32>,
                              %offset: tensor<f32>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %S = intent.map_binary %X, %R {fn = "add"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %sq = intent.map_binary %S, %S {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %ms = intent.reduce %sq {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %mse = intent.map_binary %ms, %eps {fn = "add"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rstd = intent.map_unary %mse {fn = "rsqrt"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %Sn = intent.map_binary %S, %rstd {fn = "mul"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %Woff = intent.map_binary %W, %offset {fn = "add"} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  %Y = intent.map_binary %Sn, %Woff {fn = "mul"} : (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4x8xf32>
  return %S, %Y : tensor<4x8xf32>, tensor<4x8xf32>
}
