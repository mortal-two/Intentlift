// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// layer_norm (centered variance, multi-output): two sequential last-axis mean
// reduces — mean over X, then mean over (X-mean)^2 — with a centered-square map
// between them, lowered to ONE read-once block-cooperative gpu.launch. Two
// cooperative passes -> two gpu.all_reduce ...add (each a mean: all_reduce then
// /N); the register tile is loaded once and the centered/normalized epilogue is
// recomputed from it. THREE results: Y[M,N] (the row output, written from the
// register tile) plus the per-row stats mean[M,1] and rstd[M,1] — each a per-row
// uniform written ONCE by thread 0 (cmpi eq(tx,0) + scf.if) into a rank-1
// memref<Mxf32> (the liger/flash (n_rows,) Mean/RSTD pointer layout). Same pass,
// same DAG scheduler as rms_norm/softmax — multi-output is the only new shape.
// CHECK-LABEL: func.func @layer_norm(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<8xf32>, %{{.*}}: memref<8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<4xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     scf.for
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         math.rsqrt
// CHECK:         arith.cmpi eq, %{{.*}}, %{{.*}} : index
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         scf.if
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         scf.if
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @layer_norm(%X: tensor<4x8xf32>, %W: tensor<8xf32>, %B: tensor<8xf32>, %eps: tensor<f32>) -> (tensor<4x8xf32>, tensor<4x1xf32>, tensor<4x1xf32>) {
  %mean = intent.reduce %X {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %xc = intent.map_binary %X, %mean {fn = "sub"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %xc2 = intent.map_binary %xc, %xc {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %var = intent.reduce %xc2 {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %vare = intent.map_binary %var, %eps {fn = "add"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rstd = intent.map_unary %vare {fn = "rsqrt"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %xn = intent.map_binary %xc, %rstd {fn = "mul"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %xnw = intent.map_binary %xn, %W {fn = "mul"} : (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4x8xf32>
  %Y = intent.map_binary %xnw, %B {fn = "add"} : (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4x8xf32>
  return %Y, %mean, %rstd : tensor<4x8xf32>, tensor<4x1xf32>, tensor<4x1xf32>
}
