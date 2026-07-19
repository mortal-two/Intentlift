// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// rms_norm: one mean reduce over the last axis + broadcast-affine epilogue,
// lowered to ONE read-once block-cooperative gpu.launch. The K=N/T feature tile
// is loaded ONCE into SSA registers (fully unrolled over k — a memref cache
// would land in local memory and cap bandwidth); the per-row scalar (mean+eps,
// sqrt, recip) is hoisted once; the epilogue reuses the register tile to write
// Y. The pass is general over the DAG — there is no "recognize rms_norm" branch.
// CHECK-LABEL: func.func @rms_norm(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     scf.for
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         math.rsqrt
// CHECK-NOT:     math.sqrt
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @rms_norm(%X: tensor<4x8xf32>, %W: tensor<8xf32>, %eps: tensor<f32>) -> tensor<4x8xf32> {
  %x2 = intent.map_binary %X, %X {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %mean = intent.reduce %x2 {fn = "mean", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %me = intent.map_binary %mean, %eps {fn = "add"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rms = intent.map_unary %me {fn = "sqrt"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %inv = intent.map_unary %rms {fn = "recip"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %xs = intent.map_binary %X, %inv {fn = "mul"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %Y = intent.map_binary %xs, %W {fn = "mul"} : (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4x8xf32>
  return %Y : tensor<4x8xf32>
}
