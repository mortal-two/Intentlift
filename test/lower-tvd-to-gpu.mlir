// RUN: intent-opt -convert-intent-elementwise-to-gpu=block-threads=4 %s | FileCheck %s

// tvd (total-variation distance, reduction="none") lowered by the pure-
// elementwise pass to ONE read-once block-cooperative gpu.launch. Two [M,N]
// outputs from one register-tile pass:
//   grad = where(p > q, +0.5*scale, -0.5*scale)   (intent.select over a cmp)
//   loss = 0.5 * |p - q|
// p,q are Row[M,N]; half/nhalf/scale are rank-0 Scalar. The comparison feeds an
// arith.cmpf -> i1 condition; intent.select lowers to arith.select. The two
// per-row scale products (0.5*scale, -0.5*scale) are uniform -> hoisted once.
// NO gpu.all_reduce (nothing reduced), NO memref.alloca (register tile). M=4,
// N=8, T=4.
// CHECK-LABEL: func.func @tvd(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     gpu.all_reduce
// CHECK-NOT:     scf.for
// CHECK:         memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         arith.cmpf ogt
// CHECK:         arith.select
// |p-q| is emitted as max(x, -x) (arith) — math.absf would lower to a
// __nv_fabsf libdevice extern (ptxas dead end).
// CHECK:         arith.maxnumf
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @tvd(%P: tensor<4x8xf32>, %Q: tensor<4x8xf32>,
               %half: tensor<f32>, %nhalf: tensor<f32>, %scale: tensor<f32>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %pos = intent.map_binary %half, %scale {fn = "mul"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %neg = intent.map_binary %nhalf, %scale {fn = "mul"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %cond = intent.map_binary %P, %Q {fn = "gt"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
  %grad = intent.select %cond, %pos, %neg : (tensor<4x8xi1>, tensor<f32>, tensor<f32>) -> tensor<4x8xf32>
  %diff = intent.map_binary %P, %Q {fn = "sub"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %ad = intent.map_unary %diff {fn = "abs"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %loss = intent.map_binary %half, %ad {fn = "mul"} : (tensor<f32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %grad, %loss : tensor<4x8xf32>, tensor<4x8xf32>
}
