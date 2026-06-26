// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// jsd (beta=0, forward KL): an AtomicOut loss scalar + a [M,N] RowOut dX from ONE
// read-once launch. X,Y are log-probs (<= 0), so Y_prob = exp(Y) is bounded in
// (0,1] and needs NO max-shift (the source's Y-Ymax shift is a no-op to fp
// precision; computing exp(Y) directly avoids an extra per-row max reduce). Per
// element: loss = Y_prob*(Y-X)*scale, dX = -Y_prob*scale (scale = 1/n_non_ignore,
// folded per element). The per-row loss sum (gpu.all_reduce add) is then folded
// cross-row into the rank-0 loss via thread-0 memref.atomic_rmw addf — never
// materializing/re-reading the full [BT,V] loss buffer the library sums on host.
// dX is written from the register tile. M=4, N=8, T=4.
// CHECK-LABEL: func.func @jsd(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK:         math.exp
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         arith.cmpi eq
// CHECK:         scf.if
// CHECK:         memref.atomic_rmw addf %{{.*}}, %{{.*}}[] : (f32, memref<f32>) -> f32
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @jsd(%X: tensor<4x8xf32>, %Y: tensor<4x8xf32>,
               %scale: tensor<f32>) -> (tensor<f32>, tensor<4x8xf32>) {
  %Yp = intent.map_unary %Y {fn = "exp"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %ymx = intent.map_binary %Y, %X {fn = "sub"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %le = intent.map_binary %Yp, %ymx {fn = "mul"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %les = intent.map_binary %le, %scale {fn = "mul"} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  %rowsum = intent.reduce %les {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4xf32>
  %total = intent.reduce %rowsum {fn = "sum", axes = [0], assoc = "float_nondeterministic"} : (tensor<4xf32>) -> tensor<f32>
  %nYp = intent.map_unary %Yp {fn = "neg"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %dX = intent.map_binary %nYp, %scale {fn = "mul"} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  return %total, %dX : tensor<f32>, tensor<4x8xf32>
}
