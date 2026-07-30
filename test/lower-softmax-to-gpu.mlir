// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=4 approximate-div=true' %s | FileCheck %s --check-prefix=APPROX

// softmax: two sequential last-axis reduces (max then sum) with a max-shift exp
// between them, lowered to ONE read-once gpu.launch. Two cooperative passes ->
// two ordered gpu.all_reduce ops (maxnumf, then add); the register tile is loaded once
// and exp(x - max) is recomputed from it in the epilogue (matching the hand
// oracle). Same pass, same DAG scheduler as rms_norm — the second reduce is the
// only structural difference.
// CHECK-LABEL: func.func @softmax(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     scf.for
// CHECK:         gpu.all_reduce{{.*}}maxnumf
// CHECK:         math.exp
// CHECK:         gpu.all_reduce{{.*}}add
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
// APPROX-LABEL: func.func @softmax(
// APPROX-SAME: intent.backend.approximate_div_count = 1 : i64
// APPROX-COUNT-1: arith.divf {{.*}} {intent.cuda.approximate_div}
func.func @softmax(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %m = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %s = intent.map_binary %x, %m {fn = "sub"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %e = intent.map_unary %s {fn = "exp"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %l = intent.reduce %e {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %o = intent.map_binary %e, %l {fn = "div"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  return %o : tensor<4x8xf32>
}
