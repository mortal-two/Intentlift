// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// cross_entropy (liger forward: per-row loss + in-place gradient), the common
// config (reduction='mean', no label_smoothing / z-loss / weight / ignore_index).
// One token's logits X[M,V] reduce to the log-sum-exp (max then sum-exp, like
// softmax); the target logit X[row, Y[row]] is gathered as a masked row-sum
// (sum(select(iota == Y, X, 0))); loss = (lse - ori)/n is a per-row scalar; the
// gradient (softmax - one-hot(Y))/n is written back in place over X. The whole
// DAG lowers to ONE read-once block-cooperative gpu.launch: three grouped
// warp/shared reductions (max, sum, sum), the register tile reused for the grad
// epilogue, an integer arith.cmpi eq for the one-hot, arith.select, math.log for
// the lse, a thread-0 guarded loss store to memref<Mxf32>, and the grad store
// back into the X memref (in place). No new pass: this is the M1 register-tile
// scheduler taught to evaluate iota/select/cmpi inside emitAt.

// CHECK-LABEL: func.func @cross_entropy(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4xi32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<f32>, %{{.*}}: memref<4xf32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     scf.for
// CHECK:         gpu.shuffle xor
// CHECK:         math.exp
// CHECK:         gpu.shuffle xor
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         gpu.shuffle xor
// CHECK:         math.log
// CHECK:         scf.if
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4xf32>
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @cross_entropy(%X: tensor<4x8xf32>, %Y: tensor<4xi32>,
                         %zero: tensor<f32>, %one: tensor<f32>,
                         %ninv: tensor<f32>)
    -> (tensor<4x1xf32>, tensor<4x8xf32>) {
  // one-hot condition: column index == target class
  %iota = intent.iota : tensor<8xi32>
  %cmp = intent.map_binary %iota, %Y {fn = "eq"} : (tensor<8xi32>, tensor<4xi32>) -> tensor<4x8xi1>

  // log-sum-exp (safe softmax denominator)
  %m = intent.reduce %X {fn = "max", axes = [1], assoc = "associative"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %xm = intent.map_binary %X, %m {fn = "sub"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %ex = intent.map_unary %xm {fn = "exp"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %se = intent.reduce %ex {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  %logse = intent.map_unary %se {fn = "log"} : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %lse = intent.map_binary %m, %logse {fn = "add"} : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>

  // gather the target logit X[row, Y[row]] = sum(select(iota==Y, X, 0))
  %selo = intent.select %cmp, %X, %zero : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  %ori = intent.reduce %selo {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4x1xf32>

  // per-row loss = (lse - ori)/n
  %lnum = intent.map_binary %lse, %ori {fn = "sub"} : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
  %loss = intent.map_binary %lnum, %ninv {fn = "mul"} : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>

  // gradient = (softmax(X) - one-hot(Y))/n, written in place over X
  %xlse = intent.map_binary %X, %lse {fn = "sub"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  %g = intent.map_unary %xlse {fn = "exp"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %oh = intent.select %cmp, %one, %zero : (tensor<4x8xi1>, tensor<f32>, tensor<f32>) -> tensor<4x8xf32>
  %gnum = intent.map_binary %g, %oh {fn = "sub"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %grad = intent.map_binary %gnum, %ninv {fn = "mul"} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>

  return %loss, %grad : tensor<4x1xf32>, tensor<4x8xf32>
}
