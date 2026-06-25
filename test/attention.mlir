// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s
// Naive (unfused) scaled dot-product attention as an intent op-graph — the
// softmax-attention bring-up. Mathematically identical to FlashAttention's
// output (NOT the fused online-softmax algorithm). Q is pre-scaled by
// 1/sqrt(d) on the host, so the graph is pure softmax(QKᵀ)V:
//   S = Q·Kᵀ -> rowmax -> S-m -> exp -> rowsum -> P/l -> P·V
// Every op here executes on the 5090D individually; this graph orchestrates
// them. Shapes: 4 queries, 8 keys, head dim 16.
// CHECK-LABEL: func.func @attention
// CHECK-NOT: intent.
// CHECK: linalg.generic
func.func @attention(%q: tensor<4x16xf32>, %k: tensor<8x16xf32>,
                     %v: tensor<8x16xf32>) -> tensor<4x16xf32> {
  // S = Q · Kᵀ  (contraction over the head dim d) -> [4 x 8]
  %s = intent.contraction %q, %k {subscripts = "qd,kd->qk"}
      : (tensor<4x16xf32>, tensor<8x16xf32>) -> tensor<4x8xf32>
  // row max (keepdims) for numerically-stable softmax -> [4 x 1]
  %m = intent.reduce %s {fn = "max", axes = [1], assoc = "associative"}
      : (tensor<4x8xf32>) -> tensor<4x1xf32>
  // S - m  (broadcast on the key axis)
  %sm = intent.map_binary %s, %m {fn = "sub"}
      : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  // P = exp(S - m)  (native ex2.approx on device)
  %p = intent.map_unary %sm {fn = "exp"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // l = rowsum(P) (keepdims) -> [4 x 1]
  %l = intent.reduce %p {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<4x8xf32>) -> tensor<4x1xf32>
  // P / l  (broadcast divide) -> normalized attention weights
  %pn = intent.map_binary %p, %l {fn = "div"}
      : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  // O = P · V  (contraction over the key axis k) -> [4 x 16]
  %o = intent.contraction %pn, %v {subscripts = "qk,kd->qd"}
      : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %o : tensor<4x16xf32>
}
