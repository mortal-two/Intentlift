// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s
// The flash body lowers fully to linalg/tensor/arith (no intent.* survives); a
// host driver runs it once per K-tile (see test_frontend_emit.py).
// CHECK-LABEL: func.func @flash_body
// CHECK: linalg.generic
// CHECK-NOT: intent.
//
// One K-tile step of the FlashAttention online-softmax recurrence, as an
// intent-dialect op-graph (the BODY of the flash `loop`). Inputs: the
// host-pre-scaled query Qs (so S = Qs·Kjᵀ already carries `scale`), the current
// K/V tile (Kj, Vj), and the carried state (running max m, running sum l,
// accumulator Oacc). Returns the updated carries (m_new, l_new, O_new).
//
// A host driver runs this once per K-tile, feeding the returned carries back in
// as the next call's m/l/Oacc (carries live in persistent CUDA buffers); after
// the last tile, O = Oacc / l. This is exactly how a NON-fused tiled flash runs:
// real kernels on the GPU, host-side loop control. Shapes: Sq=4, BK=2, D=4.
//
// This is the executable artifact the JSON→intent-MLIR emitter generates for the
// flash `intent.loop` body (see python/intentir/frontend/emit_mlir.py).
module {
  func.func @flash_body(%Qs: tensor<4x4xf32>, %Kj: tensor<2x4xf32>,
                        %Vj: tensor<2x4xf32>, %m: tensor<4x1xf32>,
                        %l: tensor<4x1xf32>, %Oacc: tensor<4x4xf32>)
      -> (tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x4xf32>) {
    // S = Qs · Kjᵀ            (contraction over the head dim d)            [4x2]
    %S = intent.contraction %Qs, %Kj {subscripts = "qd,kd->qk"}
        : (tensor<4x4xf32>, tensor<2x4xf32>) -> tensor<4x2xf32>
    // block row-max (keepdims)                                            [4x1]
    %m_blk = intent.reduce %S {fn = "max", axes = [1], assoc = "associative"}
        : (tensor<4x2xf32>) -> tensor<4x1xf32>
    // m_new = max(m, m_blk)                                               [4x1]
    %m_new = intent.map_binary %m, %m_blk {fn = "max"}
        : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    // P = exp(S - m_new)      (row broadcast of m_new)                    [4x2]
    %Sc = intent.map_binary %S, %m_new {fn = "sub"}
        : (tensor<4x2xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
    %P = intent.map_unary %Sc {fn = "exp"} : (tensor<4x2xf32>) -> tensor<4x2xf32>
    // alpha = exp(m - m_new)   (rescale factor for the running state)     [4x1]
    %corr = intent.map_binary %m, %m_new {fn = "sub"}
        : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %alpha = intent.map_unary %corr {fn = "exp"}
        : (tensor<4x1xf32>) -> tensor<4x1xf32>
    // l_new = alpha*l + rowsum(P)                                         [4x1]
    %p_sum = intent.reduce %P {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
        : (tensor<4x2xf32>) -> tensor<4x1xf32>
    %l_sc = intent.map_binary %l, %alpha {fn = "mul"}
        : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %l_new = intent.map_binary %l_sc, %p_sum {fn = "add"}
        : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    // O_new = alpha*Oacc + P·Vj                                           [4x4]
    %PV = intent.contraction %P, %Vj {subscripts = "qk,kd->qd"}
        : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
    %O_sc = intent.map_binary %Oacc, %alpha {fn = "mul"}
        : (tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
    %O_new = intent.map_binary %O_sc, %PV {fn = "add"}
        : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %m_new, %l_new, %O_new
        : tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x4xf32>
  }
}
