// RUN: intent-opt -convert-intent-attention-backward-to-gpu %s | FileCheck %s

// The fused FlashAttention BACKWARD as one intent.loop scanning the QUERY axis
// (scan Q/dO/LSE/Delta, carry the per-key-row gradients dk[Sk,D], dv[Sk,D]; body
// = recompute P = exp(scale*S - LSE) + the 5 contractions S=Q@K^T, dV=P^T@dO,
// dP=dO@V^T, dK=dS^T@Q, dQ=dS@K) lowers to ONE tiled tensor-core gpu.launch.
// The dk/dv accumulators + the recompute-P/dS transforms live in dynamic shared;
// the 5 matmuls run on WMMA. ONE block per (b,h): an OUTER scf.for over key
// tiles wrapping an INNER scf.for over query tiles, dQ read-modify-written into
// pre-zeroed global DQ (serial accumulation). The pass appends dQ/dK/dV
// destinations. No intent.* survives.
//
// CHECK-LABEL: func.func @flash_bwd(
// CHECK-SAME:    %{{.*}}: memref<128x64xf16>, %{{.*}}: memref<128x64xf16>, %{{.*}}: memref<128x64xf16>, %{{.*}}: memref<128x64xf16>, %{{.*}}: memref<128x1xf32>, %{{.*}}: memref<128x1xf32>, %{{.*}}: f32, %{{.*}}: memref<128x64xf32>, %{{.*}}: memref<128x64xf32>, %{{.*}}: memref<128x64xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             gpu.subgroup_mma_constant_matrix
// CHECK:             gpu.subgroup_mma_load_matrix
// CHECK:             gpu.subgroup_mma_compute
// CHECK:             math.exp
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.loop
// CHECK-NOT:     intent.contraction
func.func @flash_bwd(%Q: tensor<128x64xf16>, %K: tensor<128x64xf16>,
                     %V: tensor<128x64xf16>, %dO: tensor<128x64xf16>,
                     %LSE: tensor<128x1xf32>, %Delta: tensor<128x1xf32>,
                     %scale: f32)
    -> (tensor<128x64xf32>, tensor<128x64xf32>) {
  %dk0 = arith.constant dense<0.0> : tensor<128x64xf32>
  %dv0 = arith.constant dense<0.0> : tensor<128x64xf32>
  %r:2 = intent.loop scan(%Q, %dO, %LSE, %Delta : tensor<128x64xf16>, tensor<128x64xf16>, tensor<128x1xf32>, tensor<128x1xf32>)
                     init(%dk0, %dv0 : tensor<128x64xf32>, tensor<128x64xf32>)
                     {scan_axes = [0, 0, 0, 0], tile_hint = 64 : i64}
                     -> tensor<128x64xf32>, tensor<128x64xf32> do {
  ^bb0(%Qt: tensor<64x64xf16>, %dOt: tensor<64x64xf16>, %LSEt: tensor<64x1xf32>,
       %Deltat: tensor<64x1xf32>, %dk: tensor<128x64xf32>, %dv: tensor<128x64xf32>):
    // recompute P = exp(scale*S - LSE)
    %S = intent.contraction %Qt, %K {subscripts = "qd,kd->qk"} : (tensor<64x64xf16>, tensor<128x64xf16>) -> tensor<64x128xf32>
    %scsplat = tensor.splat %scale : tensor<64x128xf32>
    %Ss = intent.map_binary %S, %scsplat {fn = "mul"} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %Sm = intent.map_binary %Ss, %LSEt {fn = "sub"} : (tensor<64x128xf32>, tensor<64x1xf32>) -> tensor<64x128xf32>
    %P = intent.map_unary %Sm {fn = "exp"} : (tensor<64x128xf32>) -> tensor<64x128xf32>
    // dV += P^T @ dO
    %dVc = intent.contraction %P, %dOt {subscripts = "qk,qd->kd"} : (tensor<64x128xf32>, tensor<64x64xf16>) -> tensor<128x64xf32>
    %dvn = intent.map_binary %dv, %dVc {fn = "add"} : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    // dP = dO @ V^T ; dS = P*(dP - Delta)*scale
    %dP = intent.contraction %dOt, %V {subscripts = "qd,kd->qk"} : (tensor<64x64xf16>, tensor<128x64xf16>) -> tensor<64x128xf32>
    %dPd = intent.map_binary %dP, %Deltat {fn = "sub"} : (tensor<64x128xf32>, tensor<64x1xf32>) -> tensor<64x128xf32>
    %Pds = intent.map_binary %P, %dPd {fn = "mul"} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %dS = intent.map_binary %Pds, %scsplat {fn = "mul"} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // dK += dS^T @ Q
    %dKc = intent.contraction %dS, %Qt {subscripts = "qk,qd->kd"} : (tensor<64x128xf32>, tensor<64x64xf16>) -> tensor<128x64xf32>
    %dkn = intent.map_binary %dk, %dKc {fn = "add"} : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    // dQ = dS @ K  (scatter-accumulated into DQ[start_m] by the lowering)
    %dQc = intent.contraction %dS, %K {subscripts = "qk,kd->qd"} : (tensor<64x128xf32>, tensor<128x64xf16>) -> tensor<64x64xf32>
    intent.yield %dkn, %dvn : tensor<128x64xf32>, tensor<128x64xf32>
  }
  return %r#0, %r#1 : tensor<128x64xf32>, tensor<128x64xf32>
}
