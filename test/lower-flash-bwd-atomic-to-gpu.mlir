// RUN: intent-opt -convert-intent-attention-backward-to-gpu=dq-mode=atomic %s | FileCheck %s

// The flash-bwd ATOMIC DQ mode (SEQUENCE_PARALLEL=true). Same intent.loop as the
// serial mode, but grid parallel over key blocks: grid.x = cdiv(Sk,BN) = 2 (one
// PROGRAM per key block, NOT one block per head + an outer key loop). The dQ
// contribution is accumulated into pre-zeroed global DQ via memref.atomic_rmw
// addf (-> red.global.add.f32 on sm_120a) -- cross-program. dK/dV stay
// partitioned per key block (plain store). Only the INNER query scf.for remains
// (the outer key loop is elided -- the key tile is the program's block id).
//
// CHECK-LABEL: func.func @flash_bwd(
// CHECK:         gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %c2
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         scf.for
// CHECK-NOT:       scf.for
// CHECK:         memref.atomic_rmw addf
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
    %S = intent.contraction %Qt, %K {subscripts = "qd,kd->qk"} : (tensor<64x64xf16>, tensor<128x64xf16>) -> tensor<64x128xf32>
    %scsplat = tensor.splat %scale : tensor<64x128xf32>
    %Ss = intent.map_binary %S, %scsplat {fn = "mul"} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %Sm = intent.map_binary %Ss, %LSEt {fn = "sub"} : (tensor<64x128xf32>, tensor<64x1xf32>) -> tensor<64x128xf32>
    %P = intent.map_unary %Sm {fn = "exp"} : (tensor<64x128xf32>) -> tensor<64x128xf32>
    %dVc = intent.contraction %P, %dOt {subscripts = "qk,qd->kd"} : (tensor<64x128xf32>, tensor<64x64xf16>) -> tensor<128x64xf32>
    %dvn = intent.map_binary %dv, %dVc {fn = "add"} : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %dP = intent.contraction %dOt, %V {subscripts = "qd,kd->qk"} : (tensor<64x64xf16>, tensor<128x64xf16>) -> tensor<64x128xf32>
    %dPd = intent.map_binary %dP, %Deltat {fn = "sub"} : (tensor<64x128xf32>, tensor<64x1xf32>) -> tensor<64x128xf32>
    %Pds = intent.map_binary %P, %dPd {fn = "mul"} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %dS = intent.map_binary %Pds, %scsplat {fn = "mul"} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %dKc = intent.contraction %dS, %Qt {subscripts = "qk,qd->kd"} : (tensor<64x128xf32>, tensor<64x64xf16>) -> tensor<128x64xf32>
    %dkn = intent.map_binary %dk, %dKc {fn = "add"} : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %dQc = intent.contraction %dS, %K {subscripts = "qk,kd->qd"} : (tensor<64x128xf32>, tensor<128x64xf16>) -> tensor<64x64xf32>
    intent.yield %dkn, %dvn : tensor<128x64xf32>, tensor<128x64xf32>
  }
  return %r#0, %r#1 : tensor<128x64xf32>, tensor<128x64xf32>
}
