// RUN: intent-opt -convert-intent-loop-contraction-to-gpu %s | FileCheck %s

// The xformers _fwd_kernel_splitK forward: the M8a fused-flash online-softmax
// with an OUTER split-K axis. Each (q-tile, k-chunk) program runs the flash
// loop over ITS key chunk and writes a PARTIAL O[chunk,...] + LSE[chunk,...]
// (the LSE-write epilogue: m + log(l), instead of the plain O = Oacc/l divide).
//
// The split dim Z is the leading dim of K/V/O/LSE (batched-RHS contractions
// "qd,zkd->zqk" / "zqk,zkd->zqd"; Q stays rank-2, shared across splits). It
// lowers to ONE tensor-core gpu.launch with grid.y = Z (blockIdx.y selects the
// chunk; K/V/O/LSE global addressing gets the leading split index). Reuses the
// M8a body verbatim (Q stage, two WMMA matmuls, cooperative shared-memory online
// softmax + per-row alpha rescale); only the global addressing + the LSE epilogue
// are new. No intent.* survives.
//
// CHECK-LABEL: func.func @splitk_fwd(
// CHECK-SAME:    %{{.*}}: memref<128x64xf16>, %{{.*}}: memref<2x64x64xf16>, %{{.*}}: memref<2x64x64xf16>, %{{.*}}: f32, %{{.*}}: memref<2x128x64xf32>, %{{.*}}: memref<2x128x1xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         scf.for
// CHECK:           gpu.subgroup_mma_compute
// CHECK:           math.exp
// CHECK:           gpu.subgroup_mma_compute
// CHECK:         math.log
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.loop
// CHECK-NOT:     intent.contraction
// CHECK-NOT:     intent.reduce
func.func @splitk_fwd(%Q: tensor<128x64xf16>, %K: tensor<2x64x64xf16>,
                      %V: tensor<2x64x64xf16>, %scale: f32)
    -> (tensor<2x128x64xf32>, tensor<2x128x1xf32>) {
  %m0 = arith.constant dense<0xFF800000> : tensor<2x128x1xf32>
  %l0 = arith.constant dense<0.0> : tensor<2x128x1xf32>
  %O0 = arith.constant dense<0.0> : tensor<2x128x64xf32>
  %r:3 = intent.loop scan(%K, %V : tensor<2x64x64xf16>, tensor<2x64x64xf16>)
                     init(%m0, %l0, %O0 : tensor<2x128x1xf32>, tensor<2x128x1xf32>, tensor<2x128x64xf32>)
                     {scan_axes = [1, 1], tile_hint = 64 : i64}
                     -> tensor<2x128x1xf32>, tensor<2x128x1xf32>, tensor<2x128x64xf32> do {
  ^bb0(%Kj: tensor<2x64x64xf16>, %Vj: tensor<2x64x64xf16>,
       %m: tensor<2x128x1xf32>, %l: tensor<2x128x1xf32>, %Oacc: tensor<2x128x64xf32>):
    %S = intent.contraction %Q, %Kj {subscripts = "qd,zkd->zqk"} : (tensor<128x64xf16>, tensor<2x64x64xf16>) -> tensor<2x128x64xf32>
    %scsplat = tensor.splat %scale : tensor<2x128x64xf32>
    %Ss = intent.map_binary %S, %scsplat {fn = "mul"} : (tensor<2x128x64xf32>, tensor<2x128x64xf32>) -> tensor<2x128x64xf32>
    %mb = intent.reduce %Ss {fn = "max", axes = [2], assoc = "associative"} : (tensor<2x128x64xf32>) -> tensor<2x128x1xf32>
    %mn = intent.map_binary %m, %mb {fn = "max"} : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %Sc = intent.map_binary %Ss, %mn {fn = "sub"} : (tensor<2x128x64xf32>, tensor<2x128x1xf32>) -> tensor<2x128x64xf32>
    %P = intent.map_unary %Sc {fn = "exp"} : (tensor<2x128x64xf32>) -> tensor<2x128x64xf32>
    %co = intent.map_binary %m, %mn {fn = "sub"} : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %al = intent.map_unary %co {fn = "exp"} : (tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %ps = intent.reduce %P {fn = "sum", axes = [2], assoc = "float_nondeterministic"} : (tensor<2x128x64xf32>) -> tensor<2x128x1xf32>
    %ls = intent.map_binary %l, %al {fn = "mul"} : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %ln = intent.map_binary %ls, %ps {fn = "add"} : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %PV = intent.contraction %P, %Vj {subscripts = "zqk,zkd->zqd"} : (tensor<2x128x64xf32>, tensor<2x64x64xf16>) -> tensor<2x128x64xf32>
    %Os = intent.map_binary %Oacc, %al {fn = "mul"} : (tensor<2x128x64xf32>, tensor<2x128x1xf32>) -> tensor<2x128x64xf32>
    %On = intent.map_binary %Os, %PV {fn = "add"} : (tensor<2x128x64xf32>, tensor<2x128x64xf32>) -> tensor<2x128x64xf32>
    intent.yield %mn, %ln, %On : tensor<2x128x1xf32>, tensor<2x128x1xf32>, tensor<2x128x64xf32>
  }
  %O = intent.map_binary %r#2, %r#1 {fn = "div"} : (tensor<2x128x64xf32>, tensor<2x128x1xf32>) -> tensor<2x128x64xf32>
  %logl = intent.map_unary %r#1 {fn = "log"} : (tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
  %lse = intent.map_binary %r#0, %logl {fn = "add"} : (tensor<2x128x1xf32>, tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
  return %O, %lse : tensor<2x128x64xf32>, tensor<2x128x1xf32>
}
