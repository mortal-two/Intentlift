// RUN: intent-opt -convert-intent-flash-mma-to-gpu %s | FileCheck %s

// M8c: the FlashAttention forward as one intent.loop lowered to ONE tensor-core
// gpu.launch with the online-softmax state REGISTER-RESIDENT via nvgpu.mma.sync
// (the register-level sibling of -convert-intent-loop-contraction-to-gpu, which
// round-trips O/S/P/m/l through dynamic shared). Only Q/K/V touch shared; S is a
// per-warp vector<2x2xf32> accumulator, the per-row rowmax/rowsum are quad
// gpu.shuffle xor, P is assembled into the PV A operand straight from the S
// accumulators, and the O accumulators ride the key loop as scf.for iter_args.
// No intent.* survives. M8d: the K/V staging is a cp.async multistage pipeline
// (nvgpu.device_async_copy/create_group/wait, default num-stages=2).
//
// CHECK-LABEL: func.func @flash_fwd(
// CHECK-SAME:    %{{.*}}: memref<128x64xf16>, %{{.*}}: memref<128x64xf16>, %{{.*}}: memref<128x64xf16>, %{{.*}}: f32, %{{.*}}: memref<128x64xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         nvgpu.device_async_copy
// CHECK:         nvgpu.device_async_create_group
// CHECK:         scf.for
// CHECK:           nvgpu.device_async_wait
// CHECK:           nvgpu.ldmatrix
// CHECK:           nvgpu.mma.sync
// CHECK:           gpu.shuffle xor
// CHECK:           math.exp
// CHECK:           gpu.terminator
// CHECK-NOT:     intent.loop
// CHECK-NOT:     intent.contraction
// CHECK-NOT:     intent.reduce
// CHECK-NOT:     gpu.subgroup_mma
func.func @flash_fwd(%Q: tensor<128x64xf16>, %K: tensor<128x64xf16>,
                     %V: tensor<128x64xf16>, %scale: f32) -> tensor<128x64xf32> {
  %m0 = arith.constant dense<0xFF800000> : tensor<128x1xf32>
  %l0 = arith.constant dense<0.0> : tensor<128x1xf32>
  %O0 = arith.constant dense<0.0> : tensor<128x64xf32>
  %r:3 = intent.loop scan(%K, %V : tensor<128x64xf16>, tensor<128x64xf16>)
                     init(%m0, %l0, %O0 : tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x64xf32>)
                     {scan_axes = [0, 0], tile_hint = 64 : i64}
                     -> tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x64xf32> do {
  ^bb0(%Kj: tensor<64x64xf16>, %Vj: tensor<64x64xf16>,
       %m: tensor<128x1xf32>, %l: tensor<128x1xf32>, %Oacc: tensor<128x64xf32>):
    %S = intent.contraction %Q, %Kj {subscripts = "qd,kd->qk"} : (tensor<128x64xf16>, tensor<64x64xf16>) -> tensor<128x64xf32>
    %scsplat = tensor.splat %scale : tensor<128x64xf32>
    %Ss = intent.map_binary %S, %scsplat {fn = "mul"} : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %mb = intent.reduce %Ss {fn = "max", axes = [1], assoc = "associative"} : (tensor<128x64xf32>) -> tensor<128x1xf32>
    %mn = intent.map_binary %m, %mb {fn = "max"} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    %Sc = intent.map_binary %Ss, %mn {fn = "sub"} : (tensor<128x64xf32>, tensor<128x1xf32>) -> tensor<128x64xf32>
    %P = intent.map_unary %Sc {fn = "exp"} : (tensor<128x64xf32>) -> tensor<128x64xf32>
    %co = intent.map_binary %m, %mn {fn = "sub"} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    %al = intent.map_unary %co {fn = "exp"} : (tensor<128x1xf32>) -> tensor<128x1xf32>
    %ps = intent.reduce %P {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<128x64xf32>) -> tensor<128x1xf32>
    %ls = intent.map_binary %l, %al {fn = "mul"} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    %ln = intent.map_binary %ls, %ps {fn = "add"} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
    %PV = intent.contraction %P, %Vj {subscripts = "qk,kd->qd"} : (tensor<128x64xf32>, tensor<64x64xf16>) -> tensor<128x64xf32>
    %Os = intent.map_binary %Oacc, %al {fn = "mul"} : (tensor<128x64xf32>, tensor<128x1xf32>) -> tensor<128x64xf32>
    %On = intent.map_binary %Os, %PV {fn = "add"} : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    intent.yield %mn, %ln, %On : tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x64xf32>
  }
  %O = intent.map_binary %r#2, %r#1 {fn = "div"} : (tensor<128x64xf32>, tensor<128x1xf32>) -> tensor<128x64xf32>
  return %O : tensor<128x64xf32>
}
