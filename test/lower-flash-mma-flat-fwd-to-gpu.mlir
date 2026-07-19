// RUN: intent-opt -convert-intent-flash-mma-to-gpu %s | FileCheck %s --implicit-check-not=intent. --implicit-check-not=gpu.launch

// The live P4 B=H=1 flat FlashAttention graph is recognized by dataflow, not by
// symbol name.  The rewrite preserves the three destination outputs and emits
// exactly one register-resident tensor-core launch.  The rank-0 scale is loaded
// legally inside that launch, and no intent operation survives.
//
// CHECK-LABEL: func.func @arbitrary_symbol_42(
// CHECK-SAME: %{{.*}}: memref<1x128x1x64xf16>, %{{.*}}: memref<1x128x1x64xf16>, %{{.*}}: memref<1x128x1x64xf16>, %{{.*}}: memref<f32>, %{{.*}}: memref<1x128x1x64xf16>, %{{.*}}: memref<1x1x128xf32>, %{{.*}}: memref<1x1x128xf32>)
// CHECK: gpu.launch
// CHECK: memref.load %{{.*}}[] : memref<f32>
// CHECK: nvgpu.device_async_copy
// CHECK: nvgpu.ldmatrix
// CHECK: nvgpu.mma.sync
// CHECK: gpu.shuffle xor
// CHECK: math.log
// CHECK: arith.divf
// CHECK: arith.truncf

module {
  func.func @arbitrary_symbol_42(
      %Q: tensor<1x128x1x64xf16>,
      %K: tensor<1x128x1x64xf16>,
      %V: tensor<1x128x1x64xf16>,
      %softmax_scale: tensor<f32>)
      -> (tensor<1x128x1x64xf16>, tensor<1x1x128xf32>,
          tensor<1x1x128xf32>) {
    %qk_raw = intent.contraction %Q, %K {subscripts = "bmhd,bnhd->nbhm"} : (tensor<1x128x1x64xf16>, tensor<1x128x1x64xf16>) -> tensor<128x1x1x128xf32>
    %qk_row_max = intent.reduce %qk_raw {fn = "max", axes = [0], assoc = "associative"} : (tensor<128x1x1x128xf32>) -> tensor<1x1x128xf32>
    %m = intent.map_binary %qk_row_max, %softmax_scale {fn = "mul"} : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1x128xf32>
    %scaled_qk = intent.map_binary %qk_raw, %softmax_scale {fn = "mul"} : (tensor<128x1x1x128xf32>, tensor<f32>) -> tensor<128x1x1x128xf32>
    %shifted_qk = intent.map_binary %scaled_qk, %m {fn = "sub"} : (tensor<128x1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<128x1x1x128xf32>
    %p_f32 = intent.map_unary %shifted_qk {fn = "exp"} : (tensor<128x1x1x128xf32>) -> tensor<128x1x1x128xf32>
    %denominator = intent.reduce %p_f32 {fn = "sum", axes = [0], assoc = "float_nondeterministic"} : (tensor<128x1x1x128xf32>) -> tensor<1x1x128xf32>
    %log_denominator = intent.map_unary %denominator {fn = "log"} : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %lse = intent.map_binary %m, %log_denominator {fn = "add"} : (tensor<1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %m_minus_lse = intent.map_binary %m, %lse {fn = "sub"} : (tensor<1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %tmp = intent.map_unary %m_minus_lse {fn = "exp"} : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %p_f16 = intent.cast %p_f32 {to_dtype = "f16"} : (tensor<128x1x1x128xf32>) -> tensor<128x1x1x128xf16>
    %weighted_unscaled = intent.contraction %p_f16, %V {subscripts = "nbhm,bnhd->dbhm"} : (tensor<128x1x1x128xf16>, tensor<1x128x1x64xf16>) -> tensor<64x1x1x128xf32>
    %weighted_scaled = intent.map_binary %weighted_unscaled, %tmp {fn = "mul"} : (tensor<64x1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<64x1x1x128xf32>
    %out_f32 = intent.transpose %weighted_scaled {perm = [1, 3, 2, 0]} : (tensor<64x1x1x128xf32>) -> tensor<1x128x1x64xf32>
    %out = intent.cast %out_f32 {to_dtype = "f16"} : (tensor<1x128x1x64xf32>) -> tensor<1x128x1x64xf16>
    return %out, %lse, %tmp : tensor<1x128x1x64xf16>, tensor<1x1x128xf32>, tensor<1x1x128xf32>
  }
}
