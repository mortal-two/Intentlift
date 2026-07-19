// RUN: intent-opt -convert-intent-attention-backward-to-gpu %s | FileCheck %s
// RUN: intent-opt "-convert-intent-attention-backward-to-gpu=flat-shared-d-pad=0 flat-shared-pds-pad=0" %s | FileCheck %s --check-prefix=ZERO-PAD
// RUN: intent-opt "-convert-intent-attention-backward-to-gpu=flat-dq-init=prelaunch" %s | FileCheck %s --check-prefix=PRELAUNCH
// RUN: not intent-opt "-convert-intent-attention-backward-to-gpu=flat-shared-d-pad=4" %s 2>&1 | FileCheck %s --check-prefix=BAD-PAD
// RUN: not intent-opt "-convert-intent-attention-backward-to-gpu=flat-dq-init=hidden" %s 2>&1 | FileCheck %s --check-prefix=BAD-DQ-INIT

// The symbol is intentionally unrelated to FlashAttention.  Selection must be
// the complete rank-4 five-contraction dataflow, never a function-name match.
module {
  func.func @arbitrary_rank4_backward(%q: tensor<1x128x1x32xf16>, %k: tensor<1x128x1x32xf16>, %v: tensor<1x128x1x32xf16>, %do: tensor<1x128x1x32xf16>, %dq0: tensor<1x128x1x32xf32>, %lse: tensor<1x1x128xf32>, %delta: tensor<1x1x128xf32>, %scale: tensor<f32>) -> (tensor<1x128x1x32xf32>, tensor<1x128x1x32xf32>, tensor<1x128x1x32xf32>) {
    %q32 = intent.cast %q {to_dtype = "f32"} : (tensor<1x128x1x32xf16>) -> tensor<1x128x1x32xf32>
    %k32 = intent.cast %k {to_dtype = "f32"} : (tensor<1x128x1x32xf16>) -> tensor<1x128x1x32xf32>
    %v32 = intent.cast %v {to_dtype = "f32"} : (tensor<1x128x1x32xf16>) -> tensor<1x128x1x32xf32>
    %do32 = intent.cast %do {to_dtype = "f32"} : (tensor<1x128x1x32xf16>) -> tensor<1x128x1x32xf32>
    %qk = intent.contraction %q32, %k32 {subscripts = "bmhd,bnhd->nbhm"} : (tensor<1x128x1x32xf32>, tensor<1x128x1x32xf32>) -> tensor<128x1x1x128xf32>
    %scaled_qk = intent.map_binary %qk, %scale {fn = "mul"} : (tensor<128x1x1x128xf32>, tensor<f32>) -> tensor<128x1x1x128xf32>
    %centered_qk = intent.map_binary %scaled_qk, %lse {fn = "sub"} : (tensor<128x1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<128x1x1x128xf32>
    %p = intent.map_unary %centered_qk {fn = "exp"} : (tensor<128x1x1x128xf32>) -> tensor<128x1x1x128xf32>
    %p16 = intent.cast %p {to_dtype = "f16"} : (tensor<128x1x1x128xf32>) -> tensor<128x1x1x128xf16>
    %p32 = intent.cast %p16 {to_dtype = "f32"} : (tensor<128x1x1x128xf16>) -> tensor<128x1x1x128xf32>
    %dv = intent.contraction %p32, %do32 {subscripts = "nbhm,bmhd->bnhd"} : (tensor<128x1x1x128xf32>, tensor<1x128x1x32xf32>) -> tensor<1x128x1x32xf32>
    %dp = intent.contraction %do32, %v32 {subscripts = "bmhd,bnhd->nbhm"} : (tensor<1x128x1x32xf32>, tensor<1x128x1x32xf32>) -> tensor<128x1x1x128xf32>
    %dp_centered = intent.map_binary %dp, %delta {fn = "sub"} : (tensor<128x1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<128x1x1x128xf32>
    %ds_unscaled = intent.map_binary %p, %dp_centered {fn = "mul"} : (tensor<128x1x1x128xf32>, tensor<128x1x1x128xf32>) -> tensor<128x1x1x128xf32>
    %ds_scaled = intent.map_binary %ds_unscaled, %scale {fn = "mul"} : (tensor<128x1x1x128xf32>, tensor<f32>) -> tensor<128x1x1x128xf32>
    %ds16 = intent.cast %ds_scaled {to_dtype = "f16"} : (tensor<128x1x1x128xf32>) -> tensor<128x1x1x128xf16>
    %ds32 = intent.cast %ds16 {to_dtype = "f32"} : (tensor<128x1x1x128xf16>) -> tensor<128x1x1x128xf32>
    %dk = intent.contraction %ds32, %q32 {subscripts = "nbhm,bmhd->bnhd"} : (tensor<128x1x1x128xf32>, tensor<1x128x1x32xf32>) -> tensor<1x128x1x32xf32>
    %dq_update = intent.contraction %ds32, %k32 {subscripts = "nbhm,bnhd->bmhd"} : (tensor<128x1x1x128xf32>, tensor<1x128x1x32xf32>) -> tensor<1x128x1x32xf32>
    %dq = intent.map_binary %dq0, %dq_update {fn = "add"} : (tensor<1x128x1x32xf32>, tensor<1x128x1x32xf32>) -> tensor<1x128x1x32xf32>
    return %dq, %dk, %dv : tensor<1x128x1x32xf32>, tensor<1x128x1x32xf32>, tensor<1x128x1x32xf32>
  }
}

// CHECK-LABEL: func.func @arbitrary_rank4_backward(
// CHECK-SAME: memref<1x128x1x32xf16>
// CHECK-SAME: memref<f32>
// CHECK-SAME: memref<1x128x1x32xf32>
// CHECK: arith.constant 75776 : i32
// CHECK: gpu.launch
// CHECK-COUNT-8: memref.assume_alignment {{.*}}, 16
// CHECK: vector.load {{.*}} : memref<1x128x1x32xf32>, vector<4xf32>
// CHECK: nvgpu.ldmatrix {{.*}} {numTiles = 4 : i32, transpose = false}
// CHECK: nvgpu.mma.sync
// CHECK: vector.store {{.*}} : memref<1x128x1x32xf32>, vector<2xf32>
// CHECK-NOT: intent.

// ZERO-PAD: arith.constant 65536 : i32
// PRELAUNCH-LABEL: func.func @arbitrary_rank4_backward(
// PRELAUNCH: memref.copy [[DQ_INITIAL:%arg[0-9]+]], [[DQ:%arg[0-9]+]] : memref<1x128x1x32xf32> to memref<1x128x1x32xf32>
// PRELAUNCH-NEXT: gpu.launch
// PRELAUNCH-COUNT-7: memref.assume_alignment {{.*}}, 16
// PRELAUNCH-NOT: vector.load [[DQ_INITIAL]]
// PRELAUNCH: nvgpu.ldmatrix
// BAD-PAD: error: flash-bwd flat shared padding must be a multiple of eight f16 elements
// BAD-DQ-INIT: error: flash-bwd flat-dq-init must be 'in-kernel' or 'prelaunch'
