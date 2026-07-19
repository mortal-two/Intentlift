// RUN: intent-opt -lower-gpu-math-to-nvvm %s | FileCheck %s

// math.sqrt / math.rsqrt -> NVVM approx intrinsics. Same libdevice-dead-end fix
// as exp->ex2.approx: the stock GpuToNVVM path routes these to __nv_sqrtf /
// __nv_rsqrtf externs, which ptxas cannot resolve. Standalone sqrt graphs and
// normalization rsqrt fast paths both exercise this lowering.
// CHECK-LABEL: func.func @s
// CHECK: llvm.call_intrinsic "llvm.nvvm.sqrt.approx.f"
// CHECK-NOT: math.sqrt
func.func @s(%x: f32) -> f32 {
  %r = math.sqrt %x : f32
  return %r : f32
}

// -----

// CHECK-LABEL: func.func @rs
// CHECK: llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"
// CHECK-NOT: math.rsqrt
func.func @rs(%x: f32) -> f32 {
  %r = math.rsqrt %x : f32
  return %r : f32
}

// -----

// Non-f32 is left alone (the *.approx.f ops are f32-only).
// CHECK-LABEL: func.func @sd
// CHECK: math.sqrt
func.func @sd(%x: f64) -> f64 {
  %r = math.sqrt %x : f64
  return %r : f64
}
