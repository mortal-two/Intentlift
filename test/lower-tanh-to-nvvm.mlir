// RUN: intent-opt -lower-gpu-math-to-nvvm %s | FileCheck %s

// Scalar f32 tanh must remain self-contained: use a sign-stable ex2/reciprocal
// form instead of the unresolved __nv_tanhf libdevice symbol.
// CHECK-LABEL: func.func @tanh_f32
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"
// CHECK: arith.subf
// CHECK: arith.addf
// CHECK: llvm.call_intrinsic "llvm.nvvm.rcp.approx.ftz.f"
// CHECK-NOT: math.tanh
func.func @tanh_f32(%x: f32) -> f32 {
  %r = math.tanh %x : f32
  return %r : f32
}

// -----

// The approximation is deliberately f32-only.
// CHECK-LABEL: func.func @tanh_f64
// CHECK: math.tanh
func.func @tanh_f64(%x: f64) -> f64 {
  %r = math.tanh %x : f64
  return %r : f64
}
