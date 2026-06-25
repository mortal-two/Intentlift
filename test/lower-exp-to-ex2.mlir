// RUN: intent-opt -lower-gpu-math-to-nvvm %s | FileCheck %s

// The NVVM codegen peephole: scalar f32 math.exp becomes 2^(x*log2e) via the
// native ex2.approx PTX instruction (no __nv_expf libdevice extern).
// CHECK-LABEL: func.func @e
// CHECK: arith.mulf
// CHECK: llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"
// CHECK-NOT: math.exp
func.func @e(%x: f32) -> f32 {
  %r = math.exp %x : f32
  return %r : f32
}

// -----

// Non-f32 exp is left alone (ex2.approx.ftz.f is f32-only).
// CHECK-LABEL: func.func @ed
// CHECK: math.exp
func.func @ed(%x: f64) -> f64 {
  %r = math.exp %x : f64
  return %r : f64
}
