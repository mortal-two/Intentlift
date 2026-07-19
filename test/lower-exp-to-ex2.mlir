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

// -----

// f64 log cannot use the f32 native lg2 instruction and standalone ptxas cannot
// resolve __nv_log. It is therefore lowered to an IEEE range reduction plus a
// self-contained f64 atanh polynomial.
// CHECK-LABEL: func.func @ld
// CHECK: arith.bitcast
// CHECK: arith.shrui
// CHECK: arith.sitofp
// CHECK-NOT: math.log
func.func @ld(%x: f64) -> f64 {
  %r = math.log %x : f64
  return %r : f64
}

// -----

// The normalized sigmoid denominator uses the matching native approximate
// reciprocal only after exp has become ex2.approx.
// CHECK-LABEL: func.func @sigmoid_reciprocal
// CHECK: llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"
// CHECK: arith.addf
// CHECK: llvm.call_intrinsic "llvm.nvvm.rcp.approx.ftz.f"
// CHECK-NOT: arith.divf
func.func @sigmoid_reciprocal(%x: f32) -> f32 {
  %negative = arith.negf %x : f32
  %e = math.exp %negative : f32
  %one = arith.constant 1.0 : f32
  %denominator = arith.addf %e, %one : f32
  %result = arith.divf %one, %denominator : f32
  return %result : f32
}
