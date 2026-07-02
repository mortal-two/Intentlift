// RUN: intent-opt -lower-gpu-math-to-nvvm %s | FileCheck %s

// The NVVM codegen peephole for the C0 trig/exp2 fns: scalar f32 math.cos/sin
// become the native cos/sin.approx.f PTX ops, and math.exp2 the ex2.approx.ftz.f
// op — never a __nv_cosf/__nv_sinf/__nv_exp2f libdevice extern (ptxas dead end).

// CHECK-LABEL: func.func @c
// CHECK: llvm.call_intrinsic "llvm.nvvm.cos.approx.f"
// CHECK-NOT: math.cos
func.func @c(%x: f32) -> f32 {
  %r = math.cos %x : f32
  return %r : f32
}

// -----

// CHECK-LABEL: func.func @s
// CHECK: llvm.call_intrinsic "llvm.nvvm.sin.approx.f"
// CHECK-NOT: math.sin
func.func @s(%x: f32) -> f32 {
  %r = math.sin %x : f32
  return %r : f32
}

// -----

// exp2 is the native 2^x op with NO log2(e) prescale (unlike exp).
// CHECK-LABEL: func.func @e2
// CHECK: llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"
// CHECK-NOT: math.exp2
func.func @e2(%x: f32) -> f32 {
  %r = math.exp2 %x : f32
  return %r : f32
}

// -----

// Non-f32 is left alone (the approx PTX ops are f32-only).
// CHECK-LABEL: func.func @cd
// CHECK: math.cos
func.func @cd(%x: f64) -> f64 {
  %r = math.cos %x : f64
  return %r : f64
}
