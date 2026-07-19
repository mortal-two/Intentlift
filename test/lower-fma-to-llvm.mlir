// RUN: intent-opt -lower-gpu-math-to-nvvm %s | FileCheck %s

// Scalar f32 math.fma must avoid the unresolved __nv_fmaf libdevice path and
// become LLVM's intrinsic, which the NVIDIA backend emits as fma.rn.f32.
// CHECK-LABEL: func.func @fma_f32
// CHECK: llvm.call_intrinsic "llvm.fma"
// CHECK-NOT: math.fma
func.func @fma_f32(%a: f32, %b: f32, %c: f32) -> f32 {
  %r = math.fma %a, %b, %c : f32
  return %r : f32
}

// -----

// This target peephole is deliberately f32-only.
// CHECK-LABEL: func.func @fma_f64
// CHECK: math.fma
func.func @fma_f64(%a: f64, %b: f64, %c: f64) -> f64 {
  %r = math.fma %a, %b, %c : f64
  return %r : f64
}
