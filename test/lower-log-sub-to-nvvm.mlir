// RUN: intent-opt -lower-gpu-math-to-nvvm %s | FileCheck %s

// A single-use natural log immediately subtracted by another scalar can fold
// the ln(2) change of base and subtraction into one target FMA.
// CHECK-LABEL: func.func @log_sub
// CHECK: llvm.call_intrinsic "llvm.nvvm.lg2.approx.f"
// CHECK: arith.negf
// CHECK: llvm.call_intrinsic "llvm.fma"
// CHECK-NOT: math.log
// CHECK-NOT: arith.subf
func.func @log_sub(%x: f32, %y: f32) -> f32 {
  %log = math.log %x : f32
  %result = arith.subf %log, %y : f32
  return %result : f32
}

// -----

// Shared log values retain the generic rewrite because duplicating the native
// log approximation would be a regression.
// CHECK-LABEL: func.func @shared_log
// CHECK: llvm.call_intrinsic "llvm.nvvm.lg2.approx.f"
// CHECK: arith.mulf
// CHECK: arith.subf
func.func @shared_log(%x: f32, %y: f32) -> (f32, f32) {
  %log = math.log %x : f32
  %difference = arith.subf %log, %y : f32
  return %log, %difference : f32, f32
}
