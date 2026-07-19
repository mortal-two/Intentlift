// RUN: intent-opt -lower-gpu-math-to-nvvm %s | FileCheck %s

// These legal Intent math operations must not become unresolved NVIDIA
// libdevice symbols in the generic CUDA pipeline.

// CHECK-LABEL: func.func @abs_f32
// CHECK: llvm.call_intrinsic "llvm.fabs"
// CHECK-NOT: math.absf
func.func @abs_f32(%x: f32) -> f32 {
  %r = math.absf %x : f32
  return %r : f32
}

// -----

// CHECK-LABEL: func.func @atan_f32
// CHECK: arith.cmpf olt
// CHECK: arith.divf
// CHECK: arith.select
// CHECK-NOT: math.atan
func.func @atan_f32(%x: f32) -> f32 {
  %r = math.atan %x : f32
  return %r : f32
}

// -----

// CHECK-LABEL: func.func @atan2_f32
// CHECK: arith.bitcast
// CHECK: arith.cmpi slt
// CHECK: arith.divf
// CHECK: arith.cmpf olt
// CHECK: arith.cmpf uno
// CHECK: arith.select
// CHECK-NOT: math.atan2
func.func @atan2_f32(%y: f32, %x: f32) -> f32 {
  %r = math.atan2 %y, %x : f32
  return %r : f32
}

// -----

// CHECK-LABEL: func.func @erf_f32
// CHECK: arith.divf
// CHECK: llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"
// CHECK: arith.select
// CHECK-NOT: math.erf
func.func @erf_f32(%x: f32) -> f32 {
  %r = math.erf %x : f32
  return %r : f32
}

// -----

// CHECK-LABEL: func.func @pow_f32
// CHECK: arith.bitcast
// CHECK: arith.cmpi
// CHECK: arith.sitofp
// CHECK: llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"
// CHECK: arith.select
// CHECK-NOT: math.powf
func.func @pow_f32(%base: f32, %exponent: f32) -> f32 {
  %r = math.powf %base, %exponent : f32
  return %r : f32
}

// -----

// Half storage is widened to the shared f32 backend and rounded back only after
// the semantic math operation.
// CHECK-LABEL: func.func @exp_f16
// CHECK: arith.extf {{.*}} : f16 to f32
// CHECK: llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"
// CHECK: arith.truncf {{.*}} : f32 to f16
// CHECK-NOT: math.exp
func.func @exp_f16(%x: f16) -> f16 {
  %r = math.exp %x : f16
  return %r : f16
}

// -----

// CHECK-LABEL: func.func @erf_bf16
// CHECK: arith.extf {{.*}} : bf16 to f32
// CHECK: llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK-NOT: math.erf
func.func @erf_bf16(%x: bf16) -> bf16 {
  %r = math.erf %x : bf16
  return %r : bf16
}

// -----

// CHECK-LABEL: func.func @pow_f16
// CHECK: arith.extf {{.*}} : f16 to f32
// CHECK: llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"
// CHECK: arith.truncf {{.*}} : f32 to f16
// CHECK-NOT: math.powf
func.func @pow_f16(%base: f16, %exponent: f16) -> f16 {
  %r = math.powf %base, %exponent : f16
  return %r : f16
}

// -----

// f64 forms remain outside this f32-compute policy.
// CHECK-LABEL: func.func @atan_f64
// CHECK: math.atan
func.func @atan_f64(%x: f64) -> f64 {
  %r = math.atan %x : f64
  return %r : f64
}
