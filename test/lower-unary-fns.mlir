// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

// C0 coverage unblocker: the new scalar unary fns lower to arith/math (never a
// per-kernel branch), and never to a libdevice extern (that dead-end is fixed by
// the -lower-gpu-math-to-nvvm peephole; see lower-trig-exp2-to-nvvm.mlir).

// CHECK-LABEL: func.func @ucos
// CHECK: linalg.generic
// CHECK: math.cos
func.func @ucos(%a: tensor<256xf32>) -> tensor<256xf32> {
  %r = intent.map_unary %a {fn = "cos"} : (tensor<256xf32>) -> tensor<256xf32>
  return %r : tensor<256xf32>
}

// -----

// CHECK-LABEL: func.func @usin
// CHECK: math.sin
func.func @usin(%a: tensor<256xf32>) -> tensor<256xf32> {
  %r = intent.map_unary %a {fn = "sin"} : (tensor<256xf32>) -> tensor<256xf32>
  return %r : tensor<256xf32>
}

// -----

// tan = sin/cos (no math.tan / __nv_tanf).
// CHECK-LABEL: func.func @utan
// CHECK: math.sin
// CHECK: math.cos
// CHECK: arith.divf
func.func @utan(%a: tensor<256xf32>) -> tensor<256xf32> {
  %r = intent.map_unary %a {fn = "tan"} : (tensor<256xf32>) -> tensor<256xf32>
  return %r : tensor<256xf32>
}

// -----

// CHECK-LABEL: func.func @uexp2
// CHECK: math.exp2
func.func @uexp2(%a: tensor<256xf32>) -> tensor<256xf32> {
  %r = intent.map_unary %a {fn = "exp2"} : (tensor<256xf32>) -> tensor<256xf32>
  return %r : tensor<256xf32>
}

// -----

// erf via the accurate math.erf on the CPU/linalg path (the GPU pass uses the
// libdevice-safe A&S poly in MapEmit.h).
// CHECK-LABEL: func.func @uerf
// CHECK: math.erf
func.func @uerf(%a: tensor<256xf32>) -> tensor<256xf32> {
  %r = intent.map_unary %a {fn = "erf"} : (tensor<256xf32>) -> tensor<256xf32>
  return %r : tensor<256xf32>
}

// -----

// atan via the accurate math.atan on the CPU/linalg path (the GPU pass uses the
// libdevice-safe minimax poly in MapEmit.h). Complex angle = atan2(imag, real).
// CHECK-LABEL: func.func @uatan
// CHECK: math.atan
func.func @uatan(%a: tensor<256xf32>) -> tensor<256xf32> {
  %r = intent.map_unary %a {fn = "atan"} : (tensor<256xf32>) -> tensor<256xf32>
  return %r : tensor<256xf32>
}

// -----

// atan2(imag, real) — the complex angle primitive — via math.atan2 on linalg.
// CHECK-LABEL: func.func @batan2
// CHECK: math.atan2
func.func @batan2(%im: tensor<256xf32>, %re: tensor<256xf32>) -> tensor<256xf32> {
  %r = intent.map_binary %im, %re {fn = "atan2"} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
  return %r : tensor<256xf32>
}

// -----

// A widening dtype cast lowers to arith.extf; a same-dtype cast is an identity
// alias (folded away, no linalg.generic).
// CHECK-LABEL: func.func @ucast_up
// CHECK: arith.extf
func.func @ucast_up(%a: tensor<8x8xf16>) -> tensor<8x8xf32> {
  %r = intent.cast %a {to_dtype = "f32"} : (tensor<8x8xf16>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @ucast_down
// CHECK: arith.truncf
func.func @ucast_down(%a: tensor<8x8xf32>) -> tensor<8x8xf16> {
  %r = intent.cast %a {to_dtype = "f16"} : (tensor<8x8xf32>) -> tensor<8x8xf16>
  return %r : tensor<8x8xf16>
}

// -----

// f32->f32 cast is a value-level no-op: identity alias, no cast op emitted.
// CHECK-LABEL: func.func @ucast_id
// CHECK-NOT: arith.truncf
// CHECK-NOT: arith.extf
// CHECK-NOT: linalg.generic
func.func @ucast_id(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %r = intent.cast %a {to_dtype = "f32"} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}
