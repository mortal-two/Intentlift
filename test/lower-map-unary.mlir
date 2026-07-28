// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s
// RUN: intent-opt -convert-intent-to-linalg -expand-rsqrt-for-libm %s | FileCheck %s --check-prefix=HOST
// RUN: intent-opt -convert-intent-to-linalg %s | intent-opt -lower-gpu-math-to-nvvm | FileCheck %s --check-prefix=CUDA

// map_unary{exp} -> a single-input linalg.generic with math.exp in the body.
// CHECK-LABEL: func.func @ue
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: math.exp
func.func @ue(%a: tensor<256xf32>) -> tensor<256xf32> {
  %r = intent.map_unary %a {fn = "exp"} : (tensor<256xf32>) -> tensor<256xf32>
  return %r : tensor<256xf32>
}

// -----

// map_unary{neg} lowers to arith.negf (no transcendental).
// CHECK-LABEL: func.func @un
// CHECK: arith.negf
func.func @un(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %r = intent.map_unary %a {fn = "neg"} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// -----

// map_unary{identity} remains an explicitly authored J node while its scalar
// denotation yields the input unchanged.
// CHECK-LABEL: func.func @uid
// CHECK: linalg.generic
// CHECK: linalg.yield %{{.*}} : f32
func.func @uid(%a: tensor<8xf32>) -> tensor<8xf32> {
  %r = intent.map_unary %a {fn = "identity"} : (tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----

// Target-neutral lowering preserves math.rsqrt so production GPU consumers can
// still select the native target implementation.
// CHECK-LABEL: func.func @ursqrt
// CHECK: math.rsqrt
// CHECK-NOT: arith.divf

// The CPU-only compatibility pass expands rsqrt before convert-math-to-libm,
// avoiding the non-standard host `rsqrtf` symbol.
// HOST-LABEL: func.func @ursqrt
// HOST: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// HOST: %[[SQRT:.*]] = math.sqrt
// HOST: arith.divf %[[ONE]], %[[SQRT]]
// HOST-NOT: math.rsqrt

// The generic/fused CUDA fallback does not schedule the host rewrite and still
// reaches the established NVVM reciprocal-square-root intrinsic.
// CUDA-LABEL: func.func @ursqrt
// CUDA: llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"
// CUDA-NOT: math.sqrt
func.func @ursqrt(%a: tensor<8xf32>) -> tensor<8xf32> {
  %r = intent.map_unary %a {fn = "rsqrt"} : (tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}
