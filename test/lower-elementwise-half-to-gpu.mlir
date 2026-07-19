// RUN: intent-opt -convert-intent-elementwise-to-gpu=block-threads=32 %s | FileCheck %s

// Half and bfloat storage is promoted per semantic map op to f32, then rounded
// back to the op's declared result type.  The direct path therefore reuses the
// complete f32 math backend without changing typed intermediate semantics.

// CHECK-LABEL: func.func @f16_chain(
// CHECK-SAME: memref<4x32xvector<4xf16>>
// CHECK: gpu.launch
// CHECK: vector.extract
// CHECK: arith.extf {{.*}} : f16 to f32
// CHECK: arith.addf
// CHECK: arith.truncf {{.*}} : f32 to f16
// CHECK: arith.extf {{.*}} : f16 to f32
// CHECK: arith.mulf
// CHECK: arith.truncf {{.*}} : f32 to f16
// CHECK: vector.from_elements {{.*}} : vector<4xf16>
func.func @f16_chain(
    %a: tensor<4x128xf16>, %b: tensor<4x128xf16>, %c: tensor<4x128xf16>)
    -> tensor<4x128xf16> {
  %sum = intent.map_binary %a, %b {fn = "add"}
      : (tensor<4x128xf16>, tensor<4x128xf16>) -> tensor<4x128xf16>
  %out = intent.map_binary %sum, %c {fn = "mul"}
      : (tensor<4x128xf16>, tensor<4x128xf16>) -> tensor<4x128xf16>
  return %out : tensor<4x128xf16>
}

// -----

// CHECK-LABEL: func.func @bf16_tanh(
// CHECK-SAME: memref<4x32xbf16>
// CHECK: gpu.launch
// CHECK: arith.extf {{.*}} : bf16 to f32
// CHECK: math.exp
// CHECK: arith.truncf {{.*}} : f32 to bf16
func.func @bf16_tanh(%x: tensor<4x32xbf16>) -> tensor<4x32xbf16> {
  %out = intent.map_unary %x {fn = "tanh"}
      : (tensor<4x32xbf16>) -> tensor<4x32xbf16>
  return %out : tensor<4x32xbf16>
}

// -----

// A mixed f32 scalar remains f32 while the f16 row input is promoted and the
// f32 product is rounded once to the declared f16 result.
// CHECK-LABEL: func.func @f16_scale_f32(
// CHECK-SAME: memref<4x32xf16>, %{{.*}}: memref<f32>
// CHECK: arith.extf {{.*}} : f16 to f32
// CHECK: arith.mulf
// CHECK: arith.truncf {{.*}} : f32 to f16
func.func @f16_scale_f32(
    %x: tensor<4x32xf16>, %scale: tensor<f32>) -> tensor<4x32xf16> {
  %out = intent.map_binary %x, %scale {fn = "mul"}
      : (tensor<4x32xf16>, tensor<f32>) -> tensor<4x32xf16>
  return %out : tensor<4x32xf16>
}

// CHECK-NOT: intent.
