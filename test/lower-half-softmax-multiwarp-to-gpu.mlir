// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=128 %s | FileCheck %s

// The production multi-warp schedule keeps reduction scratch and shuffle
// values in f32 even though global storage is half precision.

// CHECK-LABEL: func.func @softmax_f16_multiwarp(
// CHECK-SAME:    {{.*}}memref<4x256xvector<4xf16>>{{.*}}memref<4x256xvector<4xf16>>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         memref.load {{.*}} : memref<4x256xvector<4xf16>>
// CHECK:         vector.extract {{.*}} : f16 from vector<4xf16>
// CHECK:         arith.extf {{.*}} : f16 to f32
// CHECK:         gpu.shuffle {{.*}} : f32
// CHECK:         math.exp {{.*}} : f32
// CHECK:         arith.truncf {{.*}} : f32 to f16
// CHECK:         vector.from_elements {{.*}} : vector<4xf16>
// CHECK:         memref.store {{.*}} : memref<4x256xvector<4xf16>>
// CHECK-NOT:     intent.
func.func @softmax_f16_multiwarp(%x: tensor<4x1024xf16>)
    -> tensor<4x1024xf16> {
  %max = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"}
    : (tensor<4x1024xf16>) -> tensor<4x1xf16>
  %shifted = intent.map_binary %x, %max {fn = "sub"}
    : (tensor<4x1024xf16>, tensor<4x1xf16>) -> tensor<4x1024xf16>
  %exp = intent.map_unary %shifted {fn = "exp"}
    : (tensor<4x1024xf16>) -> tensor<4x1024xf16>
  %sum = intent.reduce %exp
    {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x1024xf16>) -> tensor<4x1xf16>
  %out = intent.map_binary %exp, %sum {fn = "div"}
    : (tensor<4x1024xf16>, tensor<4x1xf16>) -> tensor<4x1024xf16>
  return %out : tensor<4x1024xf16>
}

// CHECK-LABEL: func.func @softmax_bf16_multiwarp(
// CHECK-SAME:    {{.*}}memref<4x256xvector<4xbf16>>{{.*}}memref<4x256xvector<4xbf16>>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         memref.load {{.*}} : memref<4x256xvector<4xbf16>>
// CHECK:         vector.extract {{.*}} : bf16 from vector<4xbf16>
// CHECK:         arith.extf {{.*}} : bf16 to f32
// CHECK:         gpu.shuffle {{.*}} : f32
// CHECK:         math.exp {{.*}} : f32
// CHECK:         arith.truncf {{.*}} : f32 to bf16
// CHECK:         vector.from_elements {{.*}} : vector<4xbf16>
// CHECK:         memref.store {{.*}} : memref<4x256xvector<4xbf16>>
// CHECK-NOT:     intent.
func.func @softmax_bf16_multiwarp(%x: tensor<4x1024xbf16>)
    -> tensor<4x1024xbf16> {
  %max = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"}
    : (tensor<4x1024xbf16>) -> tensor<4x1xbf16>
  %shifted = intent.map_binary %x, %max {fn = "sub"}
    : (tensor<4x1024xbf16>, tensor<4x1xbf16>) -> tensor<4x1024xbf16>
  %exp = intent.map_unary %shifted {fn = "exp"}
    : (tensor<4x1024xbf16>) -> tensor<4x1024xbf16>
  %sum = intent.reduce %exp
    {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x1024xbf16>) -> tensor<4x1xbf16>
  %out = intent.map_binary %exp, %sum {fn = "div"}
    : (tensor<4x1024xbf16>, tensor<4x1xbf16>) -> tensor<4x1024xbf16>
  return %out : tensor<4x1024xbf16>
}
