// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=32 %s | FileCheck %s

// Canonical half softmax is one read-once row kernel.  Storage remains f16 or
// bf16, but loads widen immediately and both collectives, exp, and division are
// f32.  The only narrowing is the final output write.

// CHECK-LABEL: func.func @softmax_f16(
// CHECK-SAME:    {{.*}}memref<4x32xf16>{{.*}}memref<4x32xf16>)
// CHECK-SAME:    intent.backend.reduction_topology = "subgroup_shuffle"
// CHECK:         gpu.launch
// CHECK:         %[[LOAD:.*]] = memref.load {{.*}} : memref<4x32xf16>
// CHECK-NEXT:    %[[WIDE:.*]] = arith.extf %[[LOAD]] : f16 to f32
// CHECK:         arith.maxnumf
// CHECK-COUNT-5: gpu.shuffle xor
// CHECK:         math.exp {{.*}} : f32
// CHECK-COUNT-5: gpu.shuffle xor
// CHECK:         arith.addf
// CHECK:         %[[NARROW:.*]] = arith.truncf {{.*}} : f32 to f16
// CHECK:         memref.store %[[NARROW]], {{.*}} : memref<4x32xf16>
// CHECK-NOT:     intent.
func.func @softmax_f16(%x: tensor<4x32xf16>) -> tensor<4x32xf16> {
  %max = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"}
    : (tensor<4x32xf16>) -> tensor<4x1xf16>
  %shifted = intent.map_binary %x, %max {fn = "sub"}
    : (tensor<4x32xf16>, tensor<4x1xf16>) -> tensor<4x32xf16>
  %exp = intent.map_unary %shifted {fn = "exp"}
    : (tensor<4x32xf16>) -> tensor<4x32xf16>
  %sum = intent.reduce %exp
    {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x32xf16>) -> tensor<4x1xf16>
  %out = intent.map_binary %exp, %sum {fn = "div"}
    : (tensor<4x32xf16>, tensor<4x1xf16>) -> tensor<4x32xf16>
  return %out : tensor<4x32xf16>
}

// CHECK-LABEL: func.func @softmax_bf16(
// CHECK-SAME:    {{.*}}memref<4x32xbf16>{{.*}}memref<4x32xbf16>)
// CHECK-SAME:    intent.backend.reduction_topology = "subgroup_shuffle"
// CHECK:         gpu.launch
// CHECK:         %[[LOAD:.*]] = memref.load {{.*}} : memref<4x32xbf16>
// CHECK-NEXT:    %[[WIDE:.*]] = arith.extf %[[LOAD]] : bf16 to f32
// CHECK:         arith.maxnumf
// CHECK-COUNT-5: gpu.shuffle xor
// CHECK:         math.exp {{.*}} : f32
// CHECK-COUNT-5: gpu.shuffle xor
// CHECK:         arith.addf
// CHECK:         %[[NARROW:.*]] = arith.truncf {{.*}} : f32 to bf16
// CHECK:         memref.store %[[NARROW]], {{.*}} : memref<4x32xbf16>
// CHECK-NOT:     intent.
func.func @softmax_bf16(%x: tensor<4x32xbf16>) -> tensor<4x32xbf16> {
  %max = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"}
    : (tensor<4x32xbf16>) -> tensor<4x1xbf16>
  %shifted = intent.map_binary %x, %max {fn = "sub"}
    : (tensor<4x32xbf16>, tensor<4x1xbf16>) -> tensor<4x32xbf16>
  %exp = intent.map_unary %shifted {fn = "exp"}
    : (tensor<4x32xbf16>) -> tensor<4x32xbf16>
  %sum = intent.reduce %exp
    {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x32xbf16>) -> tensor<4x1xbf16>
  %out = intent.map_binary %exp, %sum {fn = "div"}
    : (tensor<4x32xbf16>, tensor<4x1xbf16>) -> tensor<4x32xbf16>
  return %out : tensor<4x32xbf16>
}

// A merely similar half reduction graph is not admitted.  Mixed-precision
// fusion is restricted to canonical exp-based stable softmax.
// CHECK-LABEL: func.func @half_exp2_is_not_softmax(
// CHECK:         intent.reduce
// CHECK:         intent.map_unary {{.*}}fn = "exp2"
func.func @half_exp2_is_not_softmax(%x: tensor<4x32xf16>)
    -> tensor<4x32xf16> {
  %max = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"}
    : (tensor<4x32xf16>) -> tensor<4x1xf16>
  %shifted = intent.map_binary %x, %max {fn = "sub"}
    : (tensor<4x32xf16>, tensor<4x1xf16>) -> tensor<4x32xf16>
  %exp = intent.map_unary %shifted {fn = "exp2"}
    : (tensor<4x32xf16>) -> tensor<4x32xf16>
  %sum = intent.reduce %exp
    {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x32xf16>) -> tensor<4x1xf16>
  %out = intent.map_binary %exp, %sum {fn = "div"}
    : (tensor<4x32xf16>, tensor<4x1xf16>) -> tensor<4x32xf16>
  return %out : tensor<4x32xf16>
}
