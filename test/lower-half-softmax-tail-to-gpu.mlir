// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=32 %s | FileCheck %s

// Ragged rows use a safe address for the final register tile, replace invalid
// reduction terms with the correct f32 identity, and guard the corresponding
// stores.  Both collectives remain unconditional across the whole block.

// CHECK-LABEL: func.func @softmax_f16_tail(
// CHECK-SAME:    {{.*}}memref<4x37xf16>{{.*}}memref<4x37xf16>)
// CHECK:         %[[ACTIVE:.*]] = arith.cmpi ult
// CHECK:         %[[SAFE:.*]] = arith.select %[[ACTIVE]]
// CHECK:         memref.load {{.*}}[%{{.*}}, %[[SAFE]]] : memref<4x37xf16>
// CHECK:         arith.extf {{.*}} : f16 to f32
// CHECK:         arith.select %[[ACTIVE]], {{.*}} : f32
// CHECK:         gpu.all_reduce maxnumf
// CHECK:         gpu.all_reduce add
// CHECK:         scf.if %[[ACTIVE]] {
// CHECK:           memref.store {{.*}} : memref<4x37xf16>
// CHECK:         }
// CHECK-NOT:     intent.
func.func @softmax_f16_tail(%x: tensor<4x37xf16>) -> tensor<4x37xf16> {
  %max = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"}
    : (tensor<4x37xf16>) -> tensor<4x1xf16>
  %shifted = intent.map_binary %x, %max {fn = "sub"}
    : (tensor<4x37xf16>, tensor<4x1xf16>) -> tensor<4x37xf16>
  %exp = intent.map_unary %shifted {fn = "exp"}
    : (tensor<4x37xf16>) -> tensor<4x37xf16>
  %sum = intent.reduce %exp
    {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x37xf16>) -> tensor<4x1xf16>
  %out = intent.map_binary %exp, %sum {fn = "div"}
    : (tensor<4x37xf16>, tensor<4x1xf16>) -> tensor<4x37xf16>
  return %out : tensor<4x37xf16>
}

// CHECK-LABEL: func.func @softmax_bf16_tail(
// CHECK-SAME:    {{.*}}memref<4x37xbf16>{{.*}}memref<4x37xbf16>)
// CHECK:         %[[ACTIVE:.*]] = arith.cmpi ult
// CHECK:         %[[SAFE:.*]] = arith.select %[[ACTIVE]]
// CHECK:         memref.load {{.*}}[%{{.*}}, %[[SAFE]]] : memref<4x37xbf16>
// CHECK:         arith.extf {{.*}} : bf16 to f32
// CHECK:         arith.select %[[ACTIVE]], {{.*}} : f32
// CHECK:         gpu.all_reduce maxnumf
// CHECK:         gpu.all_reduce add
// CHECK:         scf.if %[[ACTIVE]] {
// CHECK:           memref.store {{.*}} : memref<4x37xbf16>
// CHECK:         }
// CHECK-NOT:     intent.
func.func @softmax_bf16_tail(%x: tensor<4x37xbf16>) -> tensor<4x37xbf16> {
  %max = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"}
    : (tensor<4x37xbf16>) -> tensor<4x1xbf16>
  %shifted = intent.map_binary %x, %max {fn = "sub"}
    : (tensor<4x37xbf16>, tensor<4x1xbf16>) -> tensor<4x37xbf16>
  %exp = intent.map_unary %shifted {fn = "exp"}
    : (tensor<4x37xbf16>) -> tensor<4x37xbf16>
  %sum = intent.reduce %exp
    {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x37xbf16>) -> tensor<4x1xbf16>
  %out = intent.map_binary %exp, %sum {fn = "div"}
    : (tensor<4x37xbf16>, tensor<4x1xbf16>) -> tensor<4x37xbf16>
  return %out : tensor<4x37xbf16>
}
