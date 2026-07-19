// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=32 %s | FileCheck %s
// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=32 vector-width=4' %s | FileCheck %s --check-prefix=FORCED

// Explicit mixed-precision normalization graphs share the same safe ragged-row
// schedule as half softmax.  Reduction terms are masked only at the accumulator
// boundary, so centered variance cannot be polluted by a dummy loaded value.

// FORCED-LABEL: func.func @rms_norm_f16_tail(
// FORCED:         intent.reduce

// CHECK-LABEL: func.func @rms_norm_f16_tail(
// CHECK-SAME:    {{.*}}memref<4x37xf16>{{.*}}memref<37xf16>{{.*}}memref<f32>{{.*}}memref<4x37xf16>)
// CHECK:         %[[ACTIVE:.*]] = arith.cmpi ult
// CHECK:         %[[SAFE:.*]] = arith.select %[[ACTIVE]]
// CHECK:         memref.load {{.*}}[%{{.*}}, %[[SAFE]]] : memref<4x37xf16>
// CHECK:         arith.extf {{.*}} : f16 to f32
// CHECK:         arith.select %[[ACTIVE]], {{.*}} : f32
// CHECK:         gpu.all_reduce add
// CHECK:         memref.load {{.*}}[%[[SAFE]]] : memref<37xf16>
// CHECK:         scf.if %[[ACTIVE]] {
// CHECK:           memref.store {{.*}} : memref<4x37xf16>
// CHECK:         }
// CHECK-NOT:     intent.
func.func @rms_norm_f16_tail(%X: tensor<4x37xf16>, %W: tensor<37xf16>,
                             %eps: tensor<f32>) -> tensor<4x37xf16> {
  %Xf = intent.cast %X {to_dtype = "f32"}
    : (tensor<4x37xf16>) -> tensor<4x37xf32>
  %Wf = intent.cast %W {to_dtype = "f32"}
    : (tensor<37xf16>) -> tensor<37xf32>
  %x2 = intent.map_binary %Xf, %Xf {fn = "mul"}
    : (tensor<4x37xf32>, tensor<4x37xf32>) -> tensor<4x37xf32>
  %mean = intent.reduce %x2
    {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x37xf32>) -> tensor<4x1xf32>
  %me = intent.map_binary %mean, %eps {fn = "add"}
    : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rstd = intent.map_unary %me {fn = "rsqrt"}
    : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %xs = intent.map_binary %Xf, %rstd {fn = "mul"}
    : (tensor<4x37xf32>, tensor<4x1xf32>) -> tensor<4x37xf32>
  %Yf = intent.map_binary %xs, %Wf {fn = "mul"}
    : (tensor<4x37xf32>, tensor<37xf32>) -> tensor<4x37xf32>
  %Y = intent.cast %Yf {to_dtype = "f16"}
    : (tensor<4x37xf32>) -> tensor<4x37xf16>
  return %Y : tensor<4x37xf16>
}

// CHECK-LABEL: func.func @layer_norm_bf16_tail(
// CHECK-SAME:    {{.*}}memref<4x37xbf16>{{.*}}memref<37xbf16>{{.*}}memref<37xbf16>{{.*}}memref<f32>
// CHECK-SAME:    {{.*}}memref<4x37xbf16>{{.*}}memref<4xf32>{{.*}}memref<4xf32>)
// CHECK:         %[[ACTIVE:.*]] = arith.cmpi ult
// CHECK:         %[[SAFE:.*]] = arith.select %[[ACTIVE]]
// CHECK:         memref.load {{.*}}[%{{.*}}, %[[SAFE]]] : memref<4x37xbf16>
// CHECK-COUNT-2: gpu.all_reduce add
// CHECK:         memref.load {{.*}}[%[[SAFE]]] : memref<37xbf16>
// CHECK:         scf.if %[[ACTIVE]] {
// CHECK:           memref.store {{.*}} : memref<4x37xbf16>
// CHECK:         }
// CHECK-NOT:     intent.
func.func @layer_norm_bf16_tail(%X: tensor<4x37xbf16>,
                                %W: tensor<37xbf16>,
                                %B: tensor<37xbf16>, %eps: tensor<f32>)
    -> (tensor<4x37xbf16>, tensor<4x1xf32>, tensor<4x1xf32>) {
  %Xf = intent.cast %X {to_dtype = "f32"}
    : (tensor<4x37xbf16>) -> tensor<4x37xf32>
  %Wf = intent.cast %W {to_dtype = "f32"}
    : (tensor<37xbf16>) -> tensor<37xf32>
  %Bf = intent.cast %B {to_dtype = "f32"}
    : (tensor<37xbf16>) -> tensor<37xf32>
  %mean = intent.reduce %Xf
    {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x37xf32>) -> tensor<4x1xf32>
  %xc = intent.map_binary %Xf, %mean {fn = "sub"}
    : (tensor<4x37xf32>, tensor<4x1xf32>) -> tensor<4x37xf32>
  %xc2 = intent.map_binary %xc, %xc {fn = "mul"}
    : (tensor<4x37xf32>, tensor<4x37xf32>) -> tensor<4x37xf32>
  %var = intent.reduce %xc2
    {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x37xf32>) -> tensor<4x1xf32>
  %vare = intent.map_binary %var, %eps {fn = "add"}
    : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rstd = intent.map_unary %vare {fn = "rsqrt"}
    : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %xn = intent.map_binary %xc, %rstd {fn = "mul"}
    : (tensor<4x37xf32>, tensor<4x1xf32>) -> tensor<4x37xf32>
  %xnw = intent.map_binary %xn, %Wf {fn = "mul"}
    : (tensor<4x37xf32>, tensor<37xf32>) -> tensor<4x37xf32>
  %Yf = intent.map_binary %xnw, %Bf {fn = "add"}
    : (tensor<4x37xf32>, tensor<37xf32>) -> tensor<4x37xf32>
  %Y = intent.cast %Yf {to_dtype = "bf16"}
    : (tensor<4x37xf32>) -> tensor<4x37xbf16>
  return %Y, %mean, %rstd
    : tensor<4x37xbf16>, tensor<4x1xf32>, tensor<4x1xf32>
}
