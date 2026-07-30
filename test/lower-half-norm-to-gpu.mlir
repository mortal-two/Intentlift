// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=32 %s | FileCheck %s

// Mixed-precision norms keep their ABI casts explicit in IntentIR: half/bf16
// inputs are promoted before the f32 reduction DAG, and only the row result is
// demoted.  This lets the general row scheduler fuse the graph without
// inferring a hidden compute dtype.

// CHECK-LABEL: func.func @rms_norm_f16(
// CHECK-SAME:    {{.*}}memref<4x32xf16>{{.*}}memref<32xf16>{{.*}}memref<f32>{{.*}}memref<4x32xf16>)
// CHECK-SAME:    intent.backend.reduction_topology = "subgroup_shuffle"
// CHECK:         memref.load {{.*}} : memref<4x32xf16>
// CHECK:         arith.extf {{.*}} : f16 to f32
// CHECK-COUNT-5: gpu.shuffle xor
// CHECK:         math.rsqrt {{.*}} : f32
// CHECK-NOT:     math.sqrt
// CHECK:         memref.load {{.*}} : memref<32xf16>
// CHECK:         arith.extf {{.*}} : f16 to f32
// CHECK:         arith.truncf {{.*}} : f32 to f16
// CHECK:         memref.store {{.*}} : memref<4x32xf16>
// CHECK-NOT:     intent.
func.func @rms_norm_f16(%X: tensor<4x32xf16>, %W: tensor<32xf16>,
                        %eps: tensor<f32>) -> tensor<4x32xf16> {
  %Xf = intent.cast %X {to_dtype = "f32"}
    : (tensor<4x32xf16>) -> tensor<4x32xf32>
  %Wf = intent.cast %W {to_dtype = "f32"}
    : (tensor<32xf16>) -> tensor<32xf32>
  %x2 = intent.map_binary %Xf, %Xf {fn = "mul"}
    : (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
  %mean = intent.reduce %x2
    {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x32xf32>) -> tensor<4x1xf32>
  %me = intent.map_binary %mean, %eps {fn = "add"}
    : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rms = intent.map_unary %me {fn = "sqrt"}
    : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %inv = intent.map_unary %rms {fn = "recip"}
    : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %xs = intent.map_binary %Xf, %inv {fn = "mul"}
    : (tensor<4x32xf32>, tensor<4x1xf32>) -> tensor<4x32xf32>
  %Yf = intent.map_binary %xs, %Wf {fn = "mul"}
    : (tensor<4x32xf32>, tensor<32xf32>) -> tensor<4x32xf32>
  %Y = intent.cast %Yf {to_dtype = "f16"}
    : (tensor<4x32xf32>) -> tensor<4x32xf16>
  return %Y : tensor<4x32xf16>
}

// CHECK-LABEL: func.func @layer_norm_bf16(
// CHECK-SAME:    {{.*}}memref<4x32xbf16>{{.*}}memref<32xbf16>{{.*}}memref<32xbf16>{{.*}}memref<f32>
// CHECK-SAME:    {{.*}}memref<4x32xbf16>{{.*}}memref<4xf32>{{.*}}memref<4xf32>)
// CHECK-SAME:     intent.backend.reduction_topology = "subgroup_shuffle"
// CHECK:         memref.load {{.*}} : memref<4x32xbf16>
// CHECK:         arith.extf {{.*}} : bf16 to f32
// CHECK-COUNT-10: gpu.shuffle xor
// CHECK:         math.rsqrt {{.*}} : f32
// CHECK:         memref.load {{.*}} : memref<32xbf16>
// CHECK:         arith.extf {{.*}} : bf16 to f32
// CHECK:         arith.truncf {{.*}} : f32 to bf16
// CHECK:         memref.store {{.*}} : memref<4x32xbf16>
// CHECK:         memref.store {{.*}} : memref<4xf32>
// CHECK:         memref.store {{.*}} : memref<4xf32>
// CHECK-NOT:     intent.
func.func @layer_norm_bf16(%X: tensor<4x32xbf16>, %W: tensor<32xbf16>,
                           %B: tensor<32xbf16>, %eps: tensor<f32>)
    -> (tensor<4x32xbf16>, tensor<4x1xf32>, tensor<4x1xf32>) {
  %Xf = intent.cast %X {to_dtype = "f32"}
    : (tensor<4x32xbf16>) -> tensor<4x32xf32>
  %Wf = intent.cast %W {to_dtype = "f32"}
    : (tensor<32xbf16>) -> tensor<32xf32>
  %Bf = intent.cast %B {to_dtype = "f32"}
    : (tensor<32xbf16>) -> tensor<32xf32>
  %mean = intent.reduce %Xf
    {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x32xf32>) -> tensor<4x1xf32>
  %xc = intent.map_binary %Xf, %mean {fn = "sub"}
    : (tensor<4x32xf32>, tensor<4x1xf32>) -> tensor<4x32xf32>
  %xc2 = intent.map_binary %xc, %xc {fn = "mul"}
    : (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
  %var = intent.reduce %xc2
    {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x32xf32>) -> tensor<4x1xf32>
  %vare = intent.map_binary %var, %eps {fn = "add"}
    : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rstd = intent.map_unary %vare {fn = "rsqrt"}
    : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %xn = intent.map_binary %xc, %rstd {fn = "mul"}
    : (tensor<4x32xf32>, tensor<4x1xf32>) -> tensor<4x32xf32>
  %xnw = intent.map_binary %xn, %Wf {fn = "mul"}
    : (tensor<4x32xf32>, tensor<32xf32>) -> tensor<4x32xf32>
  %Yf = intent.map_binary %xnw, %Bf {fn = "add"}
    : (tensor<4x32xf32>, tensor<32xf32>) -> tensor<4x32xf32>
  %Y = intent.cast %Yf {to_dtype = "bf16"}
    : (tensor<4x32xf32>) -> tensor<4x32xbf16>
  return %Y, %mean, %rstd
    : tensor<4x32xbf16>, tensor<4x1xf32>, tensor<4x1xf32>
}

// Half storage alone is not permission to change a graph's compute dtype.  A
// reduction without explicit half->f32/f32->half boundaries remains untouched.
// CHECK-LABEL: func.func @implicit_half_norm_is_not_promoted(
// CHECK:         intent.reduce
func.func @implicit_half_norm_is_not_promoted(%X: tensor<4x32xf16>)
    -> tensor<4x1xf16> {
  %x2 = intent.map_binary %X, %X {fn = "mul"}
    : (tensor<4x32xf16>, tensor<4x32xf16>) -> tensor<4x32xf16>
  %mean = intent.reduce %x2
    {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x32xf16>) -> tensor<4x1xf16>
  return %mean : tensor<4x1xf16>
}
