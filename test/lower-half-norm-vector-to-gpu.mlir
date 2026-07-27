// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=128 %s | FileCheck %s --check-prefix=VEC
// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=128 vector-width=1' %s | FileCheck %s --check-prefix=SCALAR

// The measured mixed-precision memory fast path changes only the storage
// packetization.  Both vector and scalar ablation lower the same explicit f32
// compute graph and use the same f32 reduction hierarchy.

// VEC-LABEL: func.func @rms_norm_bf16_vector(
// VEC-SAME:    %{{.*}}: memref<4x256xvector<4xbf16>>, %{{.*}}: memref<256xvector<4xbf16>>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x256xvector<4xbf16>>)
// VEC-SAME:    vector_width = 4 : i64
// VEC:         memref.load {{.*}} : memref<4x256xvector<4xbf16>>
// VEC:         vector.extract {{.*}} : bf16 from vector<4xbf16>
// VEC:         arith.extf {{.*}} : bf16 to f32
// VEC:         gpu.shuffle {{.*}} : f32
// VEC:         memref.load {{.*}} : memref<256xvector<4xbf16>>
// VEC:         vector.extract {{.*}} : bf16 from vector<4xbf16>
// VEC:         arith.extf {{.*}} : bf16 to f32
// VEC:         arith.truncf {{.*}} : f32 to bf16
// VEC:         vector.from_elements {{.*}} : vector<4xbf16>
// VEC:         memref.store {{.*}} : memref<4x256xvector<4xbf16>>
// VEC-NOT:     intent.

// SCALAR-LABEL: func.func @rms_norm_bf16_vector(
// SCALAR-SAME:    %{{.*}}: memref<4x1024xbf16>, %{{.*}}: memref<1024xbf16>, %{{.*}}: memref<f32>, %{{.*}}: memref<4x1024xbf16>)
// SCALAR-SAME:    vector_width = 1 : i64
// SCALAR-NOT:     vector<4xbf16>
// SCALAR:         memref.load {{.*}} : memref<4x1024xbf16>
// SCALAR:         arith.extf {{.*}} : bf16 to f32
// SCALAR:         gpu.shuffle {{.*}} : f32
// SCALAR:         arith.truncf {{.*}} : f32 to bf16
// SCALAR:         memref.store {{.*}} : memref<4x1024xbf16>
// SCALAR-NOT:     intent.

// Iota-driven masks remain scalar semantic expressions inside each packet;
// they do not disable in-bounds BF16 packet loads/stores.
// VEC-LABEL: func.func @masked_rms_norm_bf16_vector(
// VEC-SAME:    %{{.*}}: memref<4x256xvector<4xbf16>>
// VEC-SAME:    vector_width = 4 : i64
// VEC:         memref.load {{.*}} : memref<4x256xvector<4xbf16>>
// VEC:         vector.extract {{.*}} : bf16 from vector<4xbf16>
// VEC:         arith.cmpi slt
// VEC:         arith.select
// VEC:         vector.from_elements {{.*}} : vector<4xbf16>
// VEC:         memref.store {{.*}} : memref<4x256xvector<4xbf16>>
// VEC-NOT:     intent.

// SCALAR-LABEL: func.func @masked_rms_norm_bf16_vector(
// SCALAR-SAME:    %{{.*}}: memref<4x1024xbf16>
// SCALAR-SAME:    vector_width = 1 : i64
// SCALAR-NOT:     vector<4xbf16>
// SCALAR:         arith.cmpi slt
// SCALAR:         arith.select
// SCALAR-NOT:     intent.
func.func @rms_norm_bf16_vector(%X: tensor<4x1024xbf16>,
                                %W: tensor<1024xbf16>, %eps: tensor<f32>)
    -> tensor<4x1024xbf16> {
  %Xf = intent.cast %X {to_dtype = "f32"}
    : (tensor<4x1024xbf16>) -> tensor<4x1024xf32>
  %Wf = intent.cast %W {to_dtype = "f32"}
    : (tensor<1024xbf16>) -> tensor<1024xf32>
  %x2 = intent.map_binary %Xf, %Xf {fn = "mul"}
    : (tensor<4x1024xf32>, tensor<4x1024xf32>) -> tensor<4x1024xf32>
  %mean = intent.reduce %x2
    {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x1024xf32>) -> tensor<4x1xf32>
  %me = intent.map_binary %mean, %eps {fn = "add"}
    : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rms = intent.map_unary %me {fn = "sqrt"}
    : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %inv = intent.map_unary %rms {fn = "recip"}
    : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %xs = intent.map_binary %Xf, %inv {fn = "mul"}
    : (tensor<4x1024xf32>, tensor<4x1xf32>) -> tensor<4x1024xf32>
  %Yf = intent.map_binary %xs, %Wf {fn = "mul"}
    : (tensor<4x1024xf32>, tensor<1024xf32>) -> tensor<4x1024xf32>
  %Y = intent.cast %Yf {to_dtype = "bf16"}
    : (tensor<4x1024xf32>) -> tensor<4x1024xbf16>
  return %Y : tensor<4x1024xbf16>
}

func.func @masked_rms_norm_bf16_vector(
    %X: tensor<4x1024xbf16>, %W: tensor<1024xbf16>, %eps: tensor<f32>,
    %N: tensor<i32>) -> tensor<4x1024xbf16>
    attributes {intent.scalar_equalities = array<i64: 3, 1024>} {
  %Xf = intent.cast %X {to_dtype = "f32"}
    : (tensor<4x1024xbf16>) -> tensor<4x1024xf32>
  %Wf = intent.cast %W {to_dtype = "f32"}
    : (tensor<1024xbf16>) -> tensor<1024xf32>
  %columns = intent.iota : tensor<1024xi32>
  %valid = intent.map_binary %columns, %N {fn = "lt"}
    : (tensor<1024xi32>, tensor<i32>) -> tensor<1024xi1>
  %zero = arith.constant dense<0.0> : tensor<f32>
  %masked = intent.select %valid, %Xf, %zero
    : (tensor<1024xi1>, tensor<4x1024xf32>, tensor<f32>)
      -> tensor<4x1024xf32>
  %x2 = intent.map_binary %masked, %masked {fn = "mul"}
    : (tensor<4x1024xf32>, tensor<4x1024xf32>) -> tensor<4x1024xf32>
  %mean = intent.reduce %x2
    {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<4x1024xf32>) -> tensor<4x1xf32>
  %me = intent.map_binary %mean, %eps {fn = "add"}
    : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  %rms = intent.map_unary %me {fn = "sqrt"}
    : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %inv = intent.map_unary %rms {fn = "recip"}
    : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %xs = intent.map_binary %masked, %inv {fn = "mul"}
    : (tensor<4x1024xf32>, tensor<4x1xf32>) -> tensor<4x1024xf32>
  %Yf = intent.map_binary %xs, %Wf {fn = "mul"}
    : (tensor<4x1024xf32>, tensor<1024xf32>) -> tensor<4x1024xf32>
  %Y = intent.cast %Yf {to_dtype = "bf16"}
    : (tensor<4x1024xf32>) -> tensor<4x1024xbf16>
  return %Y : tensor<4x1024xbf16>
}
