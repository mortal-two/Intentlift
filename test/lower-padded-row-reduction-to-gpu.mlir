// RUN: intent-opt -convert-intent-padded-row-reduction-to-gpu='block-threads=64 vectorize-full-features=false promote-rank0-scalars=true' %s | FileCheck %s --check-prefix=HOST
// RUN: intent-opt -convert-intent-padded-row-reduction-to-gpu='block-threads=64 vectorize-full-features=false promote-rank0-scalars=false' %s | FileCheck %s --check-prefix=DEVICE
// RUN: intent-opt -convert-intent-padded-row-reduction-to-gpu='block-threads=64 vectorize-full-features=false promote-rank0-scalars=true' %s | FileCheck %s --check-prefix=STATIC
// RUN: intent-opt -convert-intent-padded-row-reduction-to-gpu='block-threads=64 vectorize-full-features=false promote-rank0-scalars=true' %s | FileCheck %s --check-prefix=NATURAL
// RUN: intent-opt -convert-intent-padded-row-reduction-to-gpu='block-threads=256 reduction-topology=subgroup-feature' %s | FileCheck %s --check-prefix=NATURAL-SUBGROUP
// RUN: intent-opt -convert-intent-padded-row-reduction-to-gpu='block-threads=256 reduction-topology=subgroup-feature rows-per-subgroup=4' %s | FileCheck %s --check-prefix=NATURAL-MULTIROW
// RUN: intent-opt -convert-intent-padded-row-reduction-to-gpu='block-threads=128 reduction-topology=subgroup-feature lanes-per-row=4' %s | FileCheck %s --check-prefix=NATURAL-SUBWARP
// RUN: intent-opt -convert-intent-padded-row-reduction-to-gpu='block-threads=128 reduction-topology=subgroup-feature lanes-per-row=16 vectorize-full-features=true' %s | FileCheck %s --check-prefix=NATURAL-PACKET

// The matcher is source/name neutral and consumes the exact structural form
// emitted from the closed FlashAttention preprocess J.

// HOST-LABEL: func.func @anonymous(
// HOST-SAME: intent.backend.host_value_capture_count = 2 : i64
// HOST-SAME: "cuda.b.rank0_host_value_capture.v1"
// HOST: memref.load
// HOST: memref.load
// HOST: gpu.launch
// HOST: memref.store
// HOST-NOT: intent.
// HOST-LABEL: func.func @static_bounds(

// DEVICE-LABEL: func.func @anonymous(
// DEVICE-SAME: intent.backend.host_value_capture_count = 0 : i64
// DEVICE: gpu.launch
// DEVICE: memref.load
// DEVICE: memref.store
// DEVICE-NOT: intent.
// DEVICE-LABEL: func.func @static_bounds(
func.func @anonymous(
    %a: tensor<1x2x33x16xf16>,
    %b: tensor<1x2x33x16xf16>,
    %valid_rows: tensor<i32>,
    %valid_features: tensor<i32>) -> tensor<1x2x64xf32> {
  %row_iota = intent.iota : tensor<33xi32>
  %row_valid = intent.map_binary %row_iota, %valid_rows {fn = "lt"}
    : (tensor<33xi32>, tensor<i32>) -> tensor<33xi1>
  %feature_iota = intent.iota : tensor<16xi32>
  %feature_valid = intent.map_binary %feature_iota, %valid_features {fn = "lt"}
    : (tensor<16xi32>, tensor<i32>) -> tensor<16xi1>
  %row_expanded = tensor.expand_shape %row_valid [[0, 1]] output_shape [33, 1]
    : tensor<33xi1> into tensor<33x1xi1>
  %mask = intent.map_binary %row_expanded, %feature_valid {fn = "logical_and"}
    : (tensor<33x1xi1>, tensor<16xi1>) -> tensor<33x16xi1>
  %zero = arith.constant dense<0.0> : tensor<f16>
  %mask_a = tensor.expand_shape %mask [[0, 1, 2], [3]]
    output_shape [1, 1, 33, 16]
    : tensor<33x16xi1> into tensor<1x1x33x16xi1>
  %masked_a = intent.select %mask_a, %a, %zero
    : (tensor<1x1x33x16xi1>, tensor<1x2x33x16xf16>, tensor<f16>)
      -> tensor<1x2x33x16xf16>
  %mask_b = tensor.expand_shape %mask [[0, 1, 2], [3]]
    output_shape [1, 1, 33, 16]
    : tensor<33x16xi1> into tensor<1x1x33x16xi1>
  %masked_b = intent.select %mask_b, %b, %zero
    : (tensor<1x1x33x16xi1>, tensor<1x2x33x16xf16>, tensor<f16>)
      -> tensor<1x2x33x16xf16>
  %af = intent.cast %masked_a {to_dtype = "f32"}
    : (tensor<1x2x33x16xf16>) -> tensor<1x2x33x16xf32>
  %bf = intent.cast %masked_b {to_dtype = "f32"}
    : (tensor<1x2x33x16xf16>) -> tensor<1x2x33x16xf32>
  %product = intent.map_binary %af, %bf {fn = "mul"}
    : (tensor<1x2x33x16xf32>, tensor<1x2x33x16xf32>)
      -> tensor<1x2x33x16xf32>
  %sum = intent.reduce %product
    {fn = "sum", axes = [3], assoc = "float_nondeterministic"}
    : (tensor<1x2x33x16xf32>) -> tensor<1x2x33xf32>
  %out = intent.pad %sum
    {lo = [0, 0, 0], hi = [0, 0, 31], value = 0.0 : f64}
    : (tensor<1x2x33xf32>) -> tensor<1x2x64xf32>
  return %out : tensor<1x2x64xf32>
}

// The Fresh14 typed J has already resolved both runtime bounds, so its graph has
// direct casts instead of mask/select nodes.  The same lowering must consume it,
// preserve the three-pointer ABI, and materialize the zero-padded tail.
// STATIC-LABEL: func.func @static_bounds(
// STATIC-SAME: %{{[^:]+}}: memref<1x2x33x16xf16>
// STATIC-SAME: %{{[^:]+}}: memref<1x2x33x16xf16>
// STATIC-SAME: %{{[^:]+}}: memref<1x2x64xf32>
// STATIC-SAME: intent.backend.host_value_capture_count = 0 : i64
// STATIC: gpu.launch
// STATIC: arith.constant 33 : index
// STATIC: arith.constant 16 : index
// STATIC: memref.store
// STATIC-NOT: intent.
// STATIC-LABEL: func.func @natural_batch(
func.func @static_bounds(
    %a: tensor<1x2x33x16xf16>,
    %b: tensor<1x2x33x16xf16>) -> tensor<1x2x64xf32> {
  %af = intent.cast %a {to_dtype = "f32"}
    : (tensor<1x2x33x16xf16>) -> tensor<1x2x33x16xf32>
  %bf = intent.cast %b {to_dtype = "f32"}
    : (tensor<1x2x33x16xf16>) -> tensor<1x2x33x16xf32>
  %product = intent.map_binary %af, %bf {fn = "mul"}
    : (tensor<1x2x33x16xf32>, tensor<1x2x33x16xf32>)
      -> tensor<1x2x33x16xf32>
  %sum = intent.reduce %product
    {fn = "sum", axes = [3], assoc = "float_nondeterministic"}
    : (tensor<1x2x33x16xf32>) -> tensor<1x2x33xf32>
  %out = intent.pad %sum
    {lo = [0, 0, 0], hi = [0, 0, 31], value = 0.0 : f64}
    : (tensor<1x2x33xf32>) -> tensor<1x2x64xf32>
  return %out : tensor<1x2x64xf32>
}

// NATURAL-LABEL: func.func @natural_batch(
// NATURAL-SAME: %{{[^:]+}}: memref<4x16x8192x128xf16>
// NATURAL-SAME: %{{[^:]+}}: memref<4x16x8192x128xf16>
// NATURAL-SAME: %{{[^:]+}}: memref<4x16x8192xf32>
// NATURAL: gpu.launch
// NATURAL: arith.constant 4 : index
// NATURAL: memref.store
// NATURAL-NOT: intent.
// NATURAL-SUBGROUP-LABEL: func.func @natural_batch(
// NATURAL-SUBGROUP-SAME: intent.backend.reduction_topology = "subgroup_feature"
// NATURAL-SUBGROUP: gpu.launch
// NATURAL-SUBGROUP: gpu.shuffle
// NATURAL-SUBGROUP: memref.store
// NATURAL-SUBGROUP-NOT: intent.
// NATURAL-MULTIROW-LABEL: func.func @natural_batch(
// NATURAL-MULTIROW-SAME: intent.backend.rows_per_block = 32 : i64
// NATURAL-MULTIROW: arith.constant 16384 : index
// NATURAL-MULTIROW: gpu.launch
// NATURAL-MULTIROW-COUNT-4: memref.store
// NATURAL-MULTIROW-NOT: intent.
// NATURAL-SUBWARP-LABEL: func.func @natural_batch(
// NATURAL-SUBWARP-SAME: intent.backend.rows_per_block = 32 : i64
// NATURAL-SUBWARP-SAME: intent.backend.subgroup_width = 4 : i64
// NATURAL-SUBWARP: arith.constant 16384 : index
// NATURAL-SUBWARP: gpu.launch
// NATURAL-SUBWARP: gpu.shuffle
// NATURAL-SUBWARP: memref.store
// NATURAL-SUBWARP-NOT: intent.
// NATURAL-PACKET-LABEL: func.func @natural_batch(
// NATURAL-PACKET-SAME: intent.backend.vector_width = 8 : i64
// NATURAL-PACKET-SAME: "cuda.memory.packetized_feature_load.v1"
// NATURAL-PACKET: vector.load
// NATURAL-PACKET: gpu.shuffle
// NATURAL-PACKET: memref.store
// NATURAL-PACKET-NOT: intent.
func.func @natural_batch(
    %a: tensor<4x16x8192x128xf16>,
    %b: tensor<4x16x8192x128xf16>) -> tensor<4x16x8192xf32> {
  %af = intent.cast %a {to_dtype = "f32"}
    : (tensor<4x16x8192x128xf16>) -> tensor<4x16x8192x128xf32>
  %bf = intent.cast %b {to_dtype = "f32"}
    : (tensor<4x16x8192x128xf16>) -> tensor<4x16x8192x128xf32>
  %product = intent.map_binary %af, %bf {fn = "mul"}
    : (tensor<4x16x8192x128xf32>, tensor<4x16x8192x128xf32>)
      -> tensor<4x16x8192x128xf32>
  %sum = intent.reduce %product
    {fn = "sum", axes = [3], assoc = "float_nondeterministic"}
    : (tensor<4x16x8192x128xf32>) -> tensor<4x16x8192xf32>
  %out = intent.pad %sum
    {lo = [0, 0, 0], hi = [0, 0, 0], value = 0.0 : f64}
    : (tensor<4x16x8192xf32>) -> tensor<4x16x8192xf32>
  return %out : tensor<4x16x8192xf32>
}
