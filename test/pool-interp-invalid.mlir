// RUN: intent-opt -split-input-file -verify-diagnostics %s

// pool fn must be max/avg.
func.func @bad_fn(%a: tensor<1x2x8x8xf32>) -> tensor<1x2x4x4xf32> {
  // expected-error @+1 {{pool fn must be 'max' or 'avg'}}
  %o = intent.pool %a {fn = "median", kH = 2 : i64, kW = 2 : i64, sH = 2 : i64,
      sW = 2 : i64, padH = 0 : i64, padW = 0 : i64}
      : (tensor<1x2x8x8xf32>) -> tensor<1x2x4x4xf32>
  return %o : tensor<1x2x4x4xf32>
}

// -----

// pool operates on rank-4 NCHW tensors.
func.func @bad_rank(%a: tensor<8x8xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{pool operates on rank-4 NCHW tensors}}
  %o = intent.pool %a {fn = "avg", kH = 2 : i64, kW = 2 : i64, sH = 2 : i64,
      sW = 2 : i64, padH = 0 : i64, padW = 0 : i64}
      : (tensor<8x8xf32>) -> tensor<4x4xf32>
  return %o : tensor<4x4xf32>
}

// -----

// pool indices only valid for fn = 'max'.
func.func @avg_idx(%a: tensor<1x2x8x8xf32>) -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xi64>) {
  // expected-error @+1 {{pool indices result is only valid for fn = 'max'}}
  %o, %i = intent.pool %a {fn = "avg", kH = 2 : i64, kW = 2 : i64, sH = 2 : i64,
      sW = 2 : i64, padH = 0 : i64, padW = 0 : i64}
      : (tensor<1x2x8x8xf32>) -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xi64>)
  return %o, %i : tensor<1x2x4x4xf32>, tensor<1x2x4x4xi64>
}

// -----

// upsample mode must be 'nearest'.
func.func @bad_mode(%a: tensor<1x2x4x4xf32>) -> tensor<1x2x8x8xf32> {
  // expected-error @+1 {{upsample mode must be 'nearest'}}
  %o = intent.upsample %a {mode = "bilinear"} : (tensor<1x2x4x4xf32>) -> tensor<1x2x8x8xf32>
  return %o : tensor<1x2x8x8xf32>
}

// -----

// upsample preserves the N and C extents.
func.func @bad_nc(%a: tensor<1x2x4x4xf32>) -> tensor<1x4x8x8xf32> {
  // expected-error @+1 {{upsample preserves the N and C extents}}
  %o = intent.upsample %a {mode = "nearest"} : (tensor<1x2x4x4xf32>) -> tensor<1x4x8x8xf32>
  return %o : tensor<1x4x8x8xf32>
}

// -----

// polar real/imag must match the abs shape+element type.
func.func @bad_polar(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> (tensor<4x4xf32>, tensor<4x8xf32>) {
  // expected-error @+1 {{polar real/imag must match the abs shape+element type}}
  %r, %i = intent.polar %a, %b : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x4xf32>, tensor<4x8xf32>)
  return %r, %i : tensor<4x4xf32>, tensor<4x8xf32>
}
