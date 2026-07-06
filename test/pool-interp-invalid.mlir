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

// -----

// conv output spatial extent must follow the conv formula.
func.func @bad_conv_ext(%x: tensor<2x4x16x16xf32>, %w: tensor<8x4x3x3xf32>) -> tensor<2x8x15x15xf32> {
  // expected-error @+1 {{conv output spatial extent mismatch on axis 0}}
  %o = intent.conv %x, %w {stride = [1, 1], padding = [0, 0], dilation = [1, 1], groups = 1 : i64}
      : (tensor<2x4x16x16xf32>, tensor<8x4x3x3xf32>) -> tensor<2x8x15x15xf32>
  return %o : tensor<2x8x15x15xf32>
}

// -----

// conv weight dim 1 must equal Cin/groups.
func.func @bad_conv_w(%x: tensor<2x4x16x16xf32>, %w: tensor<8x4x3x3xf32>) -> tensor<2x8x14x14xf32> {
  // expected-error @+1 {{conv weight dim 1 must equal Cin/groups}}
  %o = intent.conv %x, %w {stride = [1, 1], padding = [0, 0], dilation = [1, 1], groups = 2 : i64}
      : (tensor<2x4x16x16xf32>, tensor<8x4x3x3xf32>) -> tensor<2x8x14x14xf32>
  return %o : tensor<2x8x14x14xf32>
}

// -----

// conv stride/padding/dilation length must equal the spatial rank.
func.func @bad_conv_arr(%x: tensor<2x4x16x16xf32>, %w: tensor<8x4x3x3xf32>) -> tensor<2x8x14x14xf32> {
  // expected-error @+1 {{conv stride/padding/dilation must have length = spatial rank}}
  %o = intent.conv %x, %w {stride = [1], padding = [0, 0], dilation = [1, 1], groups = 1 : i64}
      : (tensor<2x4x16x16xf32>, tensor<8x4x3x3xf32>) -> tensor<2x8x14x14xf32>
  return %o : tensor<2x8x14x14xf32>
}
