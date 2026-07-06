// RUN: intent-opt -split-input-file -verify-diagnostics %s

// concat: sum of input axis extents must equal the result axis extent.
func.func @concat_sum(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x9xf32> {
  // expected-error @+1 {{sum of input axis extents 8 != result axis extent 9}}
  %c = intent.concat %a, %b {axis = 1 : i64}
      : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x9xf32>
  return %c : tensor<4x9xf32>
}

// -----

// concat: non-concat extents must agree.
func.func @concat_nonaxis(%a: tensor<4x4xf32>, %b: tensor<5x4xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{input extent 5 != result extent 4 at non-concat dim 0}}
  %c = intent.concat %a, %b {axis = 1 : i64}
      : (tensor<4x4xf32>, tensor<5x4xf32>) -> tensor<4x8xf32>
  return %c : tensor<4x8xf32>
}

// -----

// pad: result extent must be input+lo+hi.
func.func @pad_extent(%a: tensor<4x4xf32>) -> tensor<6x6xf32> {
  // expected-error @+1 {{result extent 6 != input+lo+hi (7) at dim 0}}
  %p = intent.pad %a {lo = [1, 1], hi = [2, 1], value = 0.0 : f64}
      : (tensor<4x4xf32>) -> tensor<6x6xf32>
  return %p : tensor<6x6xf32>
}

// -----

// tile: result extent must be input*reps.
func.func @tile_extent(%a: tensor<2x4xf32>) -> tensor<4x9xf32> {
  // expected-error @+1 {{result extent 9 != input*reps (8) at dim 1}}
  %t = intent.tile %a {reps = [2, 2]} : (tensor<2x4xf32>) -> tensor<4x9xf32>
  return %t : tensor<4x9xf32>
}

// -----

// flip: axis out of range.
func.func @flip_axis(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{flip axis 2 out of range for rank-2 input}}
  %f = intent.flip %a {axes = [2]} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %f : tensor<4x8xf32>
}

// -----

// triu: rank-2 only.
func.func @triu_rank(%a: tensor<4x8x2xf32>) -> tensor<4x8x2xf32> {
  // expected-error @+1 {{triu operates on a rank-2 tensor}}
  %u = intent.triu %a {k = 0 : i64} : (tensor<4x8x2xf32>) -> tensor<4x8x2xf32>
  return %u : tensor<4x8x2xf32>
}
