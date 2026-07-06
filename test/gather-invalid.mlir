// RUN: intent-opt -split-input-file -verify-diagnostics %s

// index must be integer-typed.
func.func @float_index(%d: tensor<4x8xf32>, %i: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{index must be an integer-typed tensor}}
  %g = intent.gather %d, %i {axis = 1 : i64, mode = "along_axis"}
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %g : tensor<4x8xf32>
}

// -----

// axis out of range.
func.func @bad_axis(%d: tensor<4x8xf32>, %i: tensor<4x8xi64>) -> tensor<4x8xf32> {
  // expected-error @+1 {{axis 2 out of range for rank-2 data}}
  %g = intent.gather %d, %i {axis = 2 : i64, mode = "along_axis"}
      : (tensor<4x8xf32>, tensor<4x8xi64>) -> tensor<4x8xf32>
  return %g : tensor<4x8xf32>
}

// -----

// along_axis: index rank must equal data rank.
func.func @rank_mismatch(%d: tensor<4x8xf32>, %i: tensor<4xi64>) -> tensor<4x8xf32> {
  // expected-error @+1 {{along_axis gather: index rank 1 must equal data rank 2}}
  %g = intent.gather %d, %i {axis = 1 : i64, mode = "along_axis"}
      : (tensor<4x8xf32>, tensor<4xi64>) -> tensor<4x8xf32>
  return %g : tensor<4x8xf32>
}

// -----

// take: result rank must be data.rank-1+index.rank.
func.func @take_rank(%d: tensor<6x8xf32>, %i: tensor<4xi64>) -> tensor<4x8x2xf32> {
  // expected-error @+1 {{take gather: result rank 3 != data.rank-1+index.rank (2)}}
  %g = intent.gather %d, %i {axis = 0 : i64, mode = "take"}
      : (tensor<6x8xf32>, tensor<4xi64>) -> tensor<4x8x2xf32>
  return %g : tensor<4x8x2xf32>
}

// -----

// scatter reduce must be 'none' or 'add'.
func.func @bad_reduce(%d: tensor<4x8xf32>, %i: tensor<4x8xi64>, %s: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
  // expected-error @+1 {{reduce must be 'none' or 'add'}}
  %o = intent.scatter %d, %i, %s {axis = 1 : i64, mode = "along_axis", reduce = "mul"}
      : (tensor<4x8xf32>, tensor<4x8xi64>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %o : tensor<4x8xf32>
}
