// RUN: intent-opt -split-input-file -verify-diagnostics %s

// arg_reduce fn must be argmax/argmin.
func.func @bad_fn(%x: tensor<4x8xf32>) -> tensor<4xi32> {
  // expected-error @+1 {{unknown arg_reduce fn 'argmedian'}}
  %i = intent.arg_reduce %x {fn = "argmedian", axis = 1 : i64} : (tensor<4x8xf32>) -> tensor<4xi32>
  return %i : tensor<4xi32>
}

// -----

// result must be integer-typed (it is an index).
func.func @not_int(%x: tensor<4x8xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{result must be an integer-typed tensor}}
  %i = intent.arg_reduce %x {fn = "argmax", axis = 1 : i64} : (tensor<4x8xf32>) -> tensor<4xf32>
  return %i : tensor<4xf32>
}

// -----

// axis out of range.
func.func @bad_axis(%x: tensor<4x8xf32>) -> tensor<4xi32> {
  // expected-error @+1 {{axis 3 out of range}}
  %i = intent.arg_reduce %x {fn = "argmax", axis = 3 : i64} : (tensor<4x8xf32>) -> tensor<4xi32>
  return %i : tensor<4xi32>
}

// -----

// result rank must be input rank (keepdims) or rank-1 (collapsed).
func.func @bad_rank(%x: tensor<4x8xf32>) -> tensor<i32> {
  // expected-error @+1 {{result rank must be input rank}}
  %i = intent.arg_reduce %x {fn = "argmax", axis = 1 : i64} : (tensor<4x8xf32>) -> tensor<i32>
  return %i : tensor<i32>
}
