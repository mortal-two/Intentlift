// RUN: intent-opt -split-input-file -verify-diagnostics %s

// rand must produce an f32 tensor (uniform[0,1)).
func.func @rand_not_f32() -> tensor<4x8xf16> {
  // expected-error @+1 {{result element type must be f32}}
  %r = intent.rand {seed = 1 : i64} : tensor<4x8xf16>
  return %r : tensor<4x8xf16>
}

// -----

// integer result is rejected too.
func.func @rand_int() -> tensor<4x8xi32> {
  // expected-error @+1 {{result element type must be f32}}
  %r = intent.rand {seed = 1 : i64} : tensor<4x8xi32>
  return %r : tensor<4x8xi32>
}
