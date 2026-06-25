// RUN: intent-opt -split-input-file -verify-diagnostics %s

// #inits must equal #results (carry types are loop-invariant).
func.func @carry_count(%K: tensor<8x4xf32>, %m0: tensor<4x1xf32>, %O0: tensor<4x4xf32>)
    -> tensor<4x1xf32> {
  // expected-error @+1 {{number of inits (2) must equal number of results (1)}}
  %r = intent.loop scan(%K : tensor<8x4xf32>)
                   init(%m0, %O0 : tensor<4x1xf32>, tensor<4x4xf32>)
                   {scan_axes = [0]} -> tensor<4x1xf32> do {
  ^bb0(%Kj: tensor<2x4xf32>, %m: tensor<4x1xf32>, %Oacc: tensor<4x4xf32>):
    intent.yield %m, %Oacc : tensor<4x1xf32>, tensor<4x4xf32>
  }
  return %r : tensor<4x1xf32>
}

// -----

// scan_axes must have one entry per scan input.
func.func @scan_axes_count(%K: tensor<8x4xf32>, %V: tensor<8x4xf32>, %m0: tensor<4x1xf32>)
    -> tensor<4x1xf32> {
  // expected-error @+1 {{scan_axes (1) must have one entry per scan input (2)}}
  %r = intent.loop scan(%K, %V : tensor<8x4xf32>, tensor<8x4xf32>)
                   init(%m0 : tensor<4x1xf32>)
                   {scan_axes = [0]} -> tensor<4x1xf32> do {
  ^bb0(%Kj: tensor<2x4xf32>, %Vj: tensor<2x4xf32>, %m: tensor<4x1xf32>):
    intent.yield %m : tensor<4x1xf32>
  }
  return %r : tensor<4x1xf32>
}

// -----

// A carry block arg type must match its init type.
func.func @carry_type(%K: tensor<8x4xf32>, %m0: tensor<4x1xf32>) -> tensor<4x1xf32> {
  // expected-error @+1 {{carry block arg 0 type 'tensor<4x2xf32>' != init type 'tensor<4x1xf32>'}}
  %r = intent.loop scan(%K : tensor<8x4xf32>)
                   init(%m0 : tensor<4x1xf32>)
                   {scan_axes = [0]} -> tensor<4x1xf32> do {
  ^bb0(%Kj: tensor<2x4xf32>, %m: tensor<4x2xf32>):
    intent.yield %m0 : tensor<4x1xf32>
  }
  return %r : tensor<4x1xf32>
}
