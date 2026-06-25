// RUN: intent-opt -split-input-file -verify-diagnostics %s

// Labels must contain '->'.
func.func @no_arrow(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{subscripts must contain '->'}}
  %c = intent.contraction %a, %b {subscripts = "ij,jk"}
       : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// -----

// Exactly two inputs.
func.func @three_inputs(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{exactly two inputs}}
  %c = intent.contraction %a, %b {subscripts = "ij,jk,kl->il"}
       : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// -----

// Label count must equal operand rank.
func.func @rank_mismatch(%a: tensor<4x16xf32>, %b: tensor<4x8x16xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{rhs labels 'kd' (2) != operand rank 3}}
  %c = intent.contraction %a, %b {subscripts = "qd,kd->qk"}
       : (tensor<4x16xf32>, tensor<4x8x16xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// -----

// A shared contraction index must have one consistent extent.
func.func @dim_conflict(%a: tensor<4x16xf32>, %b: tensor<4x32xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{index 'd' has conflicting extents 16 and 32}}
  %c = intent.contraction %a, %b {subscripts = "qd,kd->qk"}
       : (tensor<4x16xf32>, tensor<4x32xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// -----

// Result index must come from an input.
func.func @phantom_result_index(%a: tensor<4x16xf32>, %b: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{result index 'z' does not appear in any input}}
  %c = intent.contraction %a, %b {subscripts = "qd,kd->qz"}
       : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}
