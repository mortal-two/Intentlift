// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=2 %s | FileCheck %s

// Flattening [B,G,H] can make M=B*G equal H.  The hidden iota is nevertheless
// provably a column coordinate because it reaches the trailing label of the
// `g,h->gh` contraction.  This is the reduced analogue of live GroupNorm.
// CHECK-LABEL: func.func @square_group_column_iota
// CHECK: gpu.launch
// CHECK: arith.index_cast
// CHECK-NOT: intent.
func.func @square_group_column_iota(%X: tensor<2x2x4xf32>)
    -> (tensor<2x2x4xf32>, tensor<2x2xf32>) {
  %sum = intent.reduce %X {fn = "sum", axes = [2], assoc = "float_nondeterministic"} : (tensor<2x2x4xf32>) -> tensor<2x2xf32>
  %hidden = intent.iota : tensor<4xi32>
  %two = arith.constant dense<2> : tensor<i32>
  %local = intent.map_binary %hidden, %two {fn = "div"} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %ones_group = arith.constant dense<1> : tensor<2xi32>
  %grid = intent.contraction %ones_group, %local {subscripts = "g,h->gh"} : (tensor<2xi32>, tensor<4xi32>) -> tensor<2x4xi32>
  %gridf = intent.cast %grid {to_dtype = "f32"} : (tensor<2x4xi32>) -> tensor<2x4xf32>
  %Y = intent.map_binary %X, %gridf {fn = "add"} : (tensor<2x2x4xf32>, tensor<2x4xf32>) -> tensor<2x2x4xf32>
  return %Y, %sum : tensor<2x2x4xf32>, tensor<2x2xf32>
}
