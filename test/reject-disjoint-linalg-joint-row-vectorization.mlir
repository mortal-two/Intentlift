// RUN: not intent-opt '-vectorize-linalg-row-reduction=vector-width=4 scalable-vector=true joint-reductions=true require-joint-reductions=true' %s 2>&1 | FileCheck %s

#identity = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// CHECK: error: required joint row-reduction vectorization matched no group

// Independent reductions over different roots are not a shared-traversal
// state, even when their shape and reduction axis happen to be identical.
func.func @different_roots(%x: tensor<4x10xf32>, %y: tensor<4x10xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %zero = arith.constant 0.0 : f32
  %initXEmpty = tensor.empty() : tensor<4xf32>
  %initX = linalg.fill ins(%zero : f32) outs(%initXEmpty : tensor<4xf32>) -> tensor<4xf32>
  %initYEmpty = tensor.empty() : tensor<4xf32>
  %initY = linalg.fill ins(%zero : f32) outs(%initYEmpty : tensor<4xf32>) -> tensor<4xf32>
  %sumX = linalg.generic {
    indexing_maps = [#identity, #row],
    iterator_types = ["parallel", "reduction"]
  } ins(%x : tensor<4x10xf32>) outs(%initX : tensor<4xf32>) {
  ^bb0(%value: f32, %acc: f32):
    %next = arith.addf %acc, %value fastmath<reassoc> : f32
    linalg.yield %next : f32
  } -> tensor<4xf32>
  %sumY = linalg.generic {
    indexing_maps = [#identity, #row],
    iterator_types = ["parallel", "reduction"]
  } ins(%y : tensor<4x10xf32>) outs(%initY : tensor<4xf32>) {
  ^bb0(%value: f32, %acc: f32):
    %next = arith.addf %acc, %value fastmath<reassoc> : f32
    linalg.yield %next : f32
  } -> tensor<4xf32>
  return %sumX, %sumY : tensor<4xf32>, tensor<4xf32>
}
