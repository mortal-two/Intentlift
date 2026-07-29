// RUN: not intent-opt '-vectorize-linalg-row-reduction=vector-width=4 scalable-vector=true joint-reductions=true require-joint-reductions=true' %s 2>&1 | FileCheck %s

#identity2 = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#scalar2 = affine_map<(d0, d1) -> ()>
#identity1 = affine_map<(d0) -> (d0)>
#scalar1 = affine_map<(d0) -> ()>
#scalar0 = affine_map<() -> ()>

// CHECK: error: required joint row-reduction vectorization matched no group

// The runtime mask bound has no certified equality to the static feature
// extent.  Replacing the masked centered-square sum with full-domain raw
// moments would therefore be unsound and must fail closed.
func.func @unknown_mask_bound(
    %x: tensor<4x10xf32>, %length: tensor<i32>)
    -> (tensor<4xf32>, tensor<4xf32>) {
  %zero = arith.constant 0.0 : f32
  %meanInitEmpty = tensor.empty() : tensor<4xf32>
  %meanInit = linalg.fill ins(%zero : f32) outs(%meanInitEmpty : tensor<4xf32>) -> tensor<4xf32>
  %meanSum = linalg.generic {
    indexing_maps = [#identity2, #row],
    iterator_types = ["parallel", "reduction"]
  } ins(%x : tensor<4x10xf32>) outs(%meanInit : tensor<4xf32>) {
  ^bb0(%value: f32, %acc: f32):
    %next = arith.addf %acc, %value fastmath<reassoc> : f32
    linalg.yield %next : f32
  } -> tensor<4xf32>

  %lengthFloatEmpty = tensor.empty() : tensor<f32>
  %lengthFloat = linalg.generic {
    indexing_maps = [#scalar0, #scalar0], iterator_types = []
  } ins(%length : tensor<i32>) outs(%lengthFloatEmpty : tensor<f32>) {
  ^bb0(%value: i32, %out: f32):
    %cast = arith.sitofp %value : i32 to f32
    linalg.yield %cast : f32
  } -> tensor<f32>
  %meanEmpty = tensor.empty() : tensor<4xf32>
  %mean = linalg.generic {
    indexing_maps = [#identity1, #scalar1, #identity1],
    iterator_types = ["parallel"]
  } ins(%meanSum, %lengthFloat : tensor<4xf32>, tensor<f32>) outs(%meanEmpty : tensor<4xf32>) {
  ^bb0(%sum: f32, %count: f32, %out: f32):
    %average = arith.divf %sum, %count : f32
    linalg.yield %average : f32
  } -> tensor<4xf32>

  %centeredEmpty = tensor.empty() : tensor<4x10xf32>
  %centered = linalg.generic {
    indexing_maps = [#identity2, #row, #identity2],
    iterator_types = ["parallel", "parallel"]
  } ins(%x, %mean : tensor<4x10xf32>, tensor<4xf32>) outs(%centeredEmpty : tensor<4x10xf32>) {
  ^bb0(%value: f32, %average: f32, %out: f32):
    %difference = arith.subf %value, %average : f32
    linalg.yield %difference : f32
  } -> tensor<4x10xf32>

  %varianceInitEmpty = tensor.empty() : tensor<4xf32>
  %varianceInit = linalg.fill ins(%zero : f32) outs(%varianceInitEmpty : tensor<4xf32>) -> tensor<4xf32>
  %variance = linalg.generic {
    indexing_maps = [#scalar2, #identity2, #row],
    iterator_types = ["parallel", "reduction"]
  } ins(%length, %centered : tensor<i32>, tensor<4x10xf32>) outs(%varianceInit : tensor<4xf32>) {
  ^bb0(%bound: i32, %value: f32, %acc: f32):
    %column = linalg.index 1 : index
    %columnI32 = arith.index_cast %column : index to i32
    %active = arith.cmpi slt, %columnI32, %bound : i32
    %masked = arith.select %active, %value, %zero : f32
    %square = arith.mulf %masked, %masked : f32
    %next = arith.addf %acc, %square fastmath<reassoc> : f32
    linalg.yield %next : f32
  } -> tensor<4xf32>
  return %mean, %variance : tensor<4xf32>, tensor<4xf32>
}
