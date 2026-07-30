// RUN: intent-opt '-vectorize-linalg-row-reduction=vector-width=4 scalable-vector=true joint-reductions=true require-joint-reductions=true require-vectorization=true' -canonicalize -cse %s | FileCheck %s

#identity2 = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#scalar2 = affine_map<(d0, d1) -> ()>
#identity1 = affine_map<(d0) -> (d0)>
#scalar1 = affine_map<(d0) -> ()>
#scalar0 = affine_map<() -> ()>

// A certified scalar equality proves that the masked centered-variance
// reduction covers the complete feature axis.  Full may therefore accumulate
// sum(x) and sum(x^2) in one traversal and reconstruct the centered sum as
// sum(x^2) - N * mean^2.  Without the equality this pattern must fail closed.
// CHECK: module attributes {
// CHECK-SAME: intent.rvv.explicit_vector_reduction_count = 2 : i64
// CHECK-SAME: intent.rvv.joint_raw_moment_variance_count = 1 : i64
// CHECK-SAME: intent.rvv.joint_vector_reduction_count = 2 : i64
// CHECK-SAME: intent.rvv.joint_vector_reduction_group_count = 1 : i64
// CHECK-SAME: intent.rvv.joint_vector_reduction_root_load_count = 1 : i64
// CHECK-LABEL: func.func @masked_centered_variance
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<[4]xf32>, vector<[4]xf32>)
// CHECK-COUNT-1: vector.transfer_read
// CHECK-COUNT-2: vector.reduction <add>
// CHECK: arith.divf
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK-NOT: linalg.index
// CHECK: return
func.func @masked_centered_variance(
    %x: tensor<4x10xf32>, %length: tensor<i32>)
    -> (tensor<4xf32>, tensor<4xf32>)
    attributes {intent.scalar_equalities = array<i64: 1, 10>} {
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
