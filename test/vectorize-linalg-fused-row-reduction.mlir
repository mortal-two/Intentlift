// RUN: intent-opt '-vectorize-linalg-row-reduction=vector-width=4 scalable-vector=true require-vectorization=true' %s | FileCheck %s

#identity2 = affine_map<(d0, d1) -> (d0, d1)>
#rows1 = affine_map<(d0, d1) -> (d0)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#rows2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: module attributes {
// CHECK-SAME: intent.rvv.explicit_vector_reduction_count = 2 : i64
// CHECK-SAME: intent.rvv.explicit_vector_reduction_loop_count = 5 : i64
// CHECK-SAME: intent.rvv.explicit_vector_reduction_scalable_count = 2 : i64

// CHECK-LABEL: func.func @square_sum
// CHECK: vector.transfer_read {{.*}} : tensor<4x10xf32>, vector<[4]xf32>
// CHECK: arith.mulf {{.*}} : vector<[4]xf32>
// CHECK: arith.addf {{.*}} fastmath<reassoc> : vector<[4]xf32>
// CHECK: vector.reduction <add>
// CHECK-NOT: linalg.generic
// CHECK: return
func.func @square_sum(%x: tensor<4x10xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
  %result = linalg.generic {
    indexing_maps = [#identity2, #rows1],
    iterator_types = ["parallel", "reduction"]
  } ins(%x : tensor<4x10xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%value: f32, %acc: f32):
    %square = arith.mulf %value, %value : f32
    %next = arith.addf %acc, %square fastmath<reassoc> : f32
    linalg.yield %next : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// CHECK-LABEL: func.func @unit_parallel_sum
// CHECK: scf.for
// CHECK: scf.for
// CHECK: vector.transfer_read {{.*}} : tensor<4x1x10xf32>, vector<[4]xf32>
// CHECK: vector.reduction <add>
// CHECK-NOT: linalg.generic
// CHECK: return
func.func @unit_parallel_sum(%x: tensor<4x1x10xf32>, %init: tensor<4x1xf32>) -> tensor<4x1xf32> {
  %result = linalg.generic {
    indexing_maps = [#identity3, #rows2],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%x : tensor<4x1x10xf32>) outs(%init : tensor<4x1xf32>) {
  ^bb0(%value: f32, %acc: f32):
    %next = arith.addf %acc, %value fastmath<reassoc> : f32
    linalg.yield %next : f32
  } -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}
