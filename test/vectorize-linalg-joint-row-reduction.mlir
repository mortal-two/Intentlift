// RUN: intent-opt '-vectorize-linalg-row-reduction=vector-width=4 scalable-vector=true joint-reductions=true require-joint-reductions=true require-vectorization=true' -canonicalize -cse %s | FileCheck %s

#identity = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// Three polynomial moments share one root packet traversal.  This is a
// structural Full state: erasing vector width still leaves one loop carrying
// three accumulators instead of three independently scheduled reductions.
// CHECK: module attributes {
// CHECK-SAME: intent.rvv.explicit_vector_reduction_count = 3 : i64
// CHECK-SAME: intent.rvv.explicit_vector_reduction_loop_count = 2 : i64
// CHECK-SAME: intent.rvv.explicit_vector_reduction_scalable_count = 3 : i64
// CHECK-SAME: intent.rvv.joint_vector_reduction_count = 3 : i64
// CHECK-SAME: intent.rvv.joint_vector_reduction_group_count = 1 : i64
// CHECK-SAME: intent.rvv.joint_vector_reduction_root_load_count = 1 : i64
// CHECK-LABEL: func.func @three_moments
// CHECK: scf.for
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>)
// CHECK-COUNT-1: vector.transfer_read
// CHECK: arith.mulf
// CHECK-COUNT-3: vector.reduction <add>
// CHECK-NOT: linalg.generic
// CHECK: return
func.func @three_moments(%x: tensor<4x10xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %zero = arith.constant 0.0 : f32
  %x2Empty = tensor.empty() : tensor<4x10xf32>
  %x2 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%x : tensor<4x10xf32>) outs(%x2Empty : tensor<4x10xf32>) {
  ^bb0(%value: f32, %out: f32):
    %square = arith.mulf %value, %value : f32
    linalg.yield %square : f32
  } -> tensor<4x10xf32>
  %x3Empty = tensor.empty() : tensor<4x10xf32>
  %x3 = linalg.generic {
    indexing_maps = [#identity, #identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%x2, %x : tensor<4x10xf32>, tensor<4x10xf32>) outs(%x3Empty : tensor<4x10xf32>) {
  ^bb0(%square: f32, %value: f32, %out: f32):
    %cube = arith.mulf %square, %value : f32
    linalg.yield %cube : f32
  } -> tensor<4x10xf32>

  %init1Empty = tensor.empty() : tensor<4xf32>
  %init1 = linalg.fill ins(%zero : f32) outs(%init1Empty : tensor<4xf32>) -> tensor<4xf32>
  %init2Empty = tensor.empty() : tensor<4xf32>
  %init2 = linalg.fill ins(%zero : f32) outs(%init2Empty : tensor<4xf32>) -> tensor<4xf32>
  %init3Empty = tensor.empty() : tensor<4xf32>
  %init3 = linalg.fill ins(%zero : f32) outs(%init3Empty : tensor<4xf32>) -> tensor<4xf32>

  %moment1 = linalg.generic {
    indexing_maps = [#identity, #row],
    iterator_types = ["parallel", "reduction"]
  } ins(%x2 : tensor<4x10xf32>) outs(%init1 : tensor<4xf32>) {
  ^bb0(%value: f32, %acc: f32):
    %next = arith.addf %acc, %value fastmath<reassoc> : f32
    linalg.yield %next : f32
  } -> tensor<4xf32>
  %moment2 = linalg.generic {
    indexing_maps = [#identity, #row],
    iterator_types = ["parallel", "reduction"]
  } ins(%x2 : tensor<4x10xf32>) outs(%init2 : tensor<4xf32>) {
  ^bb0(%value: f32, %acc: f32):
    %fourth = arith.mulf %value, %value : f32
    %next = arith.addf %acc, %fourth fastmath<reassoc> : f32
    linalg.yield %next : f32
  } -> tensor<4xf32>
  %moment3 = linalg.generic {
    indexing_maps = [#identity, #row],
    iterator_types = ["parallel", "reduction"]
  } ins(%x3 : tensor<4x10xf32>) outs(%init3 : tensor<4xf32>) {
  ^bb0(%value: f32, %acc: f32):
    %sixth = arith.mulf %value, %value : f32
    %next = arith.addf %acc, %sixth fastmath<reassoc> : f32
    linalg.yield %next : f32
  } -> tensor<4xf32>
  return %moment1, %moment2, %moment3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}
