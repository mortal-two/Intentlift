// RUN: intent-opt '-tile-linalg-by-iterator-kind=parallel-tile=4 reduction-tile=8' %s | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: intent.schedule.generated_loop_count = 2 : i64
// CHECK-SAME: intent.schedule.linalg_tiled_op_count = 1 : i64
// CHECK-SAME: intent.schedule.parallel_tile = 4 : i64
// CHECK-SAME: intent.schedule.reduction_tile = 8 : i64
// CHECK-LABEL: func.func @row_sum
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.generic
// CHECK-NOT: linalg.generic
module {
  func.func @row_sum(%input: tensor<2x16xf32>, %init: tensor<2xf32>)
      -> tensor<2xf32> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<2x16xf32>)
      outs(%init : tensor<2xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %acc, %value : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}
