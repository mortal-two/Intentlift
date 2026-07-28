// RUN: intent-opt '-tile-linalg-by-iterator-kind=parallel-tile=1 reduction-tile=8 fuse-producers=true' -canonicalize -cse %s | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: intent.schedule.fused_producer_count = {{[1-9][0-9]*}} : i64
// CHECK-SAME: intent.schedule.fusion_multi_use_rejection_count = {{[1-9][0-9]*}} : i64
// CHECK-SAME: intent.schedule.fusion_outer_parallel_only_count = {{[1-9][0-9]*}} : i64
// CHECK-LABEL: func.func @producer_reduction_epilogue
// CHECK: scf.for
// CHECK: tensor.extract_slice
// CHECK: linalg.generic
module {
  func.func @producer_reduction_epilogue(
      %input: tensor<2x16xf32>, %row_init: tensor<2xf32>,
      %output_init: tensor<2x16xf32>) -> tensor<2x16xf32> {
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%input : tensor<2x16xf32>)
        outs(%output_init : tensor<2x16xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %mapped = arith.mulf %value, %value : f32
        linalg.yield %mapped : f32
      } -> tensor<2x16xf32>
    %reduction = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%producer : tensor<2x16xf32>)
        outs(%row_init : tensor<2xf32>) {
      ^bb0(%value: f32, %acc: f32):
        %sum = arith.addf %acc, %value : f32
        linalg.yield %sum : f32
      } -> tensor<2xf32>
    %epilogue = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%reduction, %producer : tensor<2xf32>, tensor<2x16xf32>)
        outs(%output_init : tensor<2x16xf32>) {
      ^bb0(%sum: f32, %mapped: f32, %unused: f32):
        %normalized = arith.addf %mapped, %sum : f32
        linalg.yield %normalized : f32
      } -> tensor<2x16xf32>
    return %epilogue : tensor<2x16xf32>
  }
}
