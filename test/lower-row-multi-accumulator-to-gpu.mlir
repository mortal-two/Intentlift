// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=32 tile-cols=4 accumulator-chains=4' %s | FileCheck %s

// Full may split each lane's serial feature strip into independent accumulator
// chains before the ordinary warp tree.  The post-pass state, rather than the
// requested option, records that the emitted floating reduction tree changed.
// CHECK-LABEL: func.func @row_sum_multi_accumulator(
// CHECK-SAME: intent.backend.accumulator_chains = 4 : i64
// CHECK-SAME: intent.backend.reduction_topology = "subgroup_shuffle"
// CHECK-SAME: "cuda.reduction.per_thread_multi_accumulator_tree.v1"
// CHECK: gpu.shuffle xor
// CHECK-NOT: intent.
func.func @row_sum_multi_accumulator(%X: tensor<2x128xf32>)
    -> tensor<2x1xf32> {
  %sum = intent.reduce %X {
    fn = "sum", axes = [1], assoc = "float_nondeterministic"
  } : (tensor<2x128xf32>) -> tensor<2x1xf32>
  return %sum : tensor<2x1xf32>
}
