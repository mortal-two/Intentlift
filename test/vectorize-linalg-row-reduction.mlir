// RUN: intent-opt '-convert-intent-to-linalg=reassociate-float-reductions=true' '-vectorize-linalg-row-reduction=vector-width=4 require-vectorization=true' %s | FileCheck %s --check-prefix=FIXED
// RUN: intent-opt '-convert-intent-to-linalg=reassociate-float-reductions=true' '-vectorize-linalg-row-reduction=vector-width=4 scalable-vector=true require-vectorization=true' %s | FileCheck %s --check-prefix=SCALABLE

// The reduction extent deliberately has a tail.  The transformed state keeps
// one vector accumulator across the complete row and performs one horizontal
// finalize, rather than reducing every strip to a scalar.

// FIXED: module attributes {
// FIXED-SAME: intent.rvv.explicit_vector_reduction_count = 1 : i64
// FIXED-SAME: intent.rvv.explicit_vector_reduction_loop_count = 2 : i64
// FIXED-SAME: intent.rvv.explicit_vector_reduction_scalable_count = 0 : i64
// FIXED-LABEL: func.func @row_sum
// FIXED: scf.for
// FIXED: vector.splat {{.*}} : vector<4xf32>
// FIXED: scf.for {{.*}} step %{{.*}} iter_args({{.*}}) -> (vector<4xf32>)
// FIXED: vector.transfer_read {{.*}} : tensor<4x10xf32>, vector<4xf32>
// FIXED: arith.addf {{.*}} fastmath<reassoc> : vector<4xf32>
// FIXED: vector.reduction <add>
// FIXED-SAME: vector<4xf32> into f32
// FIXED-NOT: vector.reduction <add>
// FIXED: return

// SCALABLE: module attributes {
// SCALABLE-SAME: intent.rvv.explicit_vector_reduction_count = 1 : i64
// SCALABLE-SAME: intent.rvv.explicit_vector_reduction_loop_count = 2 : i64
// SCALABLE-SAME: intent.rvv.explicit_vector_reduction_scalable_count = 1 : i64
// SCALABLE-LABEL: func.func @row_sum
// SCALABLE: %[[VSCALE:.*]] = vector.vscale
// SCALABLE: %[[STEP:.*]] = arith.muli {{.*}}, %[[VSCALE]] : index
// SCALABLE: vector.splat {{.*}} : vector<[4]xf32>
// SCALABLE: scf.for {{.*}} step %[[STEP]] iter_args({{.*}}) -> (vector<[4]xf32>)
// SCALABLE: vector.transfer_read {{.*}} : tensor<4x10xf32>, vector<[4]xf32>
// SCALABLE: arith.addf {{.*}} fastmath<reassoc> : vector<[4]xf32>
// SCALABLE: vector.reduction <add>
// SCALABLE-SAME: vector<[4]xf32> into f32
// SCALABLE-NOT: vector.reduction <add>
// SCALABLE: return

func.func @row_sum(%x: tensor<4x10xf32>) -> tensor<4xf32> {
  %sum = intent.reduce %x {
    fn = "sum", axes = [1], assoc = "float_nondeterministic"
  } : (tensor<4x10xf32>) -> tensor<4xf32>
  return %sum : tensor<4xf32>
}
