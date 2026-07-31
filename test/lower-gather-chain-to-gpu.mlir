// RUN: intent-opt -convert-intent-gather-to-gpu=block-threads=32 %s | FileCheck %s --check-prefix=MATERIALIZED
// RUN: intent-opt -convert-intent-gather-to-gpu='block-threads=32 compose-gather-chain=true' %s | FileCheck %s --check-prefix=COMPOSED

// The accepted graph contains two gathers.  DirectLower evaluates the complete
// first gathered row before the second gather selects one element.  Full may
// compose the two runtime index maps and eliminate that intermediate row.

// MATERIALIZED-LABEL: func.func @gather_chain(
// MATERIALIZED-SAME: memref<16x8xf32>
// MATERIALIZED-SAME: memref<16xi32>
// MATERIALIZED-SAME: memref<16xi32>
// MATERIALIZED-SAME: memref<16xf32>
// MATERIALIZED-SAME: intent.backend.lowering_path_kind = "gather"
// MATERIALIZED-SAME: intent.backend.transformation_trace = ["cuda.gather-chain.materialized-row.v1"]
// MATERIALIZED: gpu.launch
// MATERIALIZED: gpu.dynamic_shared_memory
// MATERIALIZED: memref.view {{.*}} to memref<8xf32, #gpu.address_space<workgroup>>
// MATERIALIZED: memref.load {{.*}} : memref<16x8xf32>
// MATERIALIZED: gpu.barrier
// MATERIALIZED: memref.load {{.*}} : memref<8xf32, #gpu.address_space<workgroup>>
// MATERIALIZED-NOT: intent.gather

// COMPOSED-LABEL: func.func @gather_chain(
// COMPOSED-SAME: intent.backend.lowering_path_kind = "gather_composed"
// COMPOSED-SAME: intent.backend.transformation_trace = ["cuda.gather-chain.composed-index.v1"]
// COMPOSED: gpu.launch
// COMPOSED-NOT: gpu.dynamic_shared_memory
// COMPOSED-NOT: gpu.barrier
// COMPOSED: memref.load {{.*}} : memref<16xi32>
// COMPOSED: memref.load {{.*}} : memref<16xi32>
// COMPOSED: memref.load {{.*}} : memref<16x8xf32>
// COMPOSED-NOT: intent.gather
func.func @gather_chain(
    %data: tensor<16x8xf32>,
    %rows: tensor<16xi32>,
    %columns: tensor<16xi32>) -> tensor<16xf32> {
  %row_values = intent.gather %data, %rows {axis = 0 : i64, mode = "take"}
      : (tensor<16x8xf32>, tensor<16xi32>) -> tensor<16x8xf32>
  %expanded_columns = tensor.expand_shape %columns [[0, 1]] output_shape [16, 1]
      : tensor<16xi32> into tensor<16x1xi32>
  %selected = intent.gather %row_values, %expanded_columns
      {axis = 1 : i64, mode = "along_axis"}
      : (tensor<16x8xf32>, tensor<16x1xi32>) -> tensor<16x1xf32>
  %result = tensor.collapse_shape %selected [[0, 1]]
      : tensor<16x1xf32> into tensor<16xf32>
  return %result : tensor<16xf32>
}
