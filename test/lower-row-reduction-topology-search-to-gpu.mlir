// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=64 tile-cols=32 compact-reduction-topology=false promote-rank0-scalars=false defer-epilogue-col-loads=false' %s | FileCheck %s --check-prefix=ALLREDUCE
// RUN: intent-opt -convert-intent-row-reduction-to-gpu='block-threads=64 tile-cols=32 compact-reduction-topology=true promote-rank0-scalars=false defer-epilogue-col-loads=false' %s | FileCheck %s --check-prefix=COMPACT

// Both states consume the same typed Intent graph and retain the same read-once
// row/epilogue lowering.  Only the concrete reduction topology changes: the
// canonical MLIR collective versus the compiler-owned shuffle/shared tree.

// ALLREDUCE-LABEL: func.func @row_topology(
// ALLREDUCE-SAME: intent.backend.reduction_topology = "gpu_all_reduce"
// ALLREDUCE: gpu.all_reduce
// ALLREDUCE-NOT: gpu.shuffle

// COMPACT-LABEL: func.func @row_topology(
// COMPACT-SAME: intent.backend.reduction_topology = "workgroup_hierarchical"
// COMPACT: gpu.shuffle
// COMPACT-NOT: gpu.all_reduce

func.func @row_topology(%X: tensor<64x128xf32>, %eps: tensor<f32>)
    -> tensor<64x128xf32> {
  %x2 = intent.map_binary %X, %X {fn = "mul"}
    : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %mean = intent.reduce %x2 {fn = "mean", axes = [1], assoc = "float_nondeterministic"}
    : (tensor<64x128xf32>) -> tensor<64x1xf32>
  %shifted = intent.map_binary %mean, %eps {fn = "add"}
    : (tensor<64x1xf32>, tensor<f32>) -> tensor<64x1xf32>
  %Y = intent.map_binary %X, %shifted {fn = "mul"}
    : (tensor<64x128xf32>, tensor<64x1xf32>) -> tensor<64x128xf32>
  return %Y : tensor<64x128xf32>
}
