// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=32 allow-execution-domain-sorted-prefix=true require-sorted-prefix=true' %s | FileCheck %s
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=32 tile-cols=128 vector-width=4 rows-per-block=2 allow-execution-domain-sorted-prefix=true require-sorted-prefix=true' %s | FileCheck %s --check-prefix=MULTI
// RUN: intent-opt -convert-intent-row-reduction-to-gpu=block-threads=32 %s | FileCheck %s --check-prefix=GENERIC

// This graph has no Layer-B descending-order marker.  The explicit
// execution-domain candidate may still materialize the true C++/MLIR prefix
// topology, but it records the assumption instead of claiming a contract fact.
// Production search must pass Q/D before this state is timed or selected.
// CHECK-LABEL: func.func @execution_domain_prefix_graph(
// CHECK-SAME: memref<2x32xvector<4xf32>>
// CHECK-SAME: intent.backend.execution_domain_assumption = "sorted_descending_input"
// CHECK-SAME: intent.backend.reduction_topology = "sorted_prefix_early_stop"
// CHECK-SAME: intent.fast_path_id = "cuda.row_scan.sparsemax.sorted_prefix.execution_domain.v1"
// CHECK: gpu.launch
// CHECK: scf.for
// CHECK: gpu.shuffle up
// CHECK: arith.cmpf ogt
// CHECK: arith.maxnumf
// CHECK: gpu.terminator
// CHECK-NOT: intent_sorted_descending

// MULTI-LABEL: func.func @execution_domain_prefix_graph(
// MULTI-SAME: intent.backend.block_threads = 64 : i64
// MULTI-SAME: intent.backend.reduction_topology = "sorted_prefix_multirow_warp_early_stop"
// MULTI-SAME: intent.backend.rows_per_block = 2 : i64
// MULTI-SAME: intent.fast_path_id = "cuda.row_scan.sparsemax.sorted_prefix.execution_domain.v1"
// MULTI: %[[GRID:.*]] = arith.constant 1 : index
// MULTI: %[[BLOCK:.*]] = arith.constant 64 : index
// MULTI: gpu.launch{{.*}}%[[GRID]]{{.*}}%[[BLOCK]]
// MULTI: gpu.shuffle up
// MULTI: gpu.shuffle xor
// MULTI-NOT: gpu.dynamic_shared_memory
// MULTI-NOT: gpu.barrier

// Without the explicit execution-domain state, absence of the B marker remains
// fail-closed and the same graph takes the complete generic scan lowering.
// GENERIC-LABEL: func.func @execution_domain_prefix_graph(
// GENERIC-NOT: intent.fast_path_id
// GENERIC-NOT: intent.backend.execution_domain_assumption
// GENERIC: gpu.launch
// GENERIC: gpu.dynamic_shared_memory
// GENERIC: gpu.shuffle up
// GENERIC: gpu.terminator
func.func @execution_domain_prefix_graph(
    %x: tensor<2x128xf32>, %sorted: tensor<2x128xf32>)
    -> tensor<2x128xf32> {
  %cssv = intent.scan %sorted {fn = "sum", axis = 1}
      : (tensor<2x128xf32>) -> tensor<2x128xf32>
  %offs = intent.iota : tensor<128xi32>
  %one_i32 = arith.constant dense<1> : tensor<i32>
  %rank_i32 = intent.map_binary %offs, %one_i32 {fn = "add"}
      : (tensor<128xi32>, tensor<i32>) -> tensor<128xi32>
  %rank = intent.cast %rank_i32 {to_dtype = "f32"}
      : (tensor<128xi32>) -> tensor<128xf32>
  %one = arith.constant dense<1.0> : tensor<f32>
  %cm1 = intent.map_binary %cssv, %one {fn = "sub"}
      : (tensor<2x128xf32>, tensor<f32>) -> tensor<2x128xf32>
  %threshold = intent.map_binary %cm1, %rank {fn = "div"}
      : (tensor<2x128xf32>, tensor<128xf32>) -> tensor<2x128xf32>
  %support = intent.map_binary %sorted, %threshold {fn = "gt"}
      : (tensor<2x128xf32>, tensor<2x128xf32>) -> tensor<2x128xi1>
  %support_i32 = intent.cast %support {to_dtype = "i32"}
      : (tensor<2x128xi1>) -> tensor<2x128xi32>
  %k0 = intent.reduce %support_i32 {fn = "sum", axes = [1], assoc = "associative"}
      : (tensor<2x128xi32>) -> tensor<2x1xi32>
  %k1 = intent.map_binary %k0, %one_i32 {fn = "max"}
      : (tensor<2x1xi32>, tensor<i32>) -> tensor<2x1xi32>
  %k = intent.cast %k1 {to_dtype = "f32"}
      : (tensor<2x1xi32>) -> tensor<2x1xf32>
  %zero = arith.constant dense<0.0> : tensor<f32>
  %supported = intent.select %support, %sorted, %zero
      : (tensor<2x128xi1>, tensor<2x128xf32>, tensor<f32>) -> tensor<2x128xf32>
  %s = intent.reduce %supported {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x128xf32>) -> tensor<2x1xf32>
  %sm1 = intent.map_binary %s, %one {fn = "sub"}
      : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %tau = intent.map_binary %sm1, %k {fn = "div"}
      : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %shifted = intent.map_binary %x, %tau {fn = "sub"}
      : (tensor<2x128xf32>, tensor<2x1xf32>) -> tensor<2x128xf32>
  %out = intent.map_binary %shifted, %zero {fn = "max"}
      : (tensor<2x128xf32>, tensor<f32>) -> tensor<2x128xf32>
  return %out : tensor<2x128xf32>
}
