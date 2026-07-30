// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=512 tile-cols=16 vector-width=4 rows-per-block=2 prefix-threads=32 cross-multiply-support-predicate=true allow-execution-domain-sorted-prefix=true require-sorted-prefix=true' %s | FileCheck %s
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=256 tile-cols=32 vector-width=4 rows-per-block=1 prefix-threads=32 replicate-prefix-per-warp=true cross-multiply-support-predicate=true allow-execution-domain-sorted-prefix=true require-sorted-prefix=true' %s | FileCheck %s --check-prefix=REPLICATED
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=512 tile-cols=16 vector-width=16 rows-per-block=1 prefix-threads=32 replicate-prefix-per-warp=true allow-execution-domain-sorted-prefix=true require-sorted-prefix=true' %s | FileCheck %s --check-prefix=INVALID

// The prefix/support work is owned by one warp per row while 512 logical
// threads per row write the full epilogue. Two rows share the 1024-thread CTA.
// The support comparison uses the explicit Q/D-gated cross-multiplied numeric
// state; tau itself retains the authored division.
// CHECK-LABEL: func.func @wide_prefix_graph(
// CHECK-SAME: memref<2x1024xvector<4xf32>>
// CHECK-SAME: intent.backend.block_threads = 1024 : i64
// CHECK-SAME: intent.backend.reduction_topology = "sorted_prefix_multirow_warp_scan_wide_epilogue_cross_multiply"
// CHECK-SAME: intent.backend.rows_per_block = 2 : i64
// CHECK-SAME: intent.fast_path_id = "cuda.row_scan.sparsemax.sorted_prefix.execution_domain.v1"
// CHECK: %[[GRID:.*]] = arith.constant 1 : index
// CHECK: %[[BLOCK:.*]] = arith.constant 1024 : index
// CHECK: gpu.launch{{.*}}%[[GRID]]{{.*}}%[[BLOCK]]
// CHECK: gpu.dynamic_shared_memory
// CHECK: gpu.shuffle up
// CHECK: arith.mulf
// CHECK: arith.cmpf ogt
// CHECK: gpu.shuffle xor
// CHECK: arith.divf
// CHECK-COUNT-1: gpu.barrier
// CHECK: arith.maxnumf
// CHECK: gpu.terminator
// REPLICATED-LABEL: func.func @wide_prefix_graph(
// REPLICATED-SAME: memref<2x1024xvector<4xf32>>
// REPLICATED-SAME: intent.backend.block_threads = 256 : i64
// REPLICATED-SAME: intent.backend.reduction_topology = "sorted_prefix_replicated_warp_cross_multiply"
// REPLICATED: %[[BLOCK:.*]] = arith.constant 256 : index
// REPLICATED: gpu.launch{{.*}}%[[BLOCK]]
// REPLICATED: gpu.shuffle up
// REPLICATED: arith.mulf
// REPLICATED: gpu.shuffle xor
// REPLICATED: arith.divf
// REPLICATED: arith.maxnumf
// REPLICATED-NOT: gpu.dynamic_shared_memory
// REPLICATED-NOT: gpu.barrier
// REPLICATED: gpu.terminator
// INVALID-LABEL: func.func @wide_prefix_graph(
// INVALID-NOT: gpu.launch
// INVALID: intent.scan
func.func @wide_prefix_graph(
    %x: tensor<2x4096xf32>, %sorted: tensor<2x4096xf32>)
    -> tensor<2x4096xf32> {
  %cssv = intent.scan %sorted {fn = "sum", axis = 1}
      : (tensor<2x4096xf32>) -> tensor<2x4096xf32>
  %offs = intent.iota : tensor<4096xi32>
  %one_i32 = arith.constant dense<1> : tensor<i32>
  %rank_i32 = intent.map_binary %offs, %one_i32 {fn = "add"}
      : (tensor<4096xi32>, tensor<i32>) -> tensor<4096xi32>
  %rank = intent.cast %rank_i32 {to_dtype = "f32"}
      : (tensor<4096xi32>) -> tensor<4096xf32>
  %one = arith.constant dense<1.0> : tensor<f32>
  %cm1 = intent.map_binary %cssv, %one {fn = "sub"}
      : (tensor<2x4096xf32>, tensor<f32>) -> tensor<2x4096xf32>
  %threshold = intent.map_binary %cm1, %rank {fn = "div"}
      : (tensor<2x4096xf32>, tensor<4096xf32>) -> tensor<2x4096xf32>
  %support = intent.map_binary %sorted, %threshold {fn = "gt"}
      : (tensor<2x4096xf32>, tensor<2x4096xf32>) -> tensor<2x4096xi1>
  %support_i32 = intent.cast %support {to_dtype = "i32"}
      : (tensor<2x4096xi1>) -> tensor<2x4096xi32>
  %k0 = intent.reduce %support_i32 {fn = "sum", axes = [1], assoc = "associative"}
      : (tensor<2x4096xi32>) -> tensor<2x1xi32>
  %k1 = intent.map_binary %k0, %one_i32 {fn = "max"}
      : (tensor<2x1xi32>, tensor<i32>) -> tensor<2x1xi32>
  %k = intent.cast %k1 {to_dtype = "f32"}
      : (tensor<2x1xi32>) -> tensor<2x1xf32>
  %zero = arith.constant dense<0.0> : tensor<f32>
  %supported = intent.select %support, %sorted, %zero
      : (tensor<2x4096xi1>, tensor<2x4096xf32>, tensor<f32>) -> tensor<2x4096xf32>
  %s = intent.reduce %supported {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x4096xf32>) -> tensor<2x1xf32>
  %sm1 = intent.map_binary %s, %one {fn = "sub"}
      : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %tau = intent.map_binary %sm1, %k {fn = "div"}
      : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %shifted = intent.map_binary %x, %tau {fn = "sub"}
      : (tensor<2x4096xf32>, tensor<2x1xf32>) -> tensor<2x4096xf32>
  %out = intent.map_binary %shifted, %zero {fn = "max"}
      : (tensor<2x4096xf32>, tensor<f32>) -> tensor<2x4096xf32>
  return %out : tensor<2x4096xf32>
}
