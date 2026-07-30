// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=32 require-sorted-prefix=true' %s | FileCheck %s
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=32 require-sorted-prefix=true frontier-ballot-support=true' %s | FileCheck %s --check-prefix=BALLOT
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=32 enable-sorted-prefix=false' %s | FileCheck %s --check-prefix=GENERIC

// The specialized path is legal only because P4 materialized the validated
// Layer-B descending-order precondition.  Matching is graph-structural: the
// symbol deliberately does not contain "sparsemax".
// CHECK-LABEL: func.func @certified_prefix_graph(
// CHECK-SAME: memref<2x32xvector<4xf32>>
// CHECK-SAME: memref<2x32xvector<4xf32>>
// CHECK-SAME: memref<2x32xvector<4xf32>>
// CHECK-SAME: intent.fast_path_id = "cuda.row_scan.sparsemax.sorted_prefix.v1"
// CHECK-SAME: intent_sorted_descending = true
// CHECK: gpu.launch
// CHECK: scf.for
// CHECK: memref.load {{.*}} : memref<2x32xvector<4xf32>>
// CHECK: gpu.shuffle up
// CHECK: arith.cmpf ogt
// CHECK: gpu.shuffle xor
// CHECK: arith.cmpi sgt
// CHECK: arith.maxnumf
// CHECK: memref.store {{.*}} : memref<2x32xvector<4xf32>>
// CHECK: gpu.terminator
// CHECK-NOT: gpu.dynamic_shared_memory
// CHECK-NOT: gpu.barrier
// CHECK-NOT: intent.
// BALLOT-LABEL: func.func @certified_prefix_graph(
// BALLOT-SAME: intent.backend.reduction_topology = "sorted_prefix_early_stop_frontier_ballot"
// BALLOT: gpu.launch
// BALLOT: gpu.shuffle up
// BALLOT: nvvm.vote.ballot.sync
// BALLOT: math.ctlz
// BALLOT: gpu.shuffle idx
// BALLOT-NOT: gpu.shuffle xor
// BALLOT: memref.store
// BALLOT: gpu.terminator
// BALLOT-NOT: intent.
// GENERIC-LABEL: func.func @certified_prefix_graph(
// GENERIC-NOT: intent.fast_path_id
// GENERIC: gpu.launch
// GENERIC: gpu.dynamic_shared_memory
// GENERIC: gpu.shuffle up
// GENERIC: arith.cmpf ogt
// GENERIC: memref.store
// GENERIC: gpu.terminator
// GENERIC-NOT: intent.
func.func @certified_prefix_graph(
    %x: tensor<2x128xf32>, %sorted: tensor<2x128xf32>)
    -> tensor<2x128xf32>
    attributes {intent_sorted_descending = true} {
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
