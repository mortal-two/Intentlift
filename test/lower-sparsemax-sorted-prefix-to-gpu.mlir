// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=32 require-sorted-prefix=true' %s | FileCheck %s
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=32 require-sorted-prefix=true frontier-ballot-support=true' %s | FileCheck %s --check-prefix=BALLOT
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=32 enable-sorted-prefix=false' %s | FileCheck %s --check-prefix=GENERIC
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=32 prefix-packet-width=2 require-sorted-prefix=true' %s | FileCheck %s --check-prefix=PACKET2
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=32 prefix-packet-width=1 require-sorted-prefix=true' %s | FileCheck %s --check-prefix=PACKET1
// RUN: intent-opt '-convert-intent-row-reduction-to-gpu=block-threads=64 prefix-packet-width=1 require-sorted-prefix=true frontier-ballot-support=true' %s | FileCheck %s --check-prefix=HIERBALLOT

// The specialized path is legal only because P4 materialized the validated
// Layer-B descending-order precondition.  Matching is graph-structural: the
// symbol deliberately does not contain "sparsemax".
// CHECK-LABEL: func.func @certified_prefix_graph(
// CHECK-SAME: memref<2x64xvector<4xf32>>
// CHECK-SAME: memref<2x64xvector<4xf32>>
// CHECK-SAME: memref<2x64xvector<4xf32>>
// CHECK-SAME: intent.fast_path_id = "cuda.row_scan.sparsemax.sorted_prefix.v1"
// CHECK-SAME: intent_sorted_descending = true
// CHECK: gpu.launch
// CHECK: scf.for
// CHECK: memref.load {{.*}} : memref<2x64xvector<4xf32>>
// CHECK: gpu.shuffle up
// CHECK: arith.cmpf ogt
// CHECK: gpu.shuffle xor
// CHECK: arith.cmpi sgt
// CHECK: arith.maxnumf
// CHECK: memref.store {{.*}} : memref<2x64xvector<4xf32>>
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
// HIERBALLOT-LABEL: func.func @certified_prefix_graph(
// HIERBALLOT-SAME: intent.backend.reduction_topology = "sorted_prefix_early_stop_frontier_ballot"
// HIERBALLOT: gpu.dynamic_shared_memory
// HIERBALLOT-NOT: gpu.shuffle xor
// HIERBALLOT: nvvm.vote.ballot.sync
// HIERBALLOT: math.ctlz
// HIERBALLOT-NOT: gpu.shuffle xor
// HIERBALLOT: nvvm.vote.ballot.sync
// HIERBALLOT: math.ctlz
// HIERBALLOT-NOT: gpu.shuffle xor
// HIERBALLOT: gpu.terminator
// HIERBALLOT-NOT: intent.
// PACKET2-LABEL: func.func @certified_prefix_graph(
// PACKET2-SAME: %{{[^:]+}}: memref<2x64xvector<4xf32>>
// PACKET2-SAME: %{{[^:]+}}: memref<2x128xvector<2xf32>>
// PACKET2-SAME: %{{[^:]+}}: memref<2x64xvector<4xf32>>
// PACKET2-SAME: intent.backend.prefix_packet_width = 2 : i64
// PACKET2: memref.load {{.*}} : memref<2x128xvector<2xf32>>
// PACKET2: memref.store {{.*}} : memref<2x64xvector<4xf32>>
// PACKET1-LABEL: func.func @certified_prefix_graph(
// PACKET1-SAME: %{{[^:]+}}: memref<2x64xvector<4xf32>>
// PACKET1-SAME: %{{[^:]+}}: memref<2x256xf32>
// PACKET1-SAME: %{{[^:]+}}: memref<2x64xvector<4xf32>>
// PACKET1-SAME: intent.backend.prefix_packet_width = 1 : i64
// PACKET1: memref.load {{.*}} : memref<2x256xf32>
// PACKET1: memref.store {{.*}} : memref<2x64xvector<4xf32>>
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
    %x: tensor<2x256xf32>, %sorted: tensor<2x256xf32>)
    -> tensor<2x256xf32>
    attributes {intent_sorted_descending = true} {
  %cssv = intent.scan %sorted {fn = "sum", axis = 1}
      : (tensor<2x256xf32>) -> tensor<2x256xf32>
  %offs = intent.iota : tensor<256xi32>
  %one_i32 = arith.constant dense<1> : tensor<i32>
  %rank_i32 = intent.map_binary %offs, %one_i32 {fn = "add"}
      : (tensor<256xi32>, tensor<i32>) -> tensor<256xi32>
  %rank = intent.cast %rank_i32 {to_dtype = "f32"}
      : (tensor<256xi32>) -> tensor<256xf32>
  %one = arith.constant dense<1.0> : tensor<f32>
  %cm1 = intent.map_binary %cssv, %one {fn = "sub"}
      : (tensor<2x256xf32>, tensor<f32>) -> tensor<2x256xf32>
  %threshold = intent.map_binary %cm1, %rank {fn = "div"}
      : (tensor<2x256xf32>, tensor<256xf32>) -> tensor<2x256xf32>
  %support = intent.map_binary %sorted, %threshold {fn = "gt"}
      : (tensor<2x256xf32>, tensor<2x256xf32>) -> tensor<2x256xi1>
  %support_i32 = intent.cast %support {to_dtype = "i32"}
      : (tensor<2x256xi1>) -> tensor<2x256xi32>
  %k0 = intent.reduce %support_i32 {fn = "sum", axes = [1], assoc = "associative"}
      : (tensor<2x256xi32>) -> tensor<2x1xi32>
  %k1 = intent.map_binary %k0, %one_i32 {fn = "max"}
      : (tensor<2x1xi32>, tensor<i32>) -> tensor<2x1xi32>
  %k = intent.cast %k1 {to_dtype = "f32"}
      : (tensor<2x1xi32>) -> tensor<2x1xf32>
  %zero = arith.constant dense<0.0> : tensor<f32>
  %supported = intent.select %support, %sorted, %zero
      : (tensor<2x256xi1>, tensor<2x256xf32>, tensor<f32>) -> tensor<2x256xf32>
  %s = intent.reduce %supported {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x256xf32>) -> tensor<2x1xf32>
  %sm1 = intent.map_binary %s, %one {fn = "sub"}
      : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %tau = intent.map_binary %sm1, %k {fn = "div"}
      : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %shifted = intent.map_binary %x, %tau {fn = "sub"}
      : (tensor<2x256xf32>, tensor<2x1xf32>) -> tensor<2x256xf32>
  %out = intent.map_binary %shifted, %zero {fn = "max"}
      : (tensor<2x256xf32>, tensor<f32>) -> tensor<2x256xf32>
  return %out : tensor<2x256xf32>
}
