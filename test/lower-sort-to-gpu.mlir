// RUN: intent-opt -convert-intent-sort-to-gpu=block-threads=4 %s | FileCheck %s
// RUN: env RADIX_MIN_N=0 intent-opt -convert-intent-sort-to-gpu=block-threads=16 %s | FileCheck %s --check-prefix=RADIX

// C6 HARD-tier order statistics: intent.sort (values + argsort indices) over a
// [M,N] row (axis 1). Two single-block-per-row lowerings are cost-model-
// selected: the BLOCK BITONIC network (default here; wins at/near powers of two)
// and, forced via RADIX_MIN_N for the second RUN, the CUB-style WARP-MATCH
// REGISTER-RANKING LSD RADIX sort (wins large rows AND the power-of-two GATE).
// The radix path flips f32 to an order-preserving unsigned key (arith.shrsi
// sign-fill, xor 0x80000000), then 8 LSD passes of 4-bit-digit (andi 15) ranking
// done ENTIRELY in registers: nvvm.vote.ballot.sync builds each key's warp
// peer-mask (one ballot per digit bit), math.ctpop ranks it, and a tiny cross-
// warp table replaces the old shared histogram -> bandwidth-bound scatter.
//
// RADIX-LABEL: func.func @sort_row(
// RADIX:         gpu.launch
// RADIX:         arith.bitcast %{{.*}} : f32 to i32
// RADIX:         arith.shrsi
// RADIX:         arith.xori
// RADIX:         arith.shrui
// RADIX:         arith.andi %{{.*}}, %c15_i32
// RADIX:         nvvm.vote.ballot.sync
// RADIX:         math.ctpop
// RADIX:         nvvm.bar.warp.sync
// RADIX-NOT:     intent.

// C6 HARD-tier order statistics: intent.sort (values + argsort indices) over a
// [M,N] row (axis 1) -> ONE block-cooperative BLOCK BITONIC SORT gpu.launch.
// One block per row loads the row into a [P]-element dynamic-shared value buffer
// (P = next_pow2(N)), padding N..P-1 with a +inf (ascending) / -inf (descending)
// SENTINEL so the pad sorts to the END, then runs the bitonic network:
// log2(P)*(log2(P)+1)/2 barrier-separated compare-exchange stages, each a
// gpu.barrier then a paired compare-swap of partners i and i^j under the
// direction bit (i & k)==0. Finally a barrier + write the first W entries out.
// N=8 -> P=8 -> log2(8)*(log2(8)+1)/2 = 6 network stages (+ load & final
// barriers = 7 gpu.barrier total).

// A pure sort returns only values: input + one f32 output memref, values-only
// shared buffer (8*4 = 32 bytes), ascending direction bit = (i & k)==0.
// CHECK-LABEL: func.func @sort_row(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         memref.view %{{.*}} : {{.*}} to memref<8xf32
// CHECK:         arith.constant 0x7F800000 : f32
// The load phase: sentinel-pad the row into shared.
// CHECK:         scf.if
// CHECK:           memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// A network stage: barrier then the paired compare-swap on i^j.
// CHECK:         gpu.barrier
// CHECK:         arith.xori
// CHECK:         arith.cmpi ugt
// CHECK:         arith.andi
// CHECK:         arith.cmpi eq
// CHECK:         arith.cmpf ogt
// CHECK:         arith.cmpf olt
// CHECK:         arith.select
// CHECK:         scf.if
// CHECK:           memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<8xf32
// The write-out.
// CHECK:         gpu.barrier
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:     intent.
func.func @sort_row(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %v, %i = intent.sort %x {descending = false, axis = 1 : i64} : (tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xi32>)
  return %v : tensor<4x8xf32>
}

// argsort returns only the i32 index permutation: input + one i32 output, and a
// SECOND dynamic-shared i32 index buffer swapped in lockstep. descending flips
// the direction test to (i & k) != 0.
// CHECK-LABEL: func.func @argsort_row(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xi32>)
// CHECK:         memref.view %{{.*}} : {{.*}} to memref<8xf32
// CHECK:         memref.view %{{.*}} : {{.*}} to memref<8xi32
// CHECK:         arith.constant 0xFF800000 : f32
// CHECK:         arith.cmpi ne
// The index buffer is swapped alongside the values.
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<8xi32
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x8xi32>
// CHECK-NOT:     intent.
func.func @argsort_row(%x: tensor<4x8xf32>) -> tensor<4x8xi32> {
  %v, %i = intent.sort %x {descending = true, axis = 1 : i64} : (tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xi32>)
  return %i : tensor<4x8xi32>
}

// topk = sort with a NARROWER output extent (W=3 < N=8): both values and indices
// returned -> three memref args (input + [M,3] f32 + [M,3] i32); the network
// still sorts the full P=8, only the first W=3 are written out.
// CHECK-LABEL: func.func @topk_row(
// CHECK-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x3xf32>, %{{.*}}: memref<4x3xi32>)
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x3xf32>
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x3xi32>
// CHECK-NOT:     intent.
func.func @topk_row(%x: tensor<4x8xf32>) -> (tensor<4x3xf32>, tensor<4x3xi32>) {
  %v, %i = intent.sort %x {descending = true, axis = 1 : i64} : (tensor<4x8xf32>) -> (tensor<4x3xf32>, tensor<4x3xi32>)
  return %v, %i : tensor<4x3xf32>, tensor<4x3xi32>
}
