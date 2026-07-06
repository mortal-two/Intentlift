// RUN: intent-opt -convert-intent-compact-to-gpu=block-threads=4 %s | FileCheck %s

// C8 dynamic-shape output: intent.nonzero / masked_select / masked_scatter
// lowered by `convert-intent-compact-to-gpu` to a MULTI-BLOCK BLOCK-COOPERATIVE
// stream compaction (the CUB DeviceSelect::Flagged pattern) — THREE gpu.launches
// sharing a device scratch memref<[B]xi64>:
//   (1) count   grid(B) block(T): the T threads COALESCED-load their chunk's
//               flags (thread tx handles flat idx lo+kk*T+tx over numTiles
//               sub-tiles), a block Hillis-Steele scan makes sh[T-1] the chunk
//               total -> scratch[b];
//   (2) scan    grid(1) block(Ts): two-level exclusive-scan of scratch in place
//               (scratch[b] <- global base slot) + total nnz to the count buffer;
//   (3) scatter grid(B) block(T): the chunk is walked ASCENDING in T-wide tiles;
//               a per-tile block scan of the 0/1 predicate gives each true thread
//               its exclusive-prefix local slot, plus a running block counter =>
//               the write index is the EXACT count of trues before this element.
// Contiguous chunks in block order + ascending tiles + ascending threads => EXACT
// row-major order (matches torch.nonzero / masked_select), now coalesced. Here
// Ntotal=65536, B=8192, chunk=8, T=4, numTiles=2 (block-threads=4 caps T).

// ---------------------------------------------------------------------------- //
// nonzero: cond -> [65536,2] i64 index rows (MAX-sized) + [1] i64 count.
// CHECK-LABEL: func.func @nz(
// CHECK-SAME:    %{{.*}}: memref<256x256xf32>, %{{.*}}: memref<65536x2xi64>, %{{.*}}: memref<1xi64>)
// CHECK:         memref.alloc() : memref<8192xi64>
// CHECK-NOT:     gpu.all_reduce
// count phase: coalesced flag loads + block scan + total to scratch
// CHECK:         gpu.launch
// CHECK:           memref.load %{{.*}} : memref<256x256xf32>
// CHECK:           gpu.dynamic_shared_memory
// CHECK:           memref.store %{{.*}} : memref<4xi64, #gpu.address_space<workgroup>>
// CHECK:           gpu.barrier
// CHECK:           memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<8192xi64>
// CHECK:           gpu.terminator
// scan phase
// CHECK:         gpu.launch
// CHECK:           scf.for {{.*}} iter_args
// CHECK:           memref.store {{.*}} : memref<1xi64>
// CHECK:           gpu.terminator
// scatter phase: base load + coalesced predicate scan + cooperative store
// CHECK:         gpu.launch
// CHECK:           memref.load %{{.*}} : memref<8192xi64>
// CHECK:           gpu.dynamic_shared_memory
// CHECK:           gpu.barrier
// CHECK:           scf.if
// CHECK:             memref.store
// CHECK:           gpu.terminator
func.func @nz(%cond: tensor<256x256xf32>) -> tensor<?x2xi64> {
  %r = intent.nonzero %cond : (tensor<256x256xf32>) -> tensor<?x2xi64>
  return %r : tensor<?x2xi64>
}

// -----

// masked_select: data[mask] -> [nnz] (MAX 65536) + count. sel[slot] = data[coord].
// CHECK-LABEL: func.func @msel(
// CHECK-SAME:    %{{.*}}: memref<256x256xf32>, %{{.*}}: memref<256x256xf32>, %{{.*}}: memref<65536xf32>, %{{.*}}: memref<1xi64>)
// CHECK:         memref.alloc() : memref<8192xi64>
// CHECK:         gpu.launch
// CHECK:         gpu.launch
// CHECK:         gpu.launch
// CHECK:           gpu.dynamic_shared_memory
// CHECK:           scf.if
// CHECK:             memref.load
// CHECK:             memref.store
// CHECK:           gpu.terminator
func.func @msel(%data: tensor<256x256xf32>, %mask: tensor<256x256xf32>) -> tensor<?xf32> {
  %r = intent.masked_select %data, %mask : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// masked_scatter: data-shaped [256,256] output (NO count). out[coord] = mask ?
// src[slot] : data[coord] — blocks write disjoint coords (race-free else-copy).
// CHECK-LABEL: func.func @mscat(
// CHECK-SAME:    %{{.*}}: memref<256x256xf32>, %{{.*}}: memref<256x256xf32>, %{{.*}}: memref<65536xf32>, %{{.*}}: memref<256x256xf32>)
// CHECK:         memref.alloc() : memref<8192xi64>
// CHECK:         gpu.launch
// CHECK:         gpu.launch
// CHECK:         gpu.launch
// CHECK:           scf.if
// CHECK:             memref.load
// CHECK:             memref.store
// CHECK:           } else {
// CHECK:             memref.load
// CHECK:             memref.store
// CHECK:           gpu.terminator
func.func @mscat(%data: tensor<256x256xf32>, %mask: tensor<256x256xf32>, %src: tensor<65536xf32>) -> tensor<256x256xf32> {
  %r = intent.masked_scatter %data, %mask, %src : (tensor<256x256xf32>, tensor<256x256xf32>, tensor<65536xf32>) -> tensor<256x256xf32>
  return %r : tensor<256x256xf32>
}
