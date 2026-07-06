// RUN: intent-opt -convert-intent-compact-to-gpu=block-threads=4 %s | FileCheck %s

// C8 dynamic-shape output: intent.nonzero / masked_select / masked_scatter
// lowered by `convert-intent-compact-to-gpu` to a MULTI-BLOCK parallel stream
// compaction — THREE gpu.launches sharing a device scratch memref<[B]xi64>:
//   (1) count   grid(B) block(1): block b counts trues in its contiguous chunk
//               [b*chunk,(b+1)*chunk) -> scratch[b];
//   (2) scan    grid(1) block(1): a single thread exclusive-scans scratch in
//               place (scratch[b] <- global base slot) and writes total nnz to
//               the count buffer;
//   (3) scatter grid(B) block(1): block b emits its chunk's true elements at
//               scratch[b] + local running prefix.
// Contiguous chunks in block order + ascending within => EXACT row-major order
// (matches torch.nonzero / torch.masked_select), now parallel across all SMs
// (the old grid(1,1,1) single block serialized the whole compaction). The
// runtime nnz extent is resolved host-side (slice out[:nnz]). Here B=16 (Ntotal).

// ---------------------------------------------------------------------------- //
// nonzero: [4,4] cond -> [16,2] i64 index rows (MAX-sized) + [1] i64 count. Each
// true flat index i unravels to (i/4, i%4) and stores at out[slot, 0..1].
// CHECK-LABEL: func.func @nz(
// CHECK-SAME:    %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<16x2xi64>, %{{.*}}: memref<1xi64>)
// CHECK:         memref.alloc() : memref<16xi64>
// CHECK-NOT:     gpu.all_reduce
// count phase
// CHECK:         gpu.launch
// CHECK:           scf.for {{.*}} iter_args
// CHECK:             memref.load
// CHECK:           memref.store {{.*}} : memref<16xi64>
// CHECK:           gpu.terminator
// scan phase
// CHECK:         gpu.launch
// CHECK:           scf.for {{.*}} iter_args
// CHECK:             memref.load {{.*}} : memref<16xi64>
// CHECK:             memref.store {{.*}} : memref<16xi64>
// CHECK:           memref.store {{.*}} : memref<1xi64>
// CHECK:           gpu.terminator
// scatter phase
// CHECK:         gpu.launch
// CHECK:           memref.load {{.*}} : memref<16xi64>
// CHECK:           scf.for {{.*}} iter_args
// CHECK:             scf.if
// CHECK:               memref.store
// CHECK:           gpu.terminator
func.func @nz(%cond: tensor<4x4xf32>) -> tensor<?x2xi64> {
  %r = intent.nonzero %cond : (tensor<4x4xf32>) -> tensor<?x2xi64>
  return %r : tensor<?x2xi64>
}

// -----

// masked_select: data[mask] -> [nnz] (MAX 16) + count. sel[slot] = data[coord].
// CHECK-LABEL: func.func @msel(
// CHECK-SAME:    %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<16xf32>, %{{.*}}: memref<1xi64>)
// CHECK:         memref.alloc() : memref<16xi64>
// CHECK:         gpu.launch
// CHECK:         gpu.launch
// CHECK:         gpu.launch
// CHECK:           scf.for {{.*}} iter_args
// CHECK:             scf.if
// CHECK:               memref.load
// CHECK:               memref.store
// CHECK:           gpu.terminator
func.func @msel(%data: tensor<4x4xf32>, %mask: tensor<4x4xf32>) -> tensor<?xf32> {
  %r = intent.masked_select %data, %mask : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// masked_scatter: data-shaped [4,4] output (NO count). out[coord] = mask ?
// src[slot] : data[coord] — the prefix-sum indexes src. Still 3-phase (the scan
// gives each block its base offset into src); blocks write disjoint coords.
// CHECK-LABEL: func.func @mscat(
// CHECK-SAME:    %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<16xf32>, %{{.*}}: memref<4x4xf32>)
// CHECK:         memref.alloc() : memref<16xi64>
// CHECK:         gpu.launch
// CHECK:         gpu.launch
// CHECK:         gpu.launch
// CHECK:           scf.for {{.*}} iter_args
// CHECK:             scf.if
// CHECK:               memref.load
// CHECK:               memref.store
// CHECK:             } else {
// CHECK:               memref.load
// CHECK:               memref.store
// CHECK:           gpu.terminator
func.func @mscat(%data: tensor<4x4xf32>, %mask: tensor<4x4xf32>, %src: tensor<16xf32>) -> tensor<4x4xf32> {
  %r = intent.masked_scatter %data, %mask, %src : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<16xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
