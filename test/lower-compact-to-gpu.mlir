// RUN: intent-opt -convert-intent-compact-to-gpu=block-threads=4 %s | FileCheck %s

// C8 dynamic-shape output: intent.nonzero / masked_select / masked_scatter
// lowered by `convert-intent-compact-to-gpu` to ONE block-cooperative prefix-sum
// gpu.launch (grid (1,1,1), block (T,1,1)). The mechanism is an exclusive
// prefix-sum of the flattened 0/1 mask: T contiguous segments, each thread counts
// the trues in all prior segments (its base slot), then re-scans its own segment
// emitting each true element at the running slot (EXACT row-major order). The
// runtime nnz extent is resolved host-side; the kernel writes at most nnz slots
// and the last thread stores the total to a `count` buffer. T=4.

// ---------------------------------------------------------------------------- //
// nonzero: [4,4] cond -> [16,2] i64 index rows (MAX-sized) + [1] i64 count. Each
// true flat index i unravels to (i/4, i%4) and stores at out[slot, 0..1].
// CHECK-LABEL: func.func @nz(
// CHECK-SAME:    %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<16x2xi64>, %{{.*}}: memref<1xi64>)
// CHECK:         gpu.launch
// CHECK-NOT:     gpu.all_reduce
// CHECK:         scf.for {{.*}} iter_args
// CHECK:         scf.for {{.*}} iter_args
// CHECK:           memref.load
// CHECK:           scf.if
// CHECK:             memref.store
// CHECK:         gpu.terminator
func.func @nz(%cond: tensor<4x4xf32>) -> tensor<?x2xi64> {
  %r = intent.nonzero %cond : (tensor<4x4xf32>) -> tensor<?x2xi64>
  return %r : tensor<?x2xi64>
}

// -----

// masked_select: data[mask] -> [nnz] (MAX 16) + count. sel[slot] = data[coord].
// CHECK-LABEL: func.func @msel(
// CHECK-SAME:    %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<16xf32>, %{{.*}}: memref<1xi64>)
// CHECK:         gpu.launch
// CHECK:         scf.for
// CHECK:         scf.for {{.*}} iter_args
// CHECK:           scf.if
// CHECK:             memref.load
// CHECK:             memref.store
// CHECK:         gpu.terminator
func.func @msel(%data: tensor<4x4xf32>, %mask: tensor<4x4xf32>) -> tensor<?xf32> {
  %r = intent.masked_select %data, %mask : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// masked_scatter: data-shaped [4,4] output (NO count). out[coord] = mask ?
// src[slot] : data[coord] — the prefix-sum indexes src.
// CHECK-LABEL: func.func @mscat(
// CHECK-SAME:    %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<16xf32>, %{{.*}}: memref<4x4xf32>)
// CHECK:         gpu.launch
// CHECK:         scf.for {{.*}} iter_args
// CHECK:           scf.if
// CHECK:             memref.load
// CHECK:             memref.store
// CHECK:           } else {
// CHECK:             memref.load
// CHECK:             memref.store
// CHECK:         gpu.terminator
func.func @mscat(%data: tensor<4x4xf32>, %mask: tensor<4x4xf32>, %src: tensor<16xf32>) -> tensor<4x4xf32> {
  %r = intent.masked_scatter %data, %mask, %src : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<16xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
