// RUN: intent-opt -convert-intent-elementwise-to-gpu='block-threads=64 row-vector-width=1' %s | FileCheck %s --check-prefix=CANONICAL
// RUN: intent-opt -convert-intent-elementwise-to-gpu='block-threads=64 row-mask-zero-fill=true flatten-contiguous=true flat-vector-width=4' %s | FileCheck %s --check-prefix=ZEROFILL

// Rank-1 [M] is a typed row broadcast.  The canonical path retains cast+mul;
// the Full-only path uses one uniform row predicate per vector packet and avoids
// loading X on false rows.

// CANONICAL-LABEL: func.func @rowmask(
// CANONICAL-SAME: memref<16x64xf32>
// CANONICAL-SAME: memref<16xi1>
// CANONICAL: memref.load {{.*}} : memref<16x64xf32>
// CANONICAL: memref.load {{.*}} : memref<16xi1>
// CANONICAL: arith.uitofp
// CANONICAL: arith.mulf
// CANONICAL-NOT: intent.contraction

// ZEROFILL-LABEL: func.func @rowmask(
// ZEROFILL-SAME: memref<256xvector<4xf32>>
// ZEROFILL-SAME: memref<16xi1>
// ZEROFILL-SAME: memref<256xvector<4xf32>>
// ZEROFILL-SAME: intent.backend.transformation_trace = ["cuda.elementwise.row.vectorized.v1", "cuda.elementwise.rowmask.zero-fill.v1", "cuda.elementwise.rowmask.flat-grid.v1"]
// ZEROFILL: scf.for
// ZEROFILL: memref.load {{.*}} : memref<16xi1>
// ZEROFILL: scf.if
// ZEROFILL: memref.load {{.*}} : memref<256xvector<4xf32>>
// ZEROFILL: vector.splat
// ZEROFILL-NOT: arith.mulf
// ZEROFILL-NOT: intent.contraction
func.func @rowmask(%data: tensor<16x64xf32>, %mask: tensor<16xi1>)
    -> tensor<16x64xf32> {
  %mask_f32 = intent.cast %mask {to_dtype = "f32"}
      : (tensor<16xi1>) -> tensor<16xf32>
  %result = intent.contraction %data, %mask_f32 {subscripts = "ab,a->ab"}
      : (tensor<16x64xf32>, tensor<16xf32>) -> tensor<16x64xf32>
  return %result : tensor<16x64xf32>
}
