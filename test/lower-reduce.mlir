// RUN: intent-opt -convert-intent-to-linalg %s | FileCheck %s

// reduce{max} over axis 1 with keepdims (result rank == input rank): a
// linalg.generic reduction into a rank-1 temp, then tensor.expand_shape back
// to [4x1] so downstream row-broadcasts line up.
// CHECK-LABEL: func.func @rmax
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK: arith.maxnumf
// CHECK: tensor.expand_shape
func.func @rmax(%x: tensor<4x8xf32>) -> tensor<4x1xf32> {
  %m = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  return %m : tensor<4x1xf32>
}

// -----

// reduce{sum} over axis 1, no keepdims (result rank < input rank): no expand.
// CHECK-LABEL: func.func @rsum
// CHECK: arith.addf
// CHECK-NOT: tensor.expand_shape
func.func @rsum(%x: tensor<4x8xf32>) -> tensor<4xf32> {
  %s = intent.reduce %x {fn = "sum", axes = [1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<4xf32>
  return %s : tensor<4xf32>
}

// -----

// A reduction over an early source axis is reordered to parallel-first loop
// space.  This prevents an outer host scf.for around gpu.launch after the
// generic CUDA pipeline; N remains the inner in-kernel reduction iterator.
// CHECK-LABEL: func.func @rmax_axis0_rank4
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK: arith.maxnumf
func.func @rmax_axis0_rank4(%x: tensor<128x1x1x128xf32>) -> tensor<1x1x128xf32> {
  %m = intent.reduce %x {fn = "max", axes = [0], assoc = "associative"} : (tensor<128x1x1x128xf32>) -> tensor<1x1x128xf32>
  return %m : tensor<1x1x128xf32>
}

// -----

// reduce{max} over ALL axes with keepdims (result [1x1], every dim reduced).
// C2b cubin fix: a naive all-reduction generic (all "reduction" iterators, no
// parallel dim) is valid MLIR but stays a HOST scf.for nest — the GPU pipeline
// cannot outline it, so the cubin returns only the fill identity. Instead we
// expand a leading UNIT parallel dim ([4x8] -> [1x4x8]), reduce the trailing
// axes keeping dim0 parallel, then collapse back to [1x1]. Now the whole
// reduction outlines into a 1-block GPU kernel (verified end-to-end for
// amax/vector_norm/std/var_mean/mse_loss).
// CHECK-LABEL: func.func @rall
// CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]] output_shape [1, 4, 8]
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction"]
// CHECK: arith.maxnumf
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2]]
func.func @rall(%x: tensor<4x8xf32>) -> tensor<1x1xf32> {
  %m = intent.reduce %x {fn = "max", axes = [0, 1], assoc = "associative"} : (tensor<4x8xf32>) -> tensor<1x1xf32>
  return %m : tensor<1x1xf32>
}

// -----

// reduce{mean} over ALL axes with keepdims: the leading-parallel-dim reduction
// followed by the 1/count elementwise scale, all over the [1x1x1] result.
// CHECK-LABEL: func.func @rmeanall
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction"]
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: tensor.collapse_shape
func.func @rmeanall(%x: tensor<4x8xf32>) -> tensor<1x1xf32> {
  %m = intent.reduce %x {fn = "mean", axes = [0, 1], assoc = "float_nondeterministic"} : (tensor<4x8xf32>) -> tensor<1x1xf32>
  return %m : tensor<1x1xf32>
}

// -----

// A full reduction may also match a rank-0 external ABI. Keep the leading unit
// parallel dimension for executable lowering, then collapse every unit result
// dimension to tensor<1xf32>, extract the scalar, and rebuild tensor<f32>.
// CHECK-LABEL: func.func @rscalar
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK: tensor.collapse_shape
// CHECK: tensor.extract
// CHECK: tensor.from_elements
func.func @rscalar(%x: tensor<37xf32>) -> tensor<f32> {
  %s = intent.reduce %x {fn = "sum", axes = [0], assoc = "float_nondeterministic"} : (tensor<37xf32>) -> tensor<f32>
  return %s : tensor<f32>
}

// -----

// Min uses minnum rather than minimum so the production CPU leg matches the
// Triton reduction combiner's non-NaN operand preference.
// CHECK-LABEL: func.func @rmin
// CHECK: arith.minnumf
func.func @rmin(%x: tensor<17xf32>) -> tensor<f32> {
  %m = intent.reduce %x {fn = "min", axes = [0], assoc = "associative"} : (tensor<17xf32>) -> tensor<f32>
  return %m : tensor<f32>
}

// -----

// Product remains a scalar multiply reduction.
// CHECK-LABEL: func.func @rprod
// CHECK: arith.mulf
func.func @rprod(%x: tensor<17xf32>) -> tensor<f32> {
  %p = intent.reduce %x {fn = "prod", axes = [0], assoc = "float_nondeterministic"} : (tensor<17xf32>) -> tensor<f32>
  return %p : tensor<f32>
}

// -----

// Integer reductions are needed by real Triton kernels such as count_nonzero:
// compare -> cast i1 to i64 -> reduce{sum}. They must lower to integer arith,
// not remain as a residual intent.reduce.
// CHECK-LABEL: func.func @rsum_i64
// CHECK: arith.constant 0 : i64
// CHECK: linalg.generic
// CHECK: arith.addi
// CHECK-NOT: intent.reduce
func.func @rsum_i64(%x: tensor<4x17xi64>) -> tensor<4xi64> {
  %s = intent.reduce %x {fn = "sum", axes = [1], assoc = "associative"} : (tensor<4x17xi64>) -> tensor<4xi64>
  return %s : tensor<4xi64>
}

// -----

// Integer extrema use signed semantics consistently for both the identity and
// combiner. This also exercises identities wider than a host double can encode.
// CHECK-LABEL: func.func @rmax_i64
// CHECK: arith.constant -9223372036854775808 : i64
// CHECK: arith.maxsi
func.func @rmax_i64(%x: tensor<17xi64>) -> tensor<i64> {
  %m = intent.reduce %x {fn = "max", axes = [0], assoc = "associative"} : (tensor<17xi64>) -> tensor<i64>
  return %m : tensor<i64>
}

// -----

// CHECK-LABEL: func.func @rmin_i32
// CHECK: arith.constant 2147483647 : i32
// CHECK: arith.minsi
func.func @rmin_i32(%x: tensor<17xi32>) -> tensor<i32> {
  %m = intent.reduce %x {fn = "min", axes = [0], assoc = "associative"} : (tensor<17xi32>) -> tensor<i32>
  return %m : tensor<i32>
}

// -----

// CHECK-LABEL: func.func @rprod_i32
// CHECK: arith.constant 1 : i32
// CHECK: arith.muli
func.func @rprod_i32(%x: tensor<17xi32>) -> tensor<i32> {
  %p = intent.reduce %x {fn = "prod", axes = [0], assoc = "associative"} : (tensor<17xi32>) -> tensor<i32>
  return %p : tensor<i32>
}
