// RUN: intent-opt -split-input-file -convert-intent-row-reduction-to-gpu=block-threads=4 %s | FileCheck %s

// A receipt-bound typed scalar equality is a specialization contract, not a
// name-based rewrite. The scalar pointer remains in the stable ABI, while the
// fast cubin constant-folds the validated value. The automatic selector keeps
// the generic dynamic-scalar P5 as the mismatch fallback.

// CHECK-LABEL: func.func @scalar_equality
// CHECK-SAME: %[[H:.*]]: memref<i32>
// CHECK: intent.scalar_equalities = array<i64: 1, 64>
// CHECK: gpu.launch
// CHECK: arith.constant 64 : i32
// CHECK-NOT: memref.load %[[H]][]
// CHECK: arith.divsi
// CHECK-NOT: intent.
func.func @scalar_equality(
    %X: tensor<2x64xf32>, %hidden: tensor<i32>) -> tensor<2x64xf32>
    attributes {intent.scalar_equalities = array<i64: 1, 64>} {
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x64xf32>) -> tensor<2x1xf32>
  %hidden_f = intent.cast %hidden {to_dtype = "f32"}
      : (tensor<i32>) -> tensor<f32>
  %mean = intent.map_binary %sum, %hidden_f {fn = "div"}
      : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %index = intent.iota : tensor<64xi32>
  %bucket = intent.map_binary %index, %hidden {fn = "div"}
      : (tensor<64xi32>, tensor<i32>) -> tensor<64xi32>
  %bucket_f = intent.cast %bucket {to_dtype = "f32"}
      : (tensor<64xi32>) -> tensor<64xf32>
  %centered = intent.map_binary %X, %mean {fn = "sub"}
      : (tensor<2x64xf32>, tensor<2x1xf32>) -> tensor<2x64xf32>
  %Y = intent.map_binary %centered, %bucket_f {fn = "add"}
      : (tensor<2x64xf32>, tensor<64xf32>) -> tensor<2x64xf32>
  return %Y : tensor<2x64xf32>
}

// -----

// A malformed equality on a floating scalar is rejected fail-closed.

// CHECK-LABEL: func.func @bad_scalar_equality
// CHECK: intent.reduce
// CHECK-NOT: gpu.launch
func.func @bad_scalar_equality(
    %X: tensor<2x64xf32>, %scale: tensor<f32>) -> tensor<2x64xf32>
    attributes {intent.scalar_equalities = array<i64: 1, 64>} {
  %sum = intent.reduce %X {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<2x64xf32>) -> tensor<2x1xf32>
  %Y = intent.map_binary %X, %sum {fn = "sub"}
      : (tensor<2x64xf32>, tensor<2x1xf32>) -> tensor<2x64xf32>
  return %Y : tensor<2x64xf32>
}
