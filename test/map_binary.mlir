// RUN: intent-opt %s | FileCheck %s

// CHECK-LABEL: func.func @softmax_step
func.func @softmax_step(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: intent.reduce
  %m = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"}
       : (tensor<4x8xf32>) -> tensor<4x1xf32>
  // CHECK: intent.map_binary
  %s = intent.map_binary %x, %m {fn = "sub"}
       : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  return %s : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @qk
func.func @qk(%q: tensor<2x3x16x64xf16>, %k: tensor<2x3x16x64xf16>)
    -> tensor<2x3x16x16xf32> {
  // CHECK: intent.contraction
  %s = intent.contraction %q, %k {subscripts = "bhqd,bhkd->bhqk"}
       : (tensor<2x3x16x64xf16>, tensor<2x3x16x64xf16>) -> tensor<2x3x16x16xf32>
  return %s : tensor<2x3x16x16xf32>
}
