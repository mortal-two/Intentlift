// RUN: intent-opt -convert-intent-elementwise-to-gpu="block-threads=128 flatten-contiguous=true" %s | FileCheck %s

// A Row-only contiguous graph can flatten all rows into one vec4 packet domain.
// M=4,N=256 gives 256 vector packets and two 128-thread blocks, instead of four
// row blocks. The selector is purely structural; no symbol-specific branch is
// needed for the sigmoid/multiply DAG.
// CHECK-LABEL: func.func @arbitrary_activation(
// CHECK-SAME: %{{.*}}: memref<256xvector<4xf32>>, %{{.*}}: memref<256xvector<4xf32>>, %{{.*}}: memref<256xvector<4xf32>>)
// CHECK: %[[GRID:.*]] = arith.constant 2 : index
// CHECK: %[[BLOCK:.*]] = arith.constant 128 : index
// CHECK: gpu.launch{{.*}}%[[GRID]]{{.*}}%[[BLOCK]]
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<256xvector<4xf32>>
// CHECK: math.exp
// CHECK: vector.from_elements
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<256xvector<4xf32>>
// CHECK: gpu.terminator
// CHECK-NOT: intent.
func.func @arbitrary_activation(
    %A: tensor<4x256xf32>, %B: tensor<4x256xf32>) -> tensor<4x256xf32> {
  %s = intent.map_unary %A {fn = "sigmoid"} :
      (tensor<4x256xf32>) -> tensor<4x256xf32>
  %gate = intent.map_binary %A, %s {fn = "mul"} :
      (tensor<4x256xf32>, tensor<4x256xf32>) -> tensor<4x256xf32>
  %C = intent.map_binary %gate, %B {fn = "mul"} :
      (tensor<4x256xf32>, tensor<4x256xf32>) -> tensor<4x256xf32>
  return %C : tensor<4x256xf32>
}

// -----

// A flat packet domain remains legal with ordinary column and scalar
// broadcasts. The absolute packet address indexes X/Y, while remui recovers the
// logical feature column before gamma/beta loads.
// CHECK-LABEL: func.func @row_col_scalar_activation(
// CHECK-SAME: %{{[^:]+}}: memref<1024xvector<4xf32>>
// CHECK-SAME: %{{[^:]+}}: memref<1xf32>
// CHECK-SAME: %{{[^:]+}}: memref<256xf32>
// CHECK-SAME: %{{[^:]+}}: memref<256xf32>
// CHECK-SAME: %{{[^:]+}}: memref<1024xvector<4xf32>>)
// CHECK: arith.remui
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<256xf32>
// CHECK: vector.from_elements
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xvector<4xf32>>
func.func @row_col_scalar_activation(
    %X: tensor<16x256xf32>, %alpha: tensor<1xf32>,
    %gamma: tensor<256xf32>, %beta: tensor<256xf32>)
    -> tensor<16x256xf32> {
  %scaled = intent.map_binary %X, %alpha {fn = "mul"} :
      (tensor<16x256xf32>, tensor<1xf32>) -> tensor<16x256xf32>
  %activated = intent.map_unary %scaled {fn = "tanh"} :
      (tensor<16x256xf32>) -> tensor<16x256xf32>
  %weighted = intent.map_binary %activated, %gamma {fn = "mul"} :
      (tensor<16x256xf32>, tensor<256xf32>) -> tensor<16x256xf32>
  %Y = intent.map_binary %weighted, %beta {fn = "add"} :
      (tensor<16x256xf32>, tensor<256xf32>) -> tensor<16x256xf32>
  return %Y : tensor<16x256xf32>
}
