// RUN: intent-opt -convert-intent-elementwise-to-gpu='block-threads=64 promote-uniform-scalars=true' %s | FileCheck %s

// Constexpr control specialization may remove every use of an optional ABI
// input. The C++ lowering must preserve that memref in the public host
// signature without inventing a device load or rejecting the whole graph.
// CHECK-LABEL: func.func @dead_optional_input(
// CHECK-SAME: %{{.*}}: memref<2x64xf32>, %{{.*}}: memref<2x64xf32>, %[[DEAD:.*]]: memref<2xi64>, %{{.*}}: memref<f32>
// CHECK-NOT: memref.load %[[DEAD]]
// CHECK: gpu.launch
// CHECK-NOT: memref.load %[[DEAD]]
// CHECK: gpu.terminator
// CHECK-NOT: intent.
func.func @dead_optional_input(
    %p: tensor<2x64xf32>, %q: tensor<2x64xf32>,
    %optional_label: tensor<2xi64>, %scale: tensor<f32>)
    -> (tensor<2x64xf32>, tensor<2x64xf32>) {
  %half = arith.constant dense<0.5> : tensor<f32>
  %negative_half = arith.constant dense<-0.5> : tensor<f32>
  %diff = intent.map_binary %p, %q {fn = "sub"}
    : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
  %abs = intent.map_unary %diff {fn = "abs"}
    : (tensor<2x64xf32>) -> tensor<2x64xf32>
  %loss = intent.map_binary %half, %abs {fn = "mul"}
    : (tensor<f32>, tensor<2x64xf32>) -> tensor<2x64xf32>
  %positive = intent.map_binary %half, %scale {fn = "mul"}
    : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %negative = intent.map_binary %negative_half, %scale {fn = "mul"}
    : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %condition = intent.map_binary %p, %q {fn = "gt"}
    : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xi1>
  %grads = intent.select %condition, %positive, %negative
    : (tensor<2x64xi1>, tensor<f32>, tensor<f32>) -> tensor<2x64xf32>
  return %loss, %grads : tensor<2x64xf32>, tensor<2x64xf32>
}
