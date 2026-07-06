// RUN: intent-opt -convert-intent-conv-to-gpu=block-threads=2 %s | FileCheck %s

// C7 convolution: forward conv1d/conv2d/conv3d lowered to ONE block-cooperative
// gpu.launch (direct windowed-weighted accumulation extending the pool reduce).
// grid = (N*Cout, prod(outer out-spatial), 1); threads over the innermost
// out-spatial axis. Each output cell = bias[co] + sum over (ci, kernel taps) of
// in*weight, with a single in-bounds scf.if per (ci,tap).

// ---------------------------------------------------------------------------- //
// conv2d, no pad, stride 1, groups 1, with bias. N=2,Cin=4,Cout=8,16x16,k=3.
// CHECK-LABEL: func.func @conv2d(
// CHECK-SAME:    %{{.*}}: memref<2x4x16x16xf32>, %{{.*}}: memref<8x4x3x3xf32>, %{{.*}}: memref<8xf32>, %{{.*}}: memref<2x8x14x14xf32>)
// CHECK:         gpu.launch
// CHECK-NOT:     gpu.all_reduce
// CHECK:         memref.load {{.*}} : memref<8xf32>
// CHECK:         scf.if
// CHECK:           memref.load {{.*}} : memref<2x4x16x16xf32>
// CHECK:           memref.load {{.*}} : memref<8x4x3x3xf32>
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:         memref.store
// CHECK:         gpu.terminator
func.func @conv2d(%in: tensor<2x4x16x16xf32>, %w: tensor<8x4x3x3xf32>,
    %b: tensor<8xf32>) -> tensor<2x8x14x14xf32> {
  %o = intent.conv %in, %w, %b {stride = [1, 1], padding = [0, 0],
      dilation = [1, 1], groups = 1 : i64}
      : (tensor<2x4x16x16xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>)
        -> tensor<2x8x14x14xf32>
  return %o : tensor<2x8x14x14xf32>
}

// -----

// conv2d, pad=1 stride=2 groups=2, no bias.
// CHECK-LABEL: func.func @conv2d_g(
// CHECK-SAME:    %{{.*}}: memref<2x4x16x16xf32>, %{{.*}}: memref<8x2x3x3xf32>, %{{.*}}: memref<2x8x8x8xf32>)
// CHECK:         gpu.launch
// CHECK:         scf.if
// CHECK:         gpu.terminator
func.func @conv2d_g(%in: tensor<2x4x16x16xf32>, %w: tensor<8x2x3x3xf32>)
    -> tensor<2x8x8x8xf32> {
  %o = intent.conv %in, %w {stride = [2, 2], padding = [1, 1],
      dilation = [1, 1], groups = 2 : i64}
      : (tensor<2x4x16x16xf32>, tensor<8x2x3x3xf32>) -> tensor<2x8x8x8xf32>
  return %o : tensor<2x8x8x8xf32>
}

// -----

// conv1d (spatial rank 1): grid.y = 1, threads over the single out axis.
// CHECK-LABEL: func.func @conv1d(
// CHECK-SAME:    %{{.*}}: memref<2x4x16xf32>, %{{.*}}: memref<8x4x3xf32>, %{{.*}}: memref<2x8x14xf32>)
// CHECK:         gpu.launch
// CHECK:         scf.if
// CHECK:         memref.store
// CHECK:         gpu.terminator
func.func @conv1d(%in: tensor<2x4x16xf32>, %w: tensor<8x4x3xf32>)
    -> tensor<2x8x14xf32> {
  %o = intent.conv %in, %w {stride = [1], padding = [0], dilation = [1],
      groups = 1 : i64}
      : (tensor<2x4x16xf32>, tensor<8x4x3xf32>) -> tensor<2x8x14xf32>
  return %o : tensor<2x8x14xf32>
}

// -----

// conv3d (spatial rank 3): grid.y = prod of the two outer out-spatial dims.
// CHECK-LABEL: func.func @conv3d(
// CHECK-SAME:    %{{.*}}: memref<1x2x8x8x8xf32>, %{{.*}}: memref<4x2x3x3x3xf32>, %{{.*}}: memref<1x4x6x6x6xf32>)
// CHECK:         gpu.launch
// CHECK:         scf.if
// CHECK:         memref.store
// CHECK:         gpu.terminator
func.func @conv3d(%in: tensor<1x2x8x8x8xf32>, %w: tensor<4x2x3x3x3xf32>)
    -> tensor<1x4x6x6x6xf32> {
  %o = intent.conv %in, %w {stride = [1, 1, 1], padding = [0, 0, 0],
      dilation = [1, 1, 1], groups = 1 : i64}
      : (tensor<1x2x8x8x8xf32>, tensor<4x2x3x3x3xf32>) -> tensor<1x4x6x6x6xf32>
  return %o : tensor<1x4x6x6x6xf32>
}
