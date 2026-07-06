// RUN: intent-opt -convert-intent-pool-to-gpu=block-threads=4 %s | FileCheck %s --check-prefix=POOL
// RUN: intent-opt -convert-intent-upsample-to-gpu=block-threads=8 %s | FileCheck %s --check-prefix=UP
// RUN: intent-opt -convert-intent-upsample-to-gpu=block-threads=8 %s | FileCheck %s --check-prefix=UPB
// RUN: intent-opt -convert-intent-polar-to-gpu=block-threads=8 %s | FileCheck %s --check-prefix=POLAR

// C7 windowed-reduce / interpolation / complex. Each op lowered to ONE
// block-cooperative gpu.launch over NCHW (pool/upsample: grid = (N*C, OH, 1),
// threads over OW) or rank-2 (polar: one block per row, threads over cols).

// ---------------------------------------------------------------------------- //
// avg_pool2d: bounded per-cell window reduce, divide by kH*kW (count_include_pad).
// POOL-LABEL: func.func @avgp(
// POOL-SAME:    %{{.*}}: memref<1x2x8x8xf32>, %{{.*}}: memref<1x2x4x4xf32>)
// POOL:         gpu.launch
// POOL-NOT:     gpu.all_reduce
// POOL:         scf.if
// POOL:           memref.load
// POOL:         arith.divf
// POOL:         memref.store
// POOL:         gpu.terminator
func.func @avgp(%a: tensor<1x2x8x8xf32>) -> tensor<1x2x4x4xf32> {
  %o = intent.pool %a {fn = "avg", kH = 2 : i64, kW = 2 : i64, sH = 2 : i64,
      sW = 2 : i64, padH = 0 : i64, padW = 0 : i64}
      : (tensor<1x2x8x8xf32>) -> tensor<1x2x4x4xf32>
  return %o : tensor<1x2x4x4xf32>
}

// -----

// max_pool2d_with_indices: window max + the argmax FLAT index (i64), two outputs.
// POOL-LABEL: func.func @maxp(
// POOL-SAME:    %{{.*}}: memref<1x2x8x8xf32>, %{{.*}}: memref<1x2x4x4xf32>, %{{.*}}: memref<1x2x4x4xi64>)
// POOL:         gpu.launch
// POOL:         scf.if
// POOL:           arith.cmpf
// POOL:           arith.select
// POOL:         memref.store
// POOL:         memref.store
// POOL:         gpu.terminator
func.func @maxp(%a: tensor<1x2x8x8xf32>) -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xi64>) {
  %o, %i = intent.pool %a {fn = "max", kH = 2 : i64, kW = 2 : i64, sH = 2 : i64,
      sW = 2 : i64, padH = 0 : i64, padW = 0 : i64}
      : (tensor<1x2x8x8xf32>) -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xi64>)
  return %o, %i : tensor<1x2x4x4xf32>, tensor<1x2x4x4xi64>
}

// -----

// upsample_nearest2d: affine floor index-remap gather (no arithmetic on value).
// UP-LABEL: func.func @up(
// UP-SAME:    %{{.*}}: memref<1x2x4x4xf32>, %{{.*}}: memref<1x2x8x8xf32>)
// UP:         gpu.launch
// UP:         arith.mulf
// UP:         arith.fptosi
// UP:         memref.load
// UP:         memref.store
// UP:         gpu.terminator
func.func @up(%a: tensor<1x2x4x4xf32>) -> tensor<1x2x8x8xf32> {
  %o = intent.upsample %a {mode = "nearest"}
      : (tensor<1x2x4x4xf32>) -> tensor<1x2x8x8xf32>
  return %o : tensor<1x2x8x8xf32>
}

// -----

// upsample_bicubic2d(_aa): 4x4 cubic-convolution neighborhood (a=-0.75). floor
// base + fraction, four weights per axis, 16 weighted loads accumulated.
// UPB-LABEL: func.func @upb(
// UPB-SAME:    %{{.*}}: memref<1x2x4x4xf32>, %{{.*}}: memref<1x2x8x8xf32>)
// UPB:         gpu.launch
// UPB:         arith.fptosi
// UPB:         arith.sitofp
// UPB:         arith.select
// UPB:         memref.load
// UPB:         arith.mulf
// UPB:         arith.addf
// UPB:         memref.store
// UPB:         gpu.terminator
func.func @upb(%a: tensor<1x2x4x4xf32>) -> tensor<1x2x8x8xf32> {
  %o = intent.upsample %a {mode = "bicubic"}
      : (tensor<1x2x4x4xf32>) -> tensor<1x2x8x8xf32>
  return %o : tensor<1x2x8x8xf32>
}

// -----

// polar: real = abs*cos(angle), imag = abs*sin(angle) — two real results.
// POLAR-LABEL: func.func @pol(
// POLAR-SAME:    %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>, %{{.*}}: memref<4x8xf32>)
// POLAR:         gpu.launch
// POLAR:         math.cos
// POLAR:         math.sin
// POLAR:         memref.store
// POLAR:         memref.store
// POLAR:         gpu.terminator
func.func @pol(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %r, %i = intent.polar %a, %b
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  return %r, %i : tensor<4x8xf32>, tensor<4x8xf32>
}
