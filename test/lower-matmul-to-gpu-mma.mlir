// RUN: intent-opt -convert-intent-contraction-to-gpu-mma %s | FileCheck %s

// The nvgpu.ldmatrix + nvgpu.mma.sync sibling of -convert-intent-contraction-
// to-gpu. Same recognizer ("mk,nk->mn", A/W f16 -> Y f32) and the same cp.async
// multistage staging, but the contraction is realized at the register level:
// nvgpu.ldmatrix loads the A (vector<4x2xf16>) and W (vector<2x2xf16>) fragments
// from the rotating dynamic-shared buffers and nvgpu.mma.sync (m16n8k16)
// accumulates into per-warp vector<2x2xf32> accumulators carried across the K
// loop as scf.for iter_args. W stays in its natural [N,K] layout (the "nk"
// transpose is folded into the ldmatrix index), so no transpose-on-copy.
// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:    %{{.*}}: memref<128x64xf16>, %{{.*}}: memref<256x64xf16>, %{{.*}}: memref<128x256xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         nvgpu.device_async_copy
// CHECK:         nvgpu.device_async_create_group
// CHECK:         scf.for
// CHECK-SAME:      iter_args
// CHECK:           nvgpu.device_async_wait
// CHECK:           gpu.barrier
// CHECK:           nvgpu.ldmatrix
// CHECK-SAME:        numTiles = 4
// CHECK:           nvgpu.ldmatrix
// CHECK-SAME:        numTiles = 2
// CHECK:           nvgpu.mma.sync
// CHECK-SAME:        mmaShape = [16, 8, 16]
// CHECK:           scf.yield
// CHECK:         vector.extract
// CHECK:         memref.store
// CHECK:         gpu.terminator
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     intent.
func.func @matmul(%A: tensor<128x64xf16>, %W: tensor<256x64xf16>) -> tensor<128x256xf32> {
  %Y = intent.contraction %A, %W {subscripts = "mk,nk->mn"}
      : (tensor<128x64xf16>, tensor<256x64xf16>) -> tensor<128x256xf32>
  return %Y : tensor<128x256xf32>
}

// -----

// Fused epilogue (bias-add + squared_relu) on the mma.sync path: the
// vector<2x2xf32> accumulators spill to a [BM,BN] shared tile, the block
// cooperatively reloads each element and evaluates bias-add + activation before
// the store -- identical epilogue evaluator to the WMMA pass.
// CHECK-LABEL: func.func @linear_sqrelu(
// CHECK:         nvgpu.mma.sync
// CHECK:         gpu.barrier
// CHECK:         arith.addf
// CHECK:         arith.cmpf oge
// CHECK:         arith.select
// CHECK:         arith.mulf
// CHECK:         memref.store
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     intent.
func.func @linear_sqrelu(%A: tensor<128x64xf16>, %W: tensor<256x64xf16>,
                         %bias: tensor<256xf32>) -> tensor<128x256xf32> {
  %Y = intent.contraction %A, %W {subscripts = "mk,nk->mn"}
      : (tensor<128x64xf16>, tensor<256x64xf16>) -> tensor<128x256xf32>
  %b = intent.map_binary %Y, %bias {fn = "add"}
      : (tensor<128x256xf32>, tensor<256xf32>) -> tensor<128x256xf32>
  %z = arith.constant dense<0.0> : tensor<128x256xf32>
  %c = intent.map_binary %b, %z {fn = "ge"}
      : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xi1>
  %r = intent.select %c, %b, %z
      : (tensor<128x256xi1>, tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %sq = intent.map_binary %r, %r {fn = "mul"}
      : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  return %sq : tensor<128x256xf32>
}
