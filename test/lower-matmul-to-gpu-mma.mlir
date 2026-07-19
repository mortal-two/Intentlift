// RUN: intent-opt -convert-intent-contraction-to-gpu-mma %s | FileCheck %s
// RUN: intent-opt -convert-intent-contraction-to-gpu-mma='num-stages=5 block-m=16 block-n=32 block-k=32 smem-pad-k=8' %s | FileCheck %s --check-prefix=LIVE

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

// -----

// Captured flash-attn linear variant: the unchanged live graph widens the f16
// A/W/bias ABI to f32 explicitly, accumulates/adds in f32, then returns TWO f16
// destinations (C and the saved pre-activation ACT_INPUT).  The structural pass
// must peel only the exact input widenings, preserve the f16 destination ABI,
// and write both results from the same tensor-core launch.
// CHECK-LABEL: func.func @linear_live_saved(
// CHECK-SAME:    %{{.*}}: memref<128x256xf16>, %{{.*}}: memref<512x256xf16>, %{{.*}}: memref<512xf16>, %{{.*}}: memref<128x512xf16>, %{{.*}}: memref<128x512xf16>)
// CHECK:         gpu.launch
// CHECK:         nvgpu.mma.sync
// CHECK:         arith.extf
// CHECK:         arith.addf
// CHECK:         arith.truncf
// CHECK:         memref.store
// CHECK:         memref.store
// CHECK-NOT:     memref<64x64xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     intent.
// LIVE-LABEL:    func.func @linear_live_saved(
// LIVE-SAME:     intent.launch_plan = {block = [32, 1, 1]
// LIVE-SAME:     grid = [128, 1, 1]
// LIVE-SAME:     pipeline_depth = 5 : i64
// LIVE-SAME:     smem = 19200 : i64
// LIVE:          %c128 = arith.constant 128 : index
// LIVE:          %c32 = arith.constant 32 : index
// LIVE:          %c19200_i32 = arith.constant 19200 : i32
// LIVE:          gpu.launch blocks(
// LIVE:          nvgpu.mma.sync
// LIVE:          arith.addf
// LIVE:          arith.truncf
// LIVE:          memref.store
// LIVE:          memref.store
// LIVE-NOT:      memref<16x32xf32, #gpu.address_space<workgroup>>
// LIVE-NOT:      intent.
func.func @linear_live_saved(%A: tensor<128x256xf16>,
                             %W: tensor<512x256xf16>,
                             %bias: tensor<512xf16>)
    -> (tensor<128x512xf16>, tensor<128x512xf16>) {
  %Af = intent.cast %A {to_dtype = "f32"}
      : (tensor<128x256xf16>) -> tensor<128x256xf32>
  %Wf = intent.cast %W {to_dtype = "f32"}
      : (tensor<512x256xf16>) -> tensor<512x256xf32>
  %Y = intent.contraction %Af, %Wf {subscripts = "mk,nk->mn"}
      : (tensor<128x256xf32>, tensor<512x256xf32>) -> tensor<128x512xf32>
  %bf = intent.cast %bias {to_dtype = "f32"}
      : (tensor<512xf16>) -> tensor<512xf32>
  %pre = intent.map_binary %Y, %bf {fn = "add"}
      : (tensor<128x512xf32>, tensor<512xf32>) -> tensor<128x512xf32>
  %saved = intent.cast %pre {to_dtype = "f16"}
      : (tensor<128x512xf32>) -> tensor<128x512xf16>
  %out = intent.cast %pre {to_dtype = "f16"}
      : (tensor<128x512xf32>) -> tensor<128x512xf16>
  return %out, %saved : tensor<128x512xf16>, tensor<128x512xf16>
}
