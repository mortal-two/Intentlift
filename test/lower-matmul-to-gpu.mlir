// RUN: intent-opt -convert-intent-contraction-to-gpu %s | FileCheck %s

// A single intent.contraction{"mk,nk->mn"} (A[M,K] f16, W[N,K] f16 -> Y[M,N]
// f32) lowered to ONE tiled-K tensor-core gpu.launch: BM=BN=64, BK=32, a 4x4
// grid of m16n16k16 WMMA tiles over 4 warps. The per-warp fp32 accumulator
// fragments are carried across the K loop as scf.for iter_args; the A/W tiles are
// streamed global->shared by an N-stage cp.async pipeline (nvgpu.device_async_*
// into rotating dynamic-shared buffers, prologue prefetch + steady-state
// prefetch-K+N-1-while-compute-K + epilogue drain); the contraction is realized
// by gpu.subgroup_mma_compute (-> wmma.mma.sync on sm_120a), NOT a linalg.generic
// scalar MAC. The B operand is loaded transposed so "mk,nk->mn" is exact.
// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:    %{{.*}}: memref<128x64xf16>, %{{.*}}: memref<256x64xf16>, %{{.*}}: memref<128x256xf32>)
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         gpu.subgroup_mma_constant_matrix
// CHECK:         nvgpu.device_async_copy
// CHECK:         nvgpu.device_async_create_group
// CHECK:         scf.for
// CHECK-SAME:      iter_args
// CHECK:           nvgpu.device_async_copy
// CHECK:           nvgpu.device_async_create_group
// CHECK:           nvgpu.device_async_wait
// CHECK:           gpu.barrier
// CHECK:           gpu.subgroup_mma_load_matrix
// CHECK:           gpu.subgroup_mma_compute
// CHECK:           scf.yield
// CHECK:         nvgpu.device_async_wait
// CHECK:         gpu.subgroup_mma_store_matrix
// CHECK:         gpu.terminator
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     intent.
func.func @matmul(%A: tensor<128x64xf16>, %W: tensor<256x64xf16>) -> tensor<128x256xf32> {
  %Y = intent.contraction %A, %W {subscripts = "mk,nk->mn"}
      : (tensor<128x64xf16>, tensor<256x64xf16>) -> tensor<128x256xf32>
  return %Y : tensor<128x256xf32>
}

// M7b fused epilogue: contraction + bias-add (Col[N]) + squared_relu
// (relu(x)=select(x>=0,x,0); square = mul(relu,relu)) applied to the COp
// accumulator fragments via an extract-to-shared-then-scalar-arith round trip.
// The 4 fragments spill to a [BM,BN] f32 staging tile, the block reloads each
// element, and the bias-add (arith.addf with a Col gather) + activation
// (arith.cmpf oge / arith.select / arith.mulf) is evaluated before the store.
// CHECK-LABEL: func.func @linear_sqrelu(
// CHECK-SAME:    %{{.*}}: memref<128x64xf16>, %{{.*}}: memref<256x64xf16>, %{{.*}}: memref<256xf32>, %{{.*}}: memref<128x256xf32>)
// CHECK:         gpu.launch
// CHECK:         scf.for
// CHECK-SAME:      iter_args
// CHECK:           gpu.subgroup_mma_compute
// CHECK:         gpu.subgroup_mma_store_matrix %{{.*}}, %{{.*}} {leadDimension = 64
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

// The canonical WMMA structure also consumes the formal Flash Linear spelling:
// explicit f16->f32 input/bias casts, spelling-independent row contraction, and
// two observable f16 saved outputs.  Unlike the register-epilogue Full state,
// this canonical path materializes one block-local f32 accumulator tile before
// evaluating the typed epilogue, which gives C-Fixed/Schedule-Only a genuine
// fixed structure that can be retiled.
// CHECK-LABEL: func.func @linear_live_saved(
// CHECK-SAME:    %{{.*}}: memref<128x256xf16>, %{{.*}}: memref<512x256xf16>, %{{.*}}: memref<512xf16>, %{{.*}}: memref<128x512xf16>, %{{.*}}: memref<128x512xf16>)
// CHECK:         gpu.launch
// CHECK:         memref<64x64xf32, #gpu.address_space<workgroup>>
// CHECK:         gpu.subgroup_mma_compute
// CHECK:         arith.extf
// CHECK:         arith.addf
// CHECK:         arith.truncf
// CHECK:         memref.store
// CHECK:         arith.truncf
// CHECK:         memref.store
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     intent.
func.func @linear_live_saved(%A: tensor<128x256xf16>,
                             %W: tensor<512x256xf16>,
                             %bias: tensor<512xf16>)
    -> (tensor<128x512xf16>, tensor<128x512xf16>) {
  %Af = intent.cast %A {to_dtype = "f32"}
      : (tensor<128x256xf16>) -> tensor<128x256xf32>
  %Wf = intent.cast %W {to_dtype = "f32"}
      : (tensor<512x256xf16>) -> tensor<512x256xf32>
  %Y = intent.contraction %Af, %Wf {subscripts = "ik,jk->ij"}
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
