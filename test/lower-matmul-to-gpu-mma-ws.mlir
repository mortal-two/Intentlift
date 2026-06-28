// RUN: intent-opt -convert-intent-contraction-to-gpu-mma-ws %s | FileCheck %s

// M7g: the WARP-SPECIALIZED (producer/consumer) sibling of
// -convert-intent-contraction-to-gpu-mma. Same recognizer and the same
// register-level nvgpu.ldmatrix + nvgpu.mma.sync contraction, but the block's
// warps are split: the producer warps issue cp.async loads into an N-stage
// shared buffer and signal a per-stage `full` mbarrier (waiting the matching
// `empty` mbarrier before reuse); the consumer warps wait `full`, run mma.sync
// into the register accumulator, and signal `empty`. The handshake uses the
// nvgpu.mbarrier primitives (create/init/arrive/try_wait.parity). Both mbarrier
// groups are created and initialized (one full + one empty), and the
// producer/consumer split is realized by an scf.if on the warp id.
// CHECK-LABEL: func.func @matmul(
// CHECK:         gpu.launch
// CHECK:         gpu.dynamic_shared_memory
// CHECK:         nvgpu.mbarrier.create
// CHECK:         nvgpu.mbarrier.create
// CHECK:         nvgpu.mbarrier.init
// CHECK:         gpu.barrier
// CHECK:         scf.if
// CHECK:           scf.for
// CHECK-SAME:        iter_args
// CHECK:             nvgpu.mbarrier.try_wait.parity
// CHECK:             nvgpu.ldmatrix
// CHECK:             nvgpu.mma.sync
// CHECK:             nvgpu.mbarrier.arrive
// CHECK:         } else {
// CHECK:           scf.for
// CHECK:             nvgpu.device_async_copy
// CHECK:             nvgpu.device_async_create_group
// CHECK:             nvgpu.device_async_wait
// CHECK:             nvgpu.mbarrier.arrive
// CHECK:         gpu.terminator
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     intent.
func.func @matmul(%A: tensor<128x64xf16>, %W: tensor<256x64xf16>) -> tensor<128x256xf32> {
  %Y = intent.contraction %A, %W {subscripts = "mk,nk->mn"}
      : (tensor<128x64xf16>, tensor<256x64xf16>) -> tensor<128x256xf32>
  return %Y : tensor<128x256xf32>
}
