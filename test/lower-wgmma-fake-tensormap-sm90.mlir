// RUN: mlir-opt %s -gpu-kernel-outlining -canonicalize -cse \
// RUN:   --gpu-lower-to-nvvm-pipeline='cubin-chip=sm_90a cubin-format=isa kernel-bare-ptr-calling-convention=true' \
// RUN:   | FileCheck %s

// The stock NVGPU lowering derives the WGMMA descriptor bits from the shared
// memref and the *type* of the tensor-map operand.  It does not consume a TMA
// descriptor value.  This small test keeps descriptor formation device-local,
// so a portable bare-pointer launch does not acquire CUDA tensor-map ABI args.

!tm64 = !nvgpu.tensormap.descriptor<tensor = memref<64x64xf16, #gpu.address_space<workgroup>>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
!wg64 = !nvgpu.warpgroup.descriptor<tensor = memref<64x64xf16, #gpu.address_space<workgroup>>>
!acc64 = !nvgpu.warpgroup.accumulator<fragmented = vector<64x64xf32>>

module {
  func.func @wgmma_fake_tensormap() {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c32768 = arith.constant 32768 : i32
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%bxn = %c128, %byn = %c1, %bzn = %c1)
               dynamic_shared_memory_size %c32768 {
      %raw = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
      %c0 = arith.constant 0 : index
      %c8192 = arith.constant 8192 : index
      %c16384 = arith.constant 16384 : index
      %a = memref.view %raw[%c0][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
      %b = memref.view %raw[%c8192][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
      %d = memref.view %raw[%c16384][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf32, #gpu.address_space<workgroup>>
      %zero = arith.constant 0 : i64
      %tm_a = builtin.unrealized_conversion_cast %zero : i64 to !tm64
      %tm_b = builtin.unrealized_conversion_cast %zero : i64 to !tm64
      %da = nvgpu.warpgroup.generate.descriptor %a, %tm_a : memref<64x64xf16, #gpu.address_space<workgroup>>, !tm64 -> !wg64
      %db = nvgpu.warpgroup.generate.descriptor %b, %tm_b : memref<64x64xf16, #gpu.address_space<workgroup>>, !tm64 -> !wg64
      %acc = nvgpu.warpgroup.mma.init.accumulator -> !acc64
      %result = nvgpu.warpgroup.mma %da, %db, %acc : !wg64, !wg64, !acc64 -> !acc64
      nvgpu.warpgroup.mma.store %result, %d : !acc64 to memref<64x64xf32, #gpu.address_space<workgroup>>
      gpu.terminator
    }
    return
  }
}

// The pipeline serializes the final PTX into gpu.binary, so assert the target
// instruction rather than the pre-serialization NVVM operation spelling.
// CHECK: wgmma.mma_async.sync.aligned
