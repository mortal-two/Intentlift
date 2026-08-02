// RUN: intent-opt -convert-intent-contraction-to-gpu-mma %s | FileCheck %s --check-prefix=MMA
// RUN: intent-opt -convert-intent-contraction-to-gpu-mma %s | mlir-opt -gpu-kernel-outlining -canonicalize | intent-opt -lower-bf16-mma-to-nvvm | FileCheck %s --check-prefix=NVVM

// BF16 uses the same 16-bit cp.async/ldmatrix storage geometry as FP16, while
// the tensor core accumulates into four f32 lanes per m16n8k16 fragment.

// MMA-LABEL: func.func @bf16_matmul(
// MMA-SAME:    %{{.*}}: memref<64x32xbf16>, %{{.*}}: memref<64x32xbf16>, %{{.*}}: memref<64x64xf32>)
// MMA:         nvgpu.device_async_copy
// MMA:         nvgpu.ldmatrix
// MMA-SAME:      memref<1x64x40xbf16
// MMA-SAME:      vector<4x2xbf16>
// MMA:         nvgpu.ldmatrix
// MMA-SAME:      vector<2x2xbf16>
// MMA:         nvgpu.mma.sync
// MMA-SAME:      (vector<4x2xbf16>, vector<2x2xbf16>, vector<2x2xf32>)
// MMA-SAME:      -> vector<2x2xf32>
// MMA-NOT:     intent.
func.func @bf16_matmul(%A: tensor<64x32xbf16>,
                       %W: tensor<64x32xbf16>) -> tensor<64x64xf32> {
  %Y = intent.contraction %A, %W {subscripts = "mk,nk->mn"}
      : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x64xf32>
  return %Y : tensor<64x64xf32>
}

// -----

// Explicit lossless widenings are semantic compute boundaries, not extra HBM
// tensors: the MMA pass peels them and still reads the original BF16 ABI.

// MMA-LABEL: func.func @bf16_matmul_explicit_f32(
// MMA-SAME:    %{{.*}}: memref<64x32xbf16>, %{{.*}}: memref<64x32xbf16>, %{{.*}}: memref<64x64xf32>)
// MMA:         nvgpu.mma.sync
// MMA-NOT:     arith.extf
// MMA-NOT:     intent.
func.func @bf16_matmul_explicit_f32(%A: tensor<64x32xbf16>,
                                    %W: tensor<64x32xbf16>)
    -> tensor<64x64xf32> {
  %Af = intent.cast %A {to_dtype = "f32"}
      : (tensor<64x32xbf16>) -> tensor<64x32xf32>
  %Wf = intent.cast %W {to_dtype = "f32"}
      : (tensor<64x32xbf16>) -> tensor<64x32xf32>
  %Y = intent.contraction %Af, %Wf {subscripts = "mk,nk->mn"}
      : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
  return %Y : tensor<64x64xf32>
}

// -----

// A normal f32 bias epilogue remains fused after BF16 tensor-core accumulation.

// MMA-LABEL: func.func @bf16_linear_bias(
// MMA:         nvgpu.mma.sync
// MMA:         gpu.barrier
// MMA:         memref.load {{.*}} : memref<64xf32>
// MMA:         arith.addf
// MMA:         memref.store {{.*}} : memref<64x64xf32>
// MMA-NOT:     intent.
func.func @bf16_linear_bias(%A: tensor<64x32xbf16>,
                            %W: tensor<64x32xbf16>,
                            %bias: tensor<64xf32>) -> tensor<64x64xf32> {
  %Y = intent.contraction %A, %W {subscripts = "mk,nk->mn"}
      : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x64xf32>
  %out = intent.map_binary %Y, %bias {fn = "add"}
      : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64x64xf32>
  return %out : tensor<64x64xf32>
}

// -----

// Two saved BF16 destinations share the register-resident bias epilogue. The
// path widens the BF16 bias once per output element, adds in f32, rounds once,
// and stores both results without a full f32 shared-memory spill.

// MMA-LABEL: func.func @bf16_linear_saved(
// MMA-SAME:    %{{.*}}: memref<64x32xbf16>, %{{.*}}: memref<64x32xbf16>, %{{.*}}: memref<32xvector<2xbf16>>, %{{.*}}: memref<64x32xvector<2xbf16>>, %{{.*}}: memref<64x32xvector<2xbf16>>)
// MMA:         nvgpu.mma.sync
// MMA:         memref.load {{.*}} : memref<32xvector<2xbf16>>
// MMA:         arith.extf {{.*}} : bf16 to f32
// MMA:         arith.addf
// MMA:         arith.truncf {{.*}} : f32 to bf16
// MMA:         memref.store {{.*}} : memref<64x32xvector<2xbf16>>
// MMA:         memref.store {{.*}} : memref<64x32xvector<2xbf16>>
// MMA-NOT:     memref<64x64xf32, #gpu.address_space<workgroup>>
// MMA-NOT:     intent.
func.func @bf16_linear_saved(%A: tensor<64x32xbf16>,
                             %W: tensor<64x32xbf16>,
                             %bias: tensor<64xbf16>)
    -> (tensor<64x64xbf16>, tensor<64x64xbf16>) {
  %Af = intent.cast %A {to_dtype = "f32"}
      : (tensor<64x32xbf16>) -> tensor<64x32xf32>
  %Wf = intent.cast %W {to_dtype = "f32"}
      : (tensor<64x32xbf16>) -> tensor<64x32xf32>
  %Y = intent.contraction %Af, %Wf {subscripts = "mk,nk->mn"}
      : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x64xf32>
  %bf = intent.cast %bias {to_dtype = "f32"}
      : (tensor<64xbf16>) -> tensor<64xf32>
  %pre = intent.map_binary %Y, %bf {fn = "add"}
      : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64x64xf32>
  %out = intent.cast %pre {to_dtype = "bf16"}
      : (tensor<64x64xf32>) -> tensor<64x64xbf16>
  %saved = intent.cast %pre {to_dtype = "bf16"}
      : (tensor<64x64xf32>) -> tensor<64x64xbf16>
  return %out, %saved : tensor<64x64xbf16>, tensor<64x64xbf16>
}

// The LLVM-20 backport must consume every BF16 nvgpu fragment and materialize
// the explicit NVVM instruction rather than leaving an unsupported mma op.
// NVVM:         nvvm.mma.sync
// NVVM-NOT:     nvgpu.mma.sync
