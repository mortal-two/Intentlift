#ifndef INTENT_CONVERSION_PASSES
#define INTENT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertIntentToLinalg : Pass<"convert-intent-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Lower intent atomic ops to linalg/tensor/arith on tensors.";
  let description = [{
    Lowers Layer-A atomic ops of the `intent` dialect to standard MLIR
    (linalg/tensor/arith), the first leg of the back-end spine
    intent -> linalg -> {scf,vector} -> gpu -> nvvm -> cubin. Starts with the
    elementwise `intent.map_binary` (equal-shape, no broadcast); reduce /
    contraction / broadcast follow.
  }];
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
  ];
}

def ConvertIntentRowReductionToGPU
    : Pass<"convert-intent-row-reduction-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a row-local reduce+broadcast intent func to one "
                "read-once block-cooperative gpu.launch.";
  let description = [{
    Recognizes `func.func`s whose body is a straight-line graph of
    `intent.map_unary`/`intent.map_binary`/`intent.reduce` that computes a
    2-D `[M,N]` output where every `intent.reduce` is over the last axis
    (keepdims `[M,1]`) — the "row normalization" shape (rms_norm, softmax,
    layer_norm, ...). It rewrites such a func in place to a memref-arg function
    holding a single `gpu.launch`: one block per row, the feature dim split
    over the block's threads, each thread reading its columns **once** into a
    private `memref.alloca` cache, one standalone `gpu.all_reduce` per reduce
    (→ warp `shfl.sync.bfly` + shared-mem on NVIDIA), per-row scalars hoisted
    once, and the epilogue writing the output from the cache (read-once = HBM
    parity with the library kernel). This is the verified block-cooperative
    path; it is general over the reduce/map DAG (NO per-kernel recognizer).
    Non-matching funcs are left untouched for `convert-intent-to-linalg`.

    Static shapes only. Run BEFORE `convert-intent-to-linalg`; the resulting
    memref+gpu.launch module goes straight to the device pipeline
    (`lower_memref_module_to_kernels`), bypassing bufferization. Transcendentals
    are emitted as `math.exp`/`math.sqrt`/`math.rsqrt` and rewritten to NVVM
    approx intrinsics by the `-lower-gpu-math-to-nvvm` target peephole.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (block size over the feature dim); 0 = auto "
           "(largest power-of-two <= min(N,256) dividing N). The PK harness "
           "passes the library kernel's geometry (e.g. 128 / 256).">,
    Option<"tileCols", "tile-cols", "unsigned", "32",
           "Max f32 columns held in registers per thread (register budget). "
           "When N/T exceeds this, the per-thread feature strip is split into "
           "num_tiles tiles of <= tile-cols each: each reduce becomes an "
           "scf.for over tiles accumulating a per-thread partial with ONE "
           "gpu.all_reduce AFTER the loop, and the row is cached in a "
           "workgroup (dynamic shared) buffer so the normalize epilogue does "
           "not re-read it from HBM. num_tiles==1 (N/T <= tile-cols) keeps the "
           "M1 register-tile path byte-identical.">,
  ];
}

def ConvertIntentElementwiseToGPU
    : Pass<"convert-intent-elementwise-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a pure-elementwise intent func to one read-once "
                "block-cooperative gpu.launch.";
  let description = [{
    The sibling of `convert-intent-row-reduction-to-gpu` for funcs whose body is
    a straight-line graph of `intent.map_unary`/`intent.map_binary` with NO
    `intent.reduce` — pure elementwise activations (dyt = tanh(alpha*x)*gamma,
    swiglu = (a*sigmoid(a))*b, geglu = the tanh-approx GELU gate * b). It
    rewrites such a func in place to a memref-arg function holding a single
    `gpu.launch`: one block per row, the feature dim split over the block's
    threads, each thread reading its columns **once** into SSA registers
    (read-once = HBM parity with the library kernel), per-row scalars hoisted
    once, and a multi-output epilogue writing each result from the register tile.
    There is NO `gpu.all_reduce` — nothing is reduced. General over the map DAG
    (NO per-kernel recognizer); the two passes' match() predicates are disjoint
    (this pass returns failure on ANY reduce; row-reduction returns failure with
    no reduce), so they compose in either order.

    Static shapes only. The resulting memref+gpu.launch module goes straight to
    the device pipeline (`lower_memref_module_to_kernels`), bypassing
    bufferization. Transcendentals (`math.exp`/`math.tanh`/...) are rewritten to
    NVVM approx intrinsics by the `-lower-gpu-math-to-nvvm` target peephole.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (block size over the feature dim); 0 = auto "
           "(largest power-of-two <= min(N,256) dividing N). The PK harness "
           "passes the library kernel's geometry.">,
  ];
}

def ConvertIntentContractionToGPU
    : Pass<"convert-intent-contraction-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.contraction matmul to one tiled-K "
                "tensor-core gpu.launch.";
  let description = [{
    Recognizes a `func.func` whose body is exactly one
    `intent.contraction{"mk,nk->mn"}` (A[M,K] f16, W[N,K] f16 -> Y[M,N] f32) and
    rewrites it in place to ONE tiled-K tensor-core `gpu.launch`: BM=BN=64,
    BK=16, a 4x4 grid of m16n16k16 WMMA tiles over 4 warps, with the per-warp
    fp32 accumulator fragments carried across the K loop as `scf.for` iter_args
    and the A/W sub-tiles staged through `gpu.dynamic_shared_memory`. The
    contraction is realized by NVIDIA WMMA (`gpu.subgroup_mma_*` ->
    wmma.mma.sync on sm_120a), NOT scalarized to a `linalg.generic` MAC.

    The B operand (W[N,K]) is loaded transposed so "mk,nk->mn" is exact:
    C[m,n] += sum_k A[m,k]*W[n,k]. Static shapes only, M%64==N%64==K%16==0
    (non-divisible tail-mask is a documented follow-up). Run BEFORE
    `convert-intent-to-linalg` (which would otherwise scalarize the contraction
    for the CPU/oracle leg). The resulting memref+gpu.launch module goes to the
    device pipeline (`lower_memref_module_to_kernels` with `scf_to_cf=True`, so
    the gpu->nvvm conversion can legalize an scf.for carrying !gpu.mma_matrix
    iter_args).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
}

#endif // INTENT_CONVERSION_PASSES
