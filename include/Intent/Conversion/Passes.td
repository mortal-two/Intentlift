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
    BK=32, a 4x4 grid of m16n16k16 WMMA tiles over 4 warps, with the per-warp
    fp32 accumulator fragments carried across the K loop as `scf.for` iter_args
    and the A/W sub-tiles staged through `gpu.dynamic_shared_memory`. The
    contraction is realized by NVIDIA WMMA (`gpu.subgroup_mma_*` ->
    wmma.mma.sync on sm_120a), NOT scalarized to a `linalg.generic` MAC.

    The B operand (W[N,K]) is loaded transposed so "mk,nk->mn" is exact:
    C[m,n] += sum_k A[m,k]*W[n,k]. Static shapes only, M%64==N%64==K%32==0
    (non-divisible tail-mask is a documented follow-up). Run BEFORE
    `convert-intent-to-linalg` (which would otherwise scalarize the contraction
    for the CPU/oracle leg). The resulting memref+gpu.launch module goes to the
    device pipeline (`lower_memref_module_to_kernels` with `scf_to_cf=True`, so
    the gpu->nvvm conversion can legalize an scf.for carrying !gpu.mma_matrix
    iter_args).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"3",
           "cp.async multistage pipeline depth (N buffers; effective N is "
           "clamped to the K-tile count). N=1 disables prefetch overlap.">,
  ];
}

def ConvertIntentLoopContractionToGPU
    : Pass<"convert-intent-loop-contraction-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a fused FlashAttention-forward intent.loop (online "
                "softmax + two contractions) to one tensor-core gpu.launch.";
  let description = [{
    Recognizes a `func.func` whose body is the FlashAttention forward as ONE
    `intent.loop` scanning the key axis: carries (running max `m[Sq,1]`, running
    sum `l[Sq,1]`, accumulator `O[Sq,D]`); the body computes `S = Q@Kj^T`
    (`intent.contraction{"qd,kd->qk"}`), a row-wise online softmax
    (`reduce{max}` -> `exp` -> `reduce{sum}`), and `O += P@Vj`
    (`intent.contraction{"qk,kd->qd"}`). It rewrites the func in place to ONE
    tiled tensor-core `gpu.launch`.

    The fusion SIDESTEPS the opaque-fragment rowmax/rescale problem by keeping
    the online-softmax state RESIDENT IN DYNAMIC SHARED MEMORY (S, P, O, m, l,
    alpha tiles) and the two matmuls on tensor cores: each K-tile does
    `S = Q@Kj^T` via WMMA (`gpu.subgroup_mma_*`) stored to a shared `[BM,BN]`
    f32 tile, a COOPERATIVE row-wise online softmax over that shared tile (per
    row: m_new = max(m, rowmax(scale*S)), alpha = exp(m-m_new), P = exp(S-m_new)
    cast to f16, l = l*alpha + rowsum), then `PV = P@Vj` via WMMA into a shared
    f32 tile and the cooperative update `O = O*alpha + PV`. The per-row alpha
    rescale and the rowmax/rowsum are plain cooperative shared-memory arithmetic
    over the BM rows -- NOT fragment-lane indexing. The scf.for over key tiles
    carries no iter_args (m/l/O live in shared, made visible across trips by
    `gpu.barrier`). Epilogue: `O/l` to global.

    Q/K/V are f16 (WMMA operands), the accumulator/softmax state is f32. Static
    shapes, BM=BN=64, D%16==0, Sq%64==0, Sk%64==0 (non-causal, no bias; tail-mask
    / causal / split-K are documented follow-ups). Run BEFORE
    `convert-intent-to-linalg`. The resulting memref+gpu.launch module goes to the
    device pipeline (`lower_memref_module_to_kernels` with `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
}

def ConvertIntentAttentionBackwardToGPU
    : Pass<"convert-intent-attention-backward-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower the fused FlashAttention BACKWARD (one intent.loop over "
                "the query axis carrying dk/dv + 5 contractions + recompute-P) "
                "to one tensor-core gpu.launch (serial-DQ).";
  let description = [{
    Recognizes a `func.func` whose body is the FlashAttention backward as ONE
    `intent.loop` scanning the QUERY axis (scan inputs Q, dO, LSE, Delta; carries
    the per-key-row gradients `dk[Sk,D]`, `dv[Sk,D]`, init zeros). The body is the
    flash bwd inner block: recompute `P = exp(scale*S - LSE)` from the saved
    log-sum-exp, then FIVE contractions

        S  = Q@K^T        (`"qd,kd->qk"`)        recompute the scores
        dV += P^T@dO      (`"qk,qd->kd"`)        accumulate into the dv carry
        dP =  dO@V^T      (`"qd,kd->qk"`)
        dS =  P*(dP - Delta)*scale               (elementwise)
        dK += dS^T@Q      (`"qk,qd->kd"`)        accumulate into the dk carry
        dQ =  dS@K        (`"qk,kd->qd"`)        scatter-accumulate into DQ[start_m]

    The match is STRUCTURAL (the 5 contraction subscripts + the recompute-P
    subtree + the carry shapes), NOT a kernel-name match -- it just happens to be
    attention-backward-shaped (hence the pass name).

    It rewrites the func in place to ONE tiled tensor-core `gpu.launch`, modeled
    on `convert-intent-loop-contraction-to-gpu` (M8 flash fwd): the matmuls run on
    WMMA (`gpu.subgroup_mma_*`, reusing the M7 tiled-K primitive), while the
    recompute-P / dS elementwise transforms and the dk/dv accumulators live
    RESIDENT IN DYNAMIC SHARED MEMORY (the M8 cooperative-shared design). The
    schedule is ONE block per (b,h): an OUTER `scf.for` over key tiles (BN=64,
    dk/dv reset per key tile in shared) wrapping an INNER `scf.for` over query
    tiles (BM=64, the intent.loop's scan). dQ uses the SERIAL accumulation mode
    (SEQUENCE_PARALLEL=false): one block owns the whole head and read-modify-
    writes the pre-zeroed global `DQ` deterministically across the outer key loop
    (the cross-program atomic_add mode + its serial==atomic gate are M9b).

    Q/K/V/dO are f16 (WMMA operands), LSE/Delta/scale and all accumulators are
    f32. The `dV=P^T@dO` and `dK=dS^T@Q` contractions reduce over the query axis,
    so their `[q,*]` operands are loaded TRANSPOSED on the WMMA A operand (probed:
    AOp-transpose lowers through `wmma.load.a...col.f16` on sm_120a). Static
    shapes, BM=BN=64, D%16==0, Sq%64==0, Sk%64==0 (non-causal, no bias; tail-mask
    / causal / atomic-DQ / split-K are documented follow-ups). The pass appends
    three destination-passing outputs `dQ`, `dK`, `dV` (host pre-zeros DQ). Run
    BEFORE `convert-intent-to-linalg`; the result goes to the device pipeline
    (`lower_memref_module_to_kernels` with `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
}

def ConvertIntentSplitKReduceToGPU
    : Pass<"convert-intent-splitk-reduce-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower the split-K attention reduce (online-LSE merge across "
                "chunks, no matmul) to one block-cooperative gpu.launch.";
  let description = [{
    Recognizes a `func.func` whose body is the xformers `_splitK_reduce` /
    `_splitK_reduce_varargs` merge as ONE `intent.loop` scanning the split axis
    of a packed partial-output `O[S,Mq,D]` and partial log-sum-exp
    `LSE[S,Mq,1]` (the varargs list is normalized to this packed form upstream,
    so both reduce variants lift to the SAME IntentIR). The loop carries the
    online-merge state (`lse_max[Mq,1]`, `sumexp[Mq,1]`, `O_acc[Mq,D]`); the body
    is a pure elementwise/broadcast online update (NO contraction, NO reduce):
    `m_new = max(lse_max, LSE_s)`, `corr = exp(lse_max - m_new)`,
    `w = exp(LSE_s - m_new)`, `sumexp = sumexp*corr + w`,
    `O_acc = O_acc*corr + w*O_s`. Epilogue: `O = O_acc / sumexp`.

    Because `lse_max`/`sumexp` depend only on the per-(row) LSE scalars, every
    thread can recompute them redundantly with no shared memory or barriers: the
    rewrite emits grid = (Mq, 1, 1) (one block per output row) with one thread
    per head-dim element owning a single `O_acc[d]` register, scanning the split
    axis with a plain `scf.for`. Memory-bound (read S partials once, write O
    once) -> parity is the honest floor. Run BEFORE `convert-intent-to-linalg`;
    the result goes to the device pipeline (`lower_memref_module_to_kernels`,
    `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
}

#endif // INTENT_CONVERSION_PASSES
