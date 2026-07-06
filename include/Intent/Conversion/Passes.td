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

def ConvertIntentGatherToGPU
    : Pass<"convert-intent-gather-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.gather / intent.scatter (data-dependent "
                "addressing) to one block-cooperative gpu.launch.";
  let description = [{
    The C3 data-dependent-addressing pass. Recognizes a `func.func` whose body is
    exactly one `intent.gather` or one `intent.scatter` over rank-2 data producing
    a rank-2 result, and rewrites it in place to ONE block-cooperative
    `gpu.launch`: one block per output/scatter row, the free (column) axis split
    over the block's threads (T threads, K = cols/T columns each, the register
    tile unrolled over K). The new mechanism is a data-dependent address: the
    per-element load/store coordinate is read at runtime from the integer `index`
    memref (a `memref.load` of `index`, `arith.index_cast` to an index, then used
    as a `memref.load`/`memref.store` coordinate on `data`) — NOT an affine
    function of the iteration.

    * gather (`along_axis` = torch.gather, `take` = index_select/embedding
      row-gather): destination-passing — the appended `[Ro,Co]` output memref is
      written from the index-computed `data` load. `index` rank 1 (`take`) or 2
      (`along_axis`); `axis` selects which output coordinate the index replaces.
    * scatter (`along_axis`, reduce `none`/`add`): IN-PLACE on the (caller-cloned)
      `data` memref — one launch per `src`/`index` element does a data-dependent
      `memref.store` (reduce `none`) or `memref.atomic_rmw addf` (reduce `add` =
      scatter_add, -> `red.global.add.f32` on sm_120a). No output arg is appended.

    Correctness-only (no bank-conflict-free guarantee needed): the index stream is
    exact so verification is bit-faithful (rel < 1e-3). Static shapes, cols % T ==
    0 (tail-mask is a follow-up). Run BEFORE `convert-intent-to-linalg`; the
    resulting memref+gpu.launch module goes straight to the device pipeline
    (`lower_memref_module_to_kernels`), bypassing bufferization.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (block size over the free/column axis); 0 = auto "
           "(largest power-of-two <= min(cols,256) dividing cols).">,
  ];
}

def ConvertIntentLayoutToGPU
    : Pass<"convert-intent-layout-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single structural-layout op (concat/pad/tile/flip/triu) "
                "to one block-cooperative gpu.launch.";
  let description = [{
    The C4 structural-layout pass. Recognizes a `func.func` whose body is exactly
    one `intent.concat` / `intent.pad` / `intent.tile` / `intent.flip` /
    `intent.triu` over rank-2 operands producing a rank-2 result, and rewrites it
    in place to ONE block-cooperative `gpu.launch`: one block per OUTPUT row, the
    output column axis split over the block's threads (T threads, K = cols/T
    columns each, unrolled). Destination-passing: the appended `[Ro,Co]` output
    memref is written; the inputs are passed as memrefs in operand order.

    The new mechanism is pure DATA MOVEMENT — each output element's SOURCE
    coordinate is a STATIC structural function of the output coord (NOT the
    data-dependent runtime index of gather), then a `memref.load` (or the pad
    constant):
    * concat: piecewise by the cumulative axis offsets — the output coord is owned
      by one input; a per-input `scf.if(off <= coord < off+extent)` loads that
      input at the offset-shifted coord and stores.
    * pad: `scf.if(in-bounds)` yields `input[coord-lo]` else the `value` constant.
    * tile: `input[coord mod input_extent]` via `arith.remui` (no guard).
    * flip: `input[N-1-coord]` on the flipped axes (affine remap).
    * triu: `arith.select(col >= row + k, input[row,col], 0)`.

    Correctness-only (pure movement -> bit-faithful, rel < 1e-3). Static shapes,
    cols % T == 0 (tail-mask is a follow-up). Run BEFORE
    `convert-intent-to-linalg`; the memref+gpu.launch module goes straight to the
    device pipeline, bypassing bufferization.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (block size over the output column axis); 0 = auto "
           "(largest power-of-two <= min(cols,256) dividing cols).">,
  ];
}

def ConvertIntentRngToGPU
    : Pass<"convert-intent-rng-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a seeded-RNG intent func (a straight-line elementwise "
                "graph containing >=1 intent.rand) to one block-cooperative "
                "gpu.launch with an inline philox4x32-10 counter RNG.";
  let description = [{
    The C5 stateful-RNG pass. Recognizes a `func.func` whose body is a
    straight-line SCALAR-elementwise graph over rank-2 `[Ro,Co]` operands
    containing at least one `intent.rand` — the RNG family: `rand`/`rand_like`
    (one rand), `dropout` (`select(rand >= p, x/(1-p), 0)`), `randn`/`normal`
    (Box-Muller of two rands), `uniform(a,b)` (`a + (b-a)*rand`) — and rewrites
    it in place to ONE block-cooperative `gpu.launch`: one block per OUTPUT row,
    the column axis split over the block's threads (T threads, K = cols/T columns
    each, unrolled). Destination-passing: the appended `[Ro,Co]` output memref is
    written; any tensor inputs (e.g. dropout's `x`) are passed as memrefs.

    The new mechanism is a STATELESS counter-based RNG: each output element at
    `(row,col)` computes `philox4x32-10(seed, row*Co + col)` entirely in `arith`
    (10 rounds of integer mul-hi/xor over the public philox constants; the seed
    splits into the two key words, the element index is the counter), maps the
    first output word to a 24-bit-mantissa uniform[0,1) f32, then runs the
    composed scalar transform (map_unary / map_binary incl. compare / select /
    const, reusing the C0/M3 scalar emitters) and stores. No global RNG state, no
    visited-order dependence — same seed -> bit-identical, different seed ->
    decorrelated.

    Verification is matched-seed against the same philox in the interpreter
    (determinism / self-consistency) and STATISTICAL against torch/flag_gems
    (uniform mean~0.5 var~1/12, normal mean~0 var~1, dropout keeps ~(1-p)) — the
    philox counter layout does NOT match Triton's `tl.rand` bit-for-bit, so
    library parity is distributional, not byte-exact. Static shapes, cols % T ==
    0 (tail-mask is a follow-up). Run BEFORE `convert-intent-to-linalg`; the
    memref+gpu.launch module goes straight to the device pipeline.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (block size over the output column axis); 0 = auto "
           "(largest power-of-two <= min(cols,256) dividing cols).">,
  ];
}

def ConvertIntentArgReduceToGPU
    : Pass<"convert-intent-arg-reduce-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower an argmax/argmin intent func (one intent.arg_reduce over "
                "the last axis) to one block-cooperative gpu.launch returning the "
                "extremum's INDEX.";
  let description = [{
    The C6 order-statistics arg-reduce pass. Recognizes a `func.func` whose body
    is a single `intent.arg_reduce` (`argmax`/`argmin`) over a rank-2 `[M,N]`
    static-f32 row (`axis = 1`), producing the integer INDEX of the row extremum
    (`[M]` or keepdims `[M,1]` i32). It lowers to ONE block-cooperative
    `gpu.launch`: one block per row, the feature axis split over T threads
    (K = N/T contiguous columns each). Each thread scans its K columns for a local
    `(best_val, best_idx)` pair (ascending columns + a STRICT compare so ties keep
    the lowest column, matching torch's first-index tie-break); a power-of-two
    shared-memory tree then reduces the T per-thread pairs keeping the extremum
    (ties -> the lower index). Thread 0 stores the winning integer index.

    The new mechanism vs. `intent.reduce` (value-only, one `gpu.all_reduce`) is
    the PAIR carried through the reduction — an argmax/argmin tree over
    `(val, idx)` in two dynamic-shared buffers — and an INTEGER output. Verified
    exact-index vs torch.argmax/argmin and the real flag_gems kernel. Static
    shapes, N % T == 0, T a power of two.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block over the feature axis; 0 = auto.">,
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
    Option<"blockM", "block-m", "unsigned", /*default=*/"64",
           "block C-tile rows BM (numWarps = BM/16 warps tile the M axis, one "
           "16-row strip each). Falls back to 64 for shapes M%BM!=0.">,
    Option<"blockN", "block-n", "unsigned", /*default=*/"128",
           "block C-tile cols BN (each warp owns BN/16 column fragments; wider "
           "BN raises A-load reuse). Falls back to 64 for shapes N%BN!=0.">,
  ];
}

def ConvertIntentContractionToGPUMMA
    : Pass<"convert-intent-contraction-to-gpu-mma", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.contraction matmul to one tiled-K "
                "tensor-core gpu.launch via nvgpu.ldmatrix + nvgpu.mma.sync.";
  let description = [{
    The `nvgpu.mma.sync` sibling of `convert-intent-contraction-to-gpu` (which
    uses the opaque `gpu.subgroup_mma_*` / `!gpu.mma_matrix` WMMA path). Same
    recognizer (one `intent.contraction{"mk,nk->mn"}`, A[M,K] f16, W[N,K] f16 ->
    Y[M,N] f32, optional bias+activation epilogue) and the SAME cp.async
    multistage staging, but the contraction is realized by the register-level
    `nvgpu.ldmatrix` (shared->register warp-cooperative fragment load) +
    `nvgpu.mma.sync` (m16n8k16 warp MMA on `vector<...>` fragments with a KNOWN
    lane layout -> `vector<2x2xf32>` accumulator) primitive, NOT the opaque
    fragment. This is the parallel path explored for full compute-bound parity;
    the `gpu.mma_matrix` pass is left 100% intact as the default / fallback.

    Fragment recipe (validated against the MLIR --convert-vector-to-gpu
    use-nvgpu lowering on sm_120a): lane = tid%32, warp = tid/32, each warp owns
    a 16-row strip; A operand `ldmatrix aSh[warp*16 + lane%16, kOff + (lane/16)*8]`
    transpose=false -> vector<4x2xf16>; W operand (natural [N,K] layout, the
    "nk" transpose folded into the load) `ldmatrix wSh[n*8 + lane%8, kOff +
    (lane/8)*8]` transpose=false -> vector<2x2xf16>; the m16n8 f32 accumulator
    stores back with the textbook layout d[i][j] -> row = lane/4 + i*8,
    col = (lane%4)*2 + j. Per-warp accumulator vectors ride the K loop as
    `scf.for` iter_args. Static shapes, M%BM==N%BN==K%BK==0. Run BEFORE
    `convert-intent-to-linalg`; the result goes to the device pipeline
    (`lower_memref_module_to_kernels` with `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"3",
           "cp.async multistage pipeline depth (N buffers; clamped to the "
           "K-tile count). N=1 disables prefetch overlap.">,
    Option<"blockM", "block-m", "unsigned", /*default=*/"64",
           "block C-tile rows BM (numWarps = BM/16 warps tile the M axis).">,
    Option<"blockN", "block-n", "unsigned", /*default=*/"128",
           "block C-tile cols BN (each warp owns BN/8 n8 mma column tiles).">,
  ];
}

def ConvertIntentContractionToGPUMMAWS
    : Pass<"convert-intent-contraction-to-gpu-mma-ws", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.contraction matmul to a WARP-SPECIALIZED "
                "(producer/consumer) tiled-K nvgpu.mma.sync gpu.launch.";
  let description = [{
    M7g: the warp-specialized sibling of `convert-intent-contraction-to-gpu-mma`.
    Same recognizer and the same register-level `nvgpu.ldmatrix` +
    `nvgpu.mma.sync` (m16n8k16) contraction, but the block's warps are split into
    PRODUCER warps (the high warp ids -- loop the K-tiles issuing cp.async loads
    into an N-stage shared buffer, signal a per-stage `full` mbarrier, wait the
    matching `empty` mbarrier) and CONSUMER warps (the low warp ids, each owning
    a 16-row C strip as in M7f -- wait `full`, run ldmatrix + mma.sync into the
    register accumulator carried as scf.for iter_args, signal `empty`). The
    handshake uses the nvgpu.mbarrier primitives (init/arrive/try_wait.parity),
    which lower to mbarrier.*.shared.b64 on sm_120a. Phase parity is computed
    directly from the loop index ((ki/nStages)%2), no phase iter_args. The M7f
    pass is left 100% intact; this is gated by its own name. Static shapes,
    M%BM==N%BN==K%BK==0. Run BEFORE `convert-intent-to-linalg`; the result goes
    to the device pipeline (`lower_memref_module_to_kernels`, `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"4",
           "cp.async multistage pipeline depth (N shared buffers; clamped to "
           "the K-tile count).">,
    Option<"blockM", "block-m", "unsigned", /*default=*/"64",
           "block C-tile rows BM (numConsumerWarps = BM/16).">,
    Option<"blockN", "block-n", "unsigned", /*default=*/"64",
           "block C-tile cols BN (each consumer warp owns BN/8 n8 tiles).">,
    Option<"numProducerWarps", "num-producer-warps", "unsigned",
           /*default=*/"1",
           "warps dedicated to issuing cp.async loads (the rest are consumers).">,
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
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"1",
           "cp.async pipeline depth for the K/V key-tile loop (N rotating "
           "dynamic-shared K/V buffers; with N>1 the global->shared loads of "
           "tiles ki+1..ki+N-1 prefetch ahead of the WMMA+softmax compute of "
           "tile ki). Effective N is clamped to the K-tile count. DEFAULT 1: the "
           "win over M8a's scalar cooperative copy comes from the *vectorized* "
           "16-byte cp.async.cg staging (N=1, single buffer, ~30% faster), NOT "
           "from multistage prefetch -- this kernel is softmax/compute-bound, so "
           "N>=2 only adds async-copy + wait/barrier overhead and (D=64) grows "
           "shared 82688->99072 B (less L1), measurably REGRESSING latency. "
           "N=2 is the only depth >1 that still fits the 101376 B optin budget "
           "for D=64 (each extra stage adds BN*D*2*2 = 16384 B).">,
  ];
}

def ConvertIntentFlashMMAToGPU
    : Pass<"convert-intent-flash-mma-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower the fused FlashAttention-forward intent.loop to one "
                "tensor-core gpu.launch with the online-softmax state "
                "REGISTER-RESIDENT via nvgpu.mma.sync (the M8c sibling of "
                "convert-intent-loop-contraction-to-gpu).";
  let description = [{
    The register-resident `nvgpu.mma.sync` sibling of
    `convert-intent-loop-contraction-to-gpu` (M8a, which round-trips the
    online-softmax state O/S/P/m/l through dynamic shared). Same recognizer (one
    `intent.loop` scanning the key axis: carries `m[Sq,1]`, `l[Sq,1]`,
    `O[Sq,D]`; body = `S=Q@Kj^T` -> online softmax -> `O+=P@Vj`), but the
    rewrite keeps O/S/m/l RESIDENT IN REGISTERS so only Q/K/V touch shared:

    Per block (BM=BN=64, 4 warps, each warp a 16-row query strip):
      S = Q@Kj^T via nvgpu.ldmatrix + nvgpu.mma.sync (m16n8k16 f16->f32). The S
      accumulators are per-warp `vector<2x2xf32>` fragments with the KNOWN PTX
      lane layout `d[i][j] -> row = lane/4 + i*8, col = 8*tile + (lane%4)*2 + j`.
      Each lane therefore owns 2 query rows (lane/4 and lane/4+8); a row's BN key
      columns are spread across that lane (over the n8 tiles and j) and the 3
      other lanes of its quad (lane%4). Per-row rowmax/rowsum are computed with a
      quad `gpu.shuffle xor` (delta 1 then 2, width 32) -- NO shared round-trip.
      alpha = exp(m_old-m_new) per row; P = exp(scale*S-m_new) cast f16, all in
      registers. The P A-operand fragment for the PV matmul is assembled DIRECTLY
      from the S accumulators (the m16n8 acc layout == the m16n8k16 A-operand
      fragment layout, two consecutive n8 S-tiles -> one k16 A-subtile, NO
      shuffle, NO shared P). O accumulators (`vector<2x2xf32>`, carried across the
      key loop as `scf.for` iter_args) are rescaled by alpha (per-row scalar
      already lane-local) then `O += P@Vj` via mma.sync (V staged transposed in
      shared as the [D,Sk] B operand). Epilogue: `O/l` straight from registers to
      global.

    Q/K/V f16, accumulators f32. Static shapes, BM=BN=64, D%16==0, Sq%64==0,
    Sk%64==0, non-causal, rank-2 (no batch/LSE/causal -- those stay on the M8a
    path). Run BEFORE `convert-intent-to-linalg`; the result goes to the device
    pipeline (`lower_memref_module_to_kernels` with `scf_to_cf=True`, the scf.for
    carries vector iter_args). M8a is left 100% intact (separate pass).
  }];
  let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"2",
           "cp.async software-pipeline depth for the K/V key-tile loop "
           "(N rotating dynamic-shared K/V buffers; clamped to the K-tile "
           "count). 1 = single-buffer (no multistage). N=2 hides the K/V "
           "global-load latency behind the mma.sync+softmax of the previous "
           "tile (~1.2x at >=512 keys); N>=3 plateaus and only wastes smem.">,
  ];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::vector::VectorDialect",
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
  let options = [
    Option<"dqMode", "dq-mode", "std::string", /*default=*/"\"serial\"",
           "DQ accumulation mode. 'serial' (SEQUENCE_PARALLEL=false): ONE block "
           "per (b,h) owns the whole head, an OUTER scf.for over key tiles "
           "read-modify-writes the pre-zeroed global DQ deterministically. "
           "'atomic' (SEQUENCE_PARALLEL=true): grid parallel over key blocks "
           "(grid.x = cdiv(Sk,BN)); each program owns ONE key block and "
           "accumulates its dQ contribution into pre-zeroed global DQ via "
           "memref.atomic_rmw addf (-> red.global.add.f32 on sm_120a), unlocking "
           "parallelism over key blocks. Both modes pre-zero DQ; they compute the "
           "same value modulo the float reassociation of the cross-program adds "
           "(the serial==atomic equivalence gate).">,
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

def ConvertIntentSplitKReduceBackwardToGPU
    : Pass<"convert-intent-splitk-reduce-backward-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower the split-K attention reduce BACKWARD (gradient of the "
                "online-LSE merge, no matmul) to one block-cooperative "
                "gpu.launch.";
  let description = [{
    Recognizes a `func.func` whose body is the xformers
    `_splitK_reduce_varargs_backward` -- the GRADIENT of the online log-sum-exp
    merge -- as a straight-line map/reduce DAG over the packed split-K partials
    (Out_splitK[S,Mq,D], LSE_splitK[S,Mq,1]) + the merged-output gradients
    (Out[Mq,D], LSE[Mq,1], DOut[Mq,D], DLSE[Mq,1]), producing the per-split
    gradients dOut_splitK[S,Mq,D], dLSE_splitK[S,Mq,1] (the varargs list is
    normalized to this packed RowList upstream, the SAME contract as the forward
    reduce). The DAG has exactly two `intent.reduce` -- a `max` over the split
    axis (lse_max) and a `sum` over the head dim (the corr term) -- and broadcast
    map ops; NO loop, NO contraction. The match is STRUCTURAL (shapes + the two
    reduce signatures), NOT a kernel-name match.

    It rewrites the func in place to ONE block-cooperative `gpu.launch`:
    grid = (Mq,1,1) (one block per output row), block = (D,1,1) (one thread per
    head-dim element). Two passes over the split axis (`scf.for`): pass 1
    computes `lse_max = max_s LSE_splitK[s]` (every thread recomputes the per-row
    scalar); `denom = exp(LSE - lse_max)`. pass 2 per split s computes
    `w_s = exp(LSE_splitK[s] - lse_max)/denom`, stores `dOut_splitK[s] = w_s*DOut`,
    and `dLSE_splitK[s] = w_s*DLSE + sum_d((Out_splitK[s]-Out)*w_s*DOut)`, where
    the head-dim sum is a `gpu.all_reduce` ADD across the block's D threads
    (inside the split-axis loop). dLSE_splitK is written once by thread 0.
    Memory-bound (read partials once, write per-split gradients once) -> parity is
    the honest floor. Run BEFORE `convert-intent-to-linalg`; the result goes to
    the device pipeline (`lower_memref_module_to_kernels`, `scf_to_cf=True`).
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
