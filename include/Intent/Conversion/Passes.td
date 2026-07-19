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
    over the block's threads, each thread reading its columns **once** into an
    SSA register tile, compact warp/shared reduction groups, per-row scalars
    hoisted once, and the epilogue writing the output from the register tile
    (read-once = HBM parity with the library kernel). This is the verified block-cooperative
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
    "::mlir::vector::VectorDialect",
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
    Option<"vectorWidth", "vector-width", "unsigned", "0",
           "Register-path contiguous memory width: 0 = structurally selected "
           "auto, 1 = force scalar coalesced loads/stores, 4 = require legal "
           "vector<4xf32> Row/Col loads and Row stores. This is a Layer-C "
           "candidate axis; unsupported forced-vector graphs fail closed.">,
    Option<"fuseFma", "fuse-fma", "bool", "false",
           "Emit explicit f32 math.fma for single-use mul+add epilogues and "
           "sum-of-products reduction accumulation. Disabled by default; an "
           "offline Layer-C candidate may enable it and must pass P6 before "
           "performance selection.">,
    Option<"deferEpilogueColLoads", "defer-epilogue-col-loads", "bool", "true",
           "Delay epilogue-only contiguous Col loads until after all row "
           "collectives, reducing cross-barrier register pressure. Set false "
           "to preload them before the reduction so small register-resident "
           "rows can overlap global-memory latency with the collective. This "
           "is a Layer-C candidate and must independently pass P6.">,
    Option<"enableSortedPrefix", "enable-sorted-prefix", "bool", "true",
           "Enable the B-certified descending sparsemax early-stop fast path. "
           "Set false to force the general scan/reduction implementation for "
           "the same Intent graph, providing an independently executable "
           "generic correctness fallback.">,
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
    `gpu.launch`: normally one block per row, the feature dim split over the
    block's threads, each thread reading its columns **once** into SSA registers
    (read-once = HBM parity with the library kernel), per-row scalars hoisted
    once, and a multi-output epilogue writing each result from the register tile.
    Contiguous strips use vector<4xf32> when divisible by four, otherwise
    vector<2xf32> when divisible by two; scalar lowering remains the fallback.
    A fail-closed `flatten-contiguous` schedule is available when every tensor
    input/result is an equal-shape contiguous Row value and there are no
    Col/broadcast/indexed views: it maps the complete tensor as one linear vec4
    domain, reducing block count without changing the normalized op DAG.
    There is NO `gpu.all_reduce` — nothing is reduced. General over the map DAG
    (NO per-kernel recognizer); the two passes' match() predicates are disjoint
    (this pass returns failure on ANY reduce; row-reduction returns failure with
    no reduce), so they compose in either order.

    Static shapes only. The resulting memref+gpu.launch module goes straight to
    the device pipeline (`lower_memref_module_to_kernels`), bypassing
    bufferization. Transcendentals (`math.exp`/`math.tanh`/...) are rewritten to
    NVVM approx intrinsics by the `-lower-gpu-math-to-nvvm` target peephole.
    Rank-2 f16/bf16 results and Row inputs use half storage with per-map f32
    computation (`extf` before the scalar op, `truncf` at that op's declared
    result boundary), preserving typed intermediate rounding. Mixed f32
    Col/Scalar broadcasts are admitted without changing the result storage type.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (block size over the feature dim); 0 = auto "
           "(largest power-of-two <= min(N,256) dividing N). The PK harness "
           "passes the library kernel's geometry.">,
    Option<"flattenContiguous", "flatten-contiguous", "bool", "false",
           "Map an admitted equal-shape contiguous Row-only graph over one "
           "linear vec4 domain instead of launching one block per row.">,
    Option<"flatVectorWidth", "flat-vector-width", "unsigned", "4",
           "Vector width for flatten-contiguous (2 or 4). Ignored by the "
           "default row schedule.">,
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

def ConvertIntentSortToGPU
    : Pass<"convert-intent-sort-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower an intent.sort/topk/argsort func (one intent.sort over the "
                "last axis) to one block-cooperative bitonic-sort gpu.launch.";
  let description = [{
    The C6 HARD-tier order-statistics pass — the last order-stats gap. Recognizes
    a `func.func` whose body is a single `intent.sort` over a rank-2 `[M,N]`
    static-f32 row (`axis = 1`) producing sorted `values` `[M,W]` f32 and/or
    argsort `indices` `[M,W]` i32 (`W <= N`; `W < N` = topk). It lowers to ONE
    block-cooperative BLOCK BITONIC SORT `gpu.launch`: one block per row, `T`
    threads.

    Each block loads its row into a `[P]`-element dynamic-shared value buffer,
    padding `N -> P` (the next power of two) with a `+inf` (ascending) / `-inf`
    (descending) sentinel so the pad sorts to the END. It then runs the bitonic
    network: for `k = 2,4,...,P` (outer) and `j = k/2,...,1` (inner),
    `log2(P)*(log2(P)+1)/2` stages, each a `gpu.barrier` then a grid-stride sweep
    where the thread owning the lower index of each pair `(i, i^j)` compare-swaps
    under the direction bit `(i & k)==0` (inverted for `descending`). After the
    network thread `t` writes `shared[t] -> values[row,t]` for the first `W`. For
    the argsort `indices` an `i32` index buffer (init `idx[i]=i`) is swapped in
    lockstep and written the same way.

    The new mechanism vs. the value-only reductions/scans is the O(N log^2 N)
    barrier-separated compare-exchange NETWORK in shared memory (not a single
    tree/scan). Correctness-only: values exact vs torch.sort/topk + flag_gems; the
    index permutation reproduces the sorted values (ties unstable, like torch).
    Static shapes, `axis` last, `T` a power of two dividing `P`.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::func::FuncDialect",
    "::mlir::math::MathDialect",
    "::mlir::NVVM::NVVMDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (power of two dividing P=next_pow2(N)); 0 = auto.">,
  ];
}

def ConvertIntentPoolToGPU
    : Pass<"convert-intent-pool-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.pool (avg / max[+indices]) over NCHW to "
                "one block-cooperative gpu.launch (windowed reduce).";
  let description = [{
    The C7 windowed-reduce pass. Recognizes a `func.func` whose body is exactly
    one `intent.pool` over a rank-4 `[N,C,H,W]` f32 input producing `[N,C,OH,OW]`
    (avg) or `[N,C,OH,OW]` + an i64 argmax-index `[N,C,OH,OW]` (max_pool2d_with_
    indices), and rewrites it in place to ONE block-cooperative `gpu.launch`:
    grid = (N*C, OH, 1) (one block per channel-plane row of the output), block =
    (T,1,1) threads over the OW output columns (K = OW/T columns each, unrolled).

    The new mechanism vs. `intent.reduce` (a full-axis tree/all_reduce) is a
    BOUNDED per-output-cell reduction over the `kH x kW` window: each thread walks
    the static window `(ih,iw) = (oh*sH - padH + kh, ow*sW - padW + kw)`, guards
    each source coord with an in-bounds `scf.if` (out-of-bounds = skip), and
    accumulates. `avg` sums (padded cells contribute 0) and divides by `kH*kW`
    (torch's default count_include_pad). `max` keeps the running max and — when the
    `indices` output is present — the argmax FLAT index `ih*W + iw` carried as an
    i64 PAIR alongside the value (strict `>` update = first-index tie-break,
    matching torch/flag_gems). Destination-passing: the appended output memref(s)
    (`[N,C,OH,OW]` f32, then the i64 index memref for max-with-indices) are
    written; the input is passed as a rank-4 memref.

    Correctness-only (bit-faithful vs torch + flag_gems: rel < 1e-3 values,
    exact indices). Static shapes, OW % T == 0. Run BEFORE
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
           "Threads per block over the OW output-column axis; 0 = auto.">,
  ];
}

def ConvertIntentUpsampleToGPU
    : Pass<"convert-intent-upsample-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.upsample (nearest) over NCHW to one "
                "block-cooperative gpu.launch (affine floor index-remap).";
  let description = [{
    The C7 interpolation pass. Recognizes a `func.func` whose body is exactly one
    `intent.upsample{mode="nearest"}` over a rank-4 `[N,C,H,W]` f32 input producing
    `[N,C,OH,OW]`, and rewrites it in place to ONE block-cooperative `gpu.launch`:
    grid = (N*C, OH, 1) (one block per output channel-plane row), block = (T,1,1)
    threads over the OW output columns (K = OW/T each, unrolled).

    Pure DATA MOVEMENT (C4-layout-like): each output element's SOURCE coord is an
    affine `floor` of the output coord — `ih = min(floor(oh*IH/OH), IH-1)`,
    `iw = min(floor(ow*IW/OW), IW-1)` computed with the reciprocal scale `IH/OH`
    in f32 then truncated to int (matching flag_gems' `(oh*IH/OH).to(int32)`) —
    followed by a `memref.load` at `(n,c,ih,iw)`. Destination-passing: the appended
    `[N,C,OH,OW]` output memref is written; the input is a rank-4 memref.

    Correctness-only (bit-EXACT gather vs torch + flag_gems). Static shapes,
    OW % T == 0. Run BEFORE `convert-intent-to-linalg`.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block over the OW output-column axis; 0 = auto.">,
  ];
}

def ConvertIntentPolarToGPU
    : Pass<"convert-intent-polar-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.polar (abs,angle -> real,imag) to one "
                "block-cooperative gpu.launch (elementwise complex build).";
  let description = [{
    The C7 complex-dtype pass. Recognizes a `func.func` whose body is exactly one
    `intent.polar` over rank-2 `[M,N]` f32 `abs`/`angle` producing two `[M,N]` f32
    results `real`/`imag`, and rewrites it in place to ONE block-cooperative
    `gpu.launch`: one block per row, the column axis split over T threads
    (K = N/T columns each, unrolled). Destination-passing: two appended `[M,N]`
    output memrefs (`real` then `imag`) are written.

    Elementwise: `real = abs*cos(angle)`, `imag = abs*sin(angle)`, reusing the C0
    `cos`/`sin` scalar emitters (`math.cos`/`math.sin` -> `cos.approx`/`sin.approx`
    NVVM hardware ops via the `-lower-gpu-math-to-nvvm` peephole). The complex
    tensor is represented as the trailing real/imag PAIR (two real results),
    compared against `torch.view_as_real(torch.polar(...))` (rel < 1e-3). Static
    shapes, N % T == 0. Run BEFORE `convert-intent-to-linalg`.
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
           "Threads per block over the column axis; 0 = auto.">,
  ];
}

def ConvertIntentConvToGPU
    : Pass<"convert-intent-conv-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.conv (conv1d/2d/3d, forward) over "
                "NC*spatial to one block-cooperative gpu.launch (direct "
                "windowed-weighted accumulation).";
  let description = [{
    The C7 convolution pass. Recognizes a `func.func` whose body is exactly one
    `intent.conv` over a rank `2+R` (`R` = 1/2/3) f32 input `[N,Cin,*spatial]`
    with a `[Cout,Cin/groups,*kernel]` weight (+ optional `[Cout]` bias)
    producing `[N,Cout,*out_spatial]`, and rewrites it in place to ONE
    block-cooperative `gpu.launch`: grid = (N*Cout, prod(outer out-spatial), 1)
    (one block per output row = fixed `(n,co,*outer)`), block = (T,1,1) threads
    over the innermost out-spatial axis (K = O_last/T columns each, unrolled).

    The mechanism EXTENDS the `intent.pool` windowed per-output-cell reduce with
    an input-channel sum and a per-tap weight multiply (direct
    windowed-weighted-accumulation, no im2col): for output cell `(n,co,*o)`,
    `acc = bias[co] + sum_{ci_local} sum_{*k} in[n, g*(Cin/groups)+ci_local,
    *(o*stride - pad + k*dil)] * w[co, ci_local, *k]`, guarding every source
    spatial coord with a single in-bounds `scf.if` (out-of-bounds = zero pad).
    Channel groups pick the input-channel slice via `g = co/(Cout/groups)`. The
    accumulation core is parameterized by the spatial rank `R`, so conv1d (R=1),
    conv2d (R=2) and conv3d (R=3) share one code path (extra kernel/window
    loops). Destination-passing: the new signature is
    `(input, weight[, bias]) ++ [output]` memrefs.

    Correctness-only (rel < 1e-3 vs torch + flag_gems; fp accumulation order).
    Static shapes, O_last % T == 0. Run BEFORE `convert-intent-to-linalg`.
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
           "Threads per block over the innermost out-spatial axis; 0 = auto.">,
  ];
}

def ConvertIntentConvToGPUMMA
    : Pass<"convert-intent-conv-to-gpu-mma", "::mlir::ModuleOp"> {
  let summary = "Lower a single conv2d intent.conv as an IMPLICIT GEMM onto the "
                "nvgpu.ldmatrix + nvgpu.mma.sync (m16n8k16) tensor-core path.";
  let description = [{
    The tensor-core sibling of `convert-intent-conv-to-gpu` (the naive direct
    windowed f32-accumulation path). Recognizes a `func.func` whose body is
    exactly one forward conv2d `intent.conv` (rank-4 `[N,Cin,H,W]` f32 input,
    `[Cout,Cin,kH,kW]` weight, optional `[Cout]` bias, `groups == 1`) and rewrites
    it as an IMPLICIT GEMM `Y = im2col(X) . W^T`:

      M = N*OH*OW (output pixels)   N_gemm = Cout   K = Cin*kH*kW

    The im2col matrix is NEVER materialized: each `(m-row, k-col)` of the A shared
    tile is mapped back to an input pixel by the SAME OOB-guarded conv address
    math as the naive pass (`ih = oh*stride - pad + kh*dil`, out-of-bounds = zero
    pad) emitted as the smem tile-fill load predicate. W is already a real
    `[Cout, Cin*kH*kW]` row-major matrix so its tile-fill is a plain gather. Both
    operands are cast f32 -> f16 during the fill (fp16 input mode: rel ~1e-2 vs
    fp32 torch, matching the TF32 cuDNN reference). The register-level contraction
    (`nvgpu.ldmatrix` + `nvgpu.mma.sync` m16n8k16 -> per-warp `vector<2x2xf32>`
    accumulators carried across the K loop as `scf.for` iter_args) and the padded
    f16 shared tile geometry are reused verbatim from
    `convert-intent-contraction-to-gpu-mma`; the store scatters each accumulator
    element back to the NCHW output (`row -> (n,oh,ow)`, `col -> co`) with an
    optional bias add.

    stride/padding/dilation are fully general (folded into the address math).
    Static shapes, `groups == 1`, `K % 32 == 0` (else the pass declines and the
    naive `convert-intent-conv-to-gpu` fallback handles it); M and N are padded
    via a ceil grid + guarded store. Bare-ptr ABI; the result goes to the device
    pipeline (`lower_memref_module_to_kernels` with `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockM", "block-m", "unsigned", /*default=*/"64",
           "block C-tile rows BM (numWarps = BM/16 warps tile the M axis).">,
    Option<"blockN", "block-n", "unsigned", /*default=*/"64",
           "block C-tile cols BN (each warp owns BN/8 n8 mma column tiles).">,
    Option<"tf32", "tf32", "bool", /*default=*/"false",
           "precise input mode: keep f32 operands, round to TF32 and contract "
           "on the m16n8k8 nvgpu.mma.sync (tf32Enabled) tensor-core shape "
           "(rel ~1e-3 vs fp32 torch, cuDNN-matching). Default false = the fast "
           "fp16 m16n8k16 path (rel ~1e-2).">,
  ];
}

def ConvertIntentCompactToGPU
    : Pass<"convert-intent-compact-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single dynamic-shape compaction op (nonzero / "
                "masked_select / masked_scatter) to one block-cooperative "
                "prefix-sum gpu.launch.";
  let description = [{
    The C8 dynamic-shape-output pass — the ONE genuinely-new mechanism vs every
    prior cluster: a data-dependent output EXTENT. Recognizes a `func.func` whose
    body is exactly one `intent.nonzero` / `intent.masked_select` /
    `intent.masked_scatter` and rewrites it in place to ONE block-cooperative
    `gpu.launch` (grid = (1,1,1), block = (T,1,1)).

    The compaction primitive is an EXCLUSIVE PREFIX-SUM of the flattened 0/1
    mask: the flattened `Ntotal = prod(shape)` elements are split into T
    contiguous SEGMENTS (Ntotal % T == 0, seg = Ntotal/T); thread `t` first counts
    the true elements in ALL prior segments `[0, t*seg)` — its exclusive-prefix
    base slot — then re-scans its own segment `[t*seg, (t+1)*seg)` ascending,
    emitting each true element at the running slot (`base`, `base+1`, ...). Because
    the segments are contiguous ranges of the row-major flat index and each is
    scanned ascending, the global write order is EXACT row-major (matches
    torch.nonzero / torch.masked_select). No shared memory / barriers (each thread
    is independent); the last thread writes the total `nnz` to a `count` buffer.

    * nonzero (`cond`): dest-passing `(cond, out[MAXNNZ,ndim] i64, count[1] i64)`.
      Each true flat index `i` is unraveled to its `ndim`-vector coord (a divmod
      chain) and written as `out[slot, 0..ndim)`; `ndim = cond.rank`.
    * masked_select (`data`,`mask`): dest-passing `(data, mask, sel[MAXNNZ],
      count[1] i64)`. `sel[slot] = data[coord]` at each true position.
    * masked_scatter (`data`,`mask`,`src`): dest-passing `(data, mask, src, out)`,
      out DATA-SHAPED (static). `out[coord] = mask[coord] ? src[slot] :
      data[coord]` — the prefix-sum indexes `src`; no dynamic extent, no count.

    The runtime extent (`nnz`) flows HOST-side: the caller sizes the `[nnz,*]`
    allocation from a reduce over the mask BEFORE the launch (two-step), OR passes
    a MAX-sized buffer + reads the kernel's `count` and slices `[:nnz]` on the host
    (single-launch). Either way the destination leading dim is concrete at launch —
    orchestrate never allocates a dynamically-sized device buffer. A truth test of
    `!= 0` accepts f32 / integer / i1 masks. Correctness-only (bit-faithful, exact
    row-major order). Static shapes, Ntotal % T == 0, T a power of two. Run BEFORE
    `convert-intent-to-linalg`; the memref+gpu.launch module goes straight to the
    device pipeline.
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
           "Threads per block over the flattened element axis; 0 = auto "
           "(largest power-of-two <= min(Ntotal,256) dividing Ntotal).">,
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

    The recognizer also consumes the canonical live-linear spelling in which
    A/W are explicitly widened f16->f32 before the contraction, a rank-1 f16
    bias is widened and added in f32, and two f16 results return the identical
    identity preactivation (`C` plus saved activation input). For this exact
    semantic pattern the known mma.sync lane layout enables a register epilogue:
    load/widen bias, add in f32, truncate once, and store the same value to both
    destinations. No f32 accumulator tile is spilled through shared memory.
    Other supported activation DAGs retain the generic shared-tile epilogue.
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
    Option<"blockK", "block-k", "unsigned", /*default=*/"32",
           "K staging step BK (must be a multiple of the m16n8k16 MMA_K=16; "
           "BK/16 mma sub-tiles per stage). Wider BK stages more K per cp.async "
           "wave; the smem budget caps nStages*(BM+BN)*BK*2 at ~100 KiB. "
           "Autotuner knob (was a hard constant 32).">,
    Option<"smemPadK", "smem-pad-k", "unsigned", /*default=*/"8",
           "Pad the shared A/W K-tile leading dim by this many f16 elements "
           "(swizzle-free bank-conflict avoidance for the ldmatrix reads). "
           "Must be a multiple of 8 (16-byte cp.async alignment) or 0 to "
           "disable. Autotuner knob.">,
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
    Option<"capturedFlatOnly", "captured-flat-only", "bool", /*default=*/"false",
           "Require the exact flat split-K graph recognizer and reject the "
           "historical normalized intent.loop form. This makes automatic paper "
           "selection incapable of claiming the split-K fast-path ID for a "
           "different fused-attention graph.">,
    Option<"blockM", "block-m", "unsigned", /*default=*/"64",
           "Query rows per CTA. Must be a positive multiple of the 16-row "
           "WMMA tile; the block uses block-m/16 consumer warps.">,
    Option<"blockN", "block-n", "unsigned", /*default=*/"64",
           "Key rows per staged tile. Must be a positive multiple of 16; a "
           "tail smaller than block-n is zero-filled and logit-masked.">,
    Option<"fastF32LSELog", "fast-f32-lse-log", "bool", /*default=*/"false",
           "For an f64 LSE destination whose online-softmax state is f32, "
           "evaluate log(l) with the native f32 fast path before extending to "
           "f64. The full f64 polynomial remains the default; this option is a "
           "paper fast path guarded by strict source-vs-generated P6 checks.">,
  ];
}

def ConvertIntentFlashMMAToGPU
    : Pass<"convert-intent-flash-mma-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a fused FlashAttention-forward loop or the live B=H=1 "
                "flat rank-4 graph to one tensor-core gpu.launch with the "
                "online-softmax state REGISTER-RESIDENT via nvgpu.mma.sync.";
  let description = [{
    The register-resident `nvgpu.mma.sync` sibling of
    `convert-intent-loop-contraction-to-gpu` (M8a, which round-trips the
    online-softmax state O/S/P/m/l through dynamic shared). It recognizes either
    one `intent.loop` scanning the key axis (carries `m[Sq,1]`, `l[Sq,1]`,
    `O[Sq,D]`; body = `S=Q@Kj^T` -> online softmax -> `O+=P@Vj`) or the exact
    loop-free B=H=1 rank-4 graph emitted by the live FlashAttention P4 path:
    `bmhd,bnhd->nbhm`, max/sum on axis 0, LSE/TMP, f32-to-f16 P,
    `nbhm,bnhd->dbhm`, transpose, and f16 Out. The flat matcher is structural,
    never symbol-name based, and rejects extra mask/bias/tail operations. The
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
    Sk%64==0. The legacy loop form remains rank-2 and supports the existing
    optional LSE/causal controls. The live flat form is non-causal B=H=1 and
    preserves all three destination outputs: Out f16, LSE f32, and TMP=1/l f32;
    its scale input is a rank-0 f32 memref loaded inside the launch. Run BEFORE
    `convert-intent-to-linalg`; the result goes to the device pipeline
    (`lower_memref_module_to_kernels` with `scf_to_cf=True`, the scf.for carries
    vector iter_args). M8a is left 100% intact (separate pass).
  }];
  let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"2",
           "cp.async software-pipeline depth for the K/V key-tile loop "
           "(N rotating dynamic-shared K/V buffers; clamped to the K-tile "
           "count). 1 = single-buffer (no multistage). N=2 hides the K/V "
           "global-load latency behind the mma.sync+softmax of the previous "
           "tile (~1.2x at >=512 keys); N>=3 plateaus and only wastes smem.">,
    Option<"emitLse", "emit-lse", "bool", /*default=*/"false",
           "Also emit the per-query log-sum-exp (LSE = m + log(l), the "
           "log-sum-exp of the SCALED scores) as an appended [Sq,1] f32 "
           "destination output. This is the statistic FlashAttention's backward "
           "(and flag_gems' flash_api) consume; it is free here because the "
           "online-softmax m (rowmax) and l (rowsum) are already register-"
           "resident per row. Written once per row (from the quad's tig==0 "
           "lane). R-FLASH-3 attention-coverage extension.">,
    Option<"causal", "causal", "bool", /*default=*/"false",
           "Apply causal (lower-triangular) masking: score S[q,k] is set to -inf "
           "when the GLOBAL key index exceeds the GLOBAL query index (assuming "
           "Sq==Sk diagonal alignment), so query q attends only to keys <= q -- "
           "the flag_gems _attn_fwd STAGE=3 diagonal mask. Applied to the scaled "
           "scores in the register-resident online softmax (each accumulator "
           "lane knows its (query-row, key-col) from the m16n8 PTX layout). This "
           "is the CORRECTNESS extension (every key tile is still visited + "
           "masked; the STAGE=1 fully-masked-tile skip is not applied). "
           "R-FLASH-3 attention-coverage extension.">,
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
                "the query axis or the live flat rank-4 five-contraction DAG) "
                "to one tensor-core gpu.launch (serial-DQ).";
  let description = [{
    Recognizes either (1) a `func.func` whose body is FlashAttention backward as ONE
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

    or (2) the unchanged live B=H=1 flat rank-4 DAG with the equivalent
    `bmhd,bnhd->nbhm` / `nbhm,bmhd->bnhd` / `nbhm,bnhd->bmhd`
    contractions, explicit f16 round trips for P and dS, and
    `DQ = DQ_initial + dQ_update`. The flat matcher starts from the three return
    values and verifies every operation/type; it does not consult a function name
    or source symbol. Its distinct DQ destination is initialized from the nonzero
    DQ_initial input inside the same launch before serial accumulation.

    The match is STRUCTURAL (the 5 contraction subscripts + the recompute-P
    subtree + the carry shapes), NOT a kernel-name match -- it just happens to be
    attention-backward-shaped (hence the pass name).

    It rewrites the func in place to ONE tiled tensor-core `gpu.launch`. N4: the 5
    matmuls run on REGISTER-RESIDENT `nvgpu.mma.sync` m16n8k16 (the forward pass's
    proven recipe), NOT WMMA. The dV/dK accumulators are carried in registers
    (`scf.for` iter_args across the inner query loop); S/dP/dS and the online-
    softmax recompute run in registers; only P (then dS) is published to a single
    `[BM,BN]` f16 shared scratch, REQUIRED there for the dV/dK operand transpose.
    Both `dV=P^T@dO` and `dK=dS^T@Q` (and `dQ=dS@K`'s `K^T`) load their transposed
    operands DIRECTLY from the untransposed shared tiles via `nvgpu.ldmatrix
    transpose=true` (the `ldmatrix.trans` PTX) -- no separate transpose
    materialization, so the smem budget (~65KB) stays within the 99KB sm_120a cap.
    K/V/Q/dO stage via cp.async. 8 warps: the q-outputs (S/dP/dQ, M=BM=128) use 8
    row-strips; the k-outputs (dV/dK, M=BN=64) use 4 row-strips x 2 column-groups.
    The schedule is ONE block per (b,h): an OUTER `scf.for` over key tiles wrapping
    an INNER `scf.for` over query tiles. dQ uses the SERIAL accumulation mode
    (SEQUENCE_PARALLEL=false): one block owns the whole head and read-modify-
    writes the pre-zeroed global `DQ` deterministically across the outer key loop
    (the cross-program atomic_add mode + its serial==atomic gate are M9b).

    Q/K/V/dO are f16, LSE/Delta/scale and all accumulators are f32. Static shapes,
    BM=128, BN=64, D%16==0, Sq%128==0, Sk%64==0 (non-causal, no bias; tail-mask /
    causal / split-K are documented follow-ups). The pass appends three
    destination-passing outputs `dQ`, `dK`, `dV`; the legacy loop form expects
    host-preinitialized DQ, while the flat form copies its explicit DQ_initial
    input itself. Run BEFORE
    `convert-intent-to-linalg`; the result goes to the device pipeline
    (`lower_memref_module_to_kernels` with `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::vector::VectorDialect",
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
    Option<"flatSharedDPad", "flat-shared-d-pad", "unsigned", /*default=*/"8",
           "Extra f16 elements in each physical Q/K/V/dO shared-memory row for "
           "the exact flat-rank4 path. Must be a multiple of eight so every "
           "logical vector access remains 16-byte aligned; eight matches the "
           "captured Triton D-tile row pitch.">,
    Option<"flatSharedPdsPad", "flat-shared-pds-pad", "unsigned", /*default=*/"8",
           "Extra f16 elements in each physical P/dS scratch row for the exact "
           "flat-rank4 path. Must be a multiple of eight; the logical [BM,BN] "
           "tensor and all A/B semantics are unchanged. Eight is the measured "
           "bank-conflict minimum for the current standard-NVGPU schedule.">,
    Option<"flatDQInit", "flat-dq-init", "std::string",
           /*default=*/"\"in-kernel\"",
           "DQ initialization schedule for the exact flat-rank4 path. "
           "'in-kernel' copies DQ_initial inside the device launch; 'prelaunch' "
           "emits an explicit host memref.copy before the launch so kernel-only "
           "timing matches a baseline whose DQ is initialized before launch.">,
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

def ConvertIntentSplitKReduceVarargsToGPU
    : Pass<"convert-intent-splitk-reduce-varargs-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower the exact xFormers four-way split-K varargs merge to a "
                "stride-aware warp-per-row gpu.launch.";
  let description = [{
    Structurally recognizes the live flat Intent graph for xFormers' four-way
    split-K merge: four independent partial-output tensors and four independent
    partial-LSE tensors are concatenated on the singleton G axis to form the
    canonical packed-B contract `[B,4,H,M,D]` / `[B,4,H,M]`; a stable
    max/exp-weighted merge produces both the transposed destination Out and LSE.
    Matching is based on operand provenance, concat/reduce axes, result shapes
    and output structure, never on the function symbol.

    The rewrite preserves the real unrolled runtime ABI rather than replacing it
    with the legacy H=G=1 packed surrogate: 10 independent pointers followed by
    all 45 i32 stride arguments. H=4, G=1, D=64 and WRITE_LSE=true are the
    captured constexpr specialization. Addressing consumes the runtime strides,
    so non-contiguous views remain semantic inputs. Four warps cooperate per
    block, one output row per warp, and every lane owns two D elements. The
    resulting single launch is compiled through the standard GPU/NVVM/PTX/cubin
    backend. If this structural contract does not match, the pass leaves the
    function unchanged for the generic `convert-intent-to-linalg` fallback.
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
    Option<"rowsPerBlock", "rows-per-block", "int64_t", /*default=*/"4",
           "Number of independent output rows assigned to one block. Must be "
           "one of 1, 2, 4 or 8; one warp owns each row.">,
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
