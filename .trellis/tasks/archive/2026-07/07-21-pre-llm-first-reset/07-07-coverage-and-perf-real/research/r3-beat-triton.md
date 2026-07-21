# R3 — The concrete BEAT-TRITON technique per losing / naive kernel

Target HW: **RTX 5090D, GB202, sm_120a** (consumer Blackwell). ~170 SMs, 512-bit
GDDR7 ≈ **1.79 TB/s** HBM-equivalent, 96 MB-class large L2, fp32 CUDA cores + 5th-gen
tensor cores. f32 both sides in the P1 study (`python/tests/perf/REPORT.md`,
`bench_clusters_results.json`).

**sm_120a tensor-core reality (the hard constraint that shapes every answer below).**
Consumer Blackwell is an **sm_80-class MMA machine, not Hopper/DC-Blackwell**:
- YES: warp-synchronous `mma.sync` (m16n8k{8,16}), `ldmatrix`, `cp.async`
  (`cp.async.cg.shared.global`, 16-byte), FP8/FP4 tensor ops, large L2.
- NO: `wgmma` (sm_90 warpgroup-async), `tcgen05` / tensor-memory (sm_100 DC-Blackwell),
  TMA bulk-tensor copy. So the fastest realizable tensor-core loop on this part is
  **cp.async multistage → ldmatrix → warp mma.sync**, exactly the sm_80 Ampere recipe.
This is *already built and validated* in-tree — see below — which is the key lever.

**What we already have (do not rebuild).** Two working, cp.async-multistaged,
tensor-core matmul lowerings, both **fp16→f32**:
- `ConvertIntentContractionToGPU.cpp` — WMMA `gpu.subgroup_mma_*` →
  `wmma...m16n16k16.f32.f32`, BM=BN=64, BK=32, 4 warps, cp.async prologue/steady/epilogue.
- `ConvertIntentContractionToGPUMMA.cpp` — `nvgpu.ldmatrix` + `nvgpu.mma.sync`
  m16n8k16, register-resident vector<2x2xf32> accumulators carried as scf.for iter_args.
Both are validated to sm_120a cubin. **Neither has a tf32 (f32-input) variant yet** —
both take f16 A/B. That matters for conv (see §1).

---

## 1. conv2d — 10.2× loss (6.3% of f32 peak) → the realistic beat-cuDNN path

**Current (`ConvertIntentConvToGPU.cpp`).** Direct windowed-weighted accumulation, NO
tensor cores. grid = (N·Cout, prod(outer out-spatial), 1) = **one block per output row**;
inner out-spatial axis split over T threads, K=O_last/T cols each. Per output cell it
loops `ci_local ∈ [0,Cin/g)` × every kernel tap, one in-bounds `scf.if` per (ci,tap),
`acc += in·w`. For the benched `[32,64,56,56] k3 s1 p1 → 64` that is
32·64·56·56·(64·9) ≈ **3.7 GFLOP of scalar FMA on the CUDA cores at 6.6 TF/s ≈ 6% of
f32 peak**, versus cuDNN running the same math on **TF32 tensor cores**. This is a pure
compute-bound gap; no amount of load tuning closes 10×.

**What cuDNN/Triton actually do.** cuDNN's default winning kernel for this regime is
**implicit precomputed GEMM (`IMPLICIT_PRECOMP_GEMM`) on tensor cores**. Conv is
GEMM `Y[N·OH·OW, Cout] = im2col(X)[N·OH·OW, Cin·kH·kW] · W[Cout, Cin·kH·kW]ᵀ`. "Implicit"
= the im2col matrix is **never materialized**; the GEMM's shared-memory tile loader
computes each `(row=n·oh·ow, k=ci·kh·kw)` gather address on the fly (unravel k→(ci,kh,kw),
row→(n,oh,ow), src=(oh·s−p+kh·d, …), zero if OOB) and streams it via cp.async into the
same staged smem the matmul consumes. With allow_tf32 (torch default), inputs are
**TF32-rounded** and run on tensor cores → ~8× the fp32-core throughput. Triton's conv
tutorial is the same idea: a `tl.dot` GEMM whose LHS pointers are the implicit-im2col
gather. On this specific shape cuDNN may also pick Winograd F(2×2,3×3) (k3 s1) for a
further ~2.25× FLOP reduction, but implicit-GEMM alone already explains the 10×.

**Concrete beat path for us (reuse, don't reinvent).**
1. **Add a TF32 input mode to the existing MMA contraction pass.** The whole staged
   pipeline (cp.async → ldmatrix → mma.sync → register accumulators → epilogue) is
   done; the only new pieces are: accept f32 A/B, round to tf32 (`cvt.rna.tf32.f32` /
   `nvgpu.mma.sync` with tf32 operand type), and use the **m16n8k8** tf32 tensor-core
   shape instead of m16n8k16. This matches cuDNN's default numerics (both ~1e-3, and the
   bench already tolerates rel 2.9e-4). Cheaper alternative that reuses the pass verbatim:
   cast X,W to **fp16** and run the existing m16n8k16 path — legitimate here because the
   cuDNN reference is itself reduced-precision (TF32).
2. **Lower conv2d as implicit GEMM onto that contraction.** New pass (or extend
   `ConvertIntentConvToGPU`): keep the exact same OOB-guarded address math it already
   computes, but instead of scalar `acc+=`, emit it as the **smem tile-fill of a
   tiled-K GEMM** with M = N·OH·OW, N = Cout, K = Cin·kH·kW, BK stepping over the
   (ci,kh,kw) contraction. The in-bounds `scf.if` becomes the load predicate (OOB → 0
   into smem), then `mma.sync` over the tile. This is the single highest-value perf item:
   it moves conv from 6% to tensor-core throughput and should land within ~2× of cuDNN
   (cuDNN's remaining edge = Winograd + hand-tuned tiling), i.e. **10× → ~1.5–2×**.
3. **Winograd is a later, optional refinement** (k3 s1 only): transform 4×4 input tiles
   and 3×3 filters, do 16 element-wise (batched-GEMM) products, inverse-transform. Big
   code surface for ~2.25× on top; defer until implicit-GEMM lands.

Reality check on the ceiling: with only sm_80-class mma.sync (no wgmma/TMA/Winograd),
matching cuDNN *exactly* is unlikely, but **implicit-GEMM TF32 turns a 10× loss into
near-parity** and is the honest, hardware-grounded target.

---

## 2. sort / topk — 1.3–1.36× loss, and only **13–15% HBM** achieved

**Current (`ConvertIntentSortToGPU.cpp`).** **Single block per row**, block bitonic sort
in dynamic smem: pad N→P=next-pow2 with ±inf sentinels, run the
`log₂P·(log₂P+1)/2`-stage compare-exchange network, each stage a `gpu.barrier` + a
grid-stride sweep. That is **O(N log²N)** compares and, critically, **one block per row
leaves most SMs idle and every stage is barrier-separated** → the 13–15% HBM number is
the tell: it is latency/synchronization-bound, not bandwidth-bound.

**What CUB/Triton do.** CUB `DeviceSegmentedRadixSort` (what torch.sort dispatches to for
this row-segmented shape) is **O(N) LSD radix**: per-digit (4–8 bit) count → global
prefix-sum → scatter, a handful of full-bandwidth passes over the data. It is work-linear
and bandwidth-bound, hence the ~230/518 µs torch numbers. Triton's approach for
in-register top-k is a **bitonic sort but only over the small K that fits in registers/one
tile**, not a full-N network.

**Beat/match path.**
- If the workload is genuinely **full-row sort of large N**: the algorithmic fix is a
  **segmented radix sort** — per-row histogram of a 4-bit digit (16 counters in smem),
  block-exclusive-scan, scatter to a ping-pong buffer, repeat 8× for f32 (with the
  standard sign/exponent flip so IEEE f32 sorts as unsigned). This is a large new pass
  but is the only way to erase the log²N factor and reach CUB parity. Multi-block per row
  (each block owns a digit-partition) if a row exceeds one block.
- **Cheaper, high-leverage partial win:** most real callers want **top-k, not full sort**.
  If W≪N, replace the full bitonic network with a **grid-stride block-wide top-K
  reduction** (per-thread local top-k in registers → warp/block merge), O(N) and
  bandwidth-bound. The pass already special-cases W<N in the store; making the *algorithm*
  W-aware is the pragmatic beat. Also: **at minimum spread rows across more blocks** so
  the SMs aren't idle — even the existing bitonic at higher occupancy recovers a chunk of
  the 1.3×.

---

## 3. pool / argmax / triu + the memory-bound movers — bandwidth on the table?

**Are we leaving bandwidth on the table?** The movers we WIN/parity are already at the
GDDR7 ceiling — gather 0.78×, pad 0.83×, dropout 0.88×, tile/flip/concat/embed ≈1.0×, all
reading/writing at ~1.5–1.8 TB/s. The elementwise/mover emitter
(`ConvertIntentElementwiseToGPU.cpp`) uses the right access pattern: T threads,
K=N/T cols, `col = tx + k·T` → **consecutive threads touch consecutive addresses
(coalesced)**. So on the big square movers we are NOT leaving bandwidth on the table.

The **losers here are under-tiled or scalar-issue, not un-coalesced:**

- **pool / argmax / triu (~2×).** grid = **one block per output row** (pool:
  `(N·C, OH)` with T threads over only **OW=56**; argmax: one block/row, T over N). On the
  pool shape each block does a trivially small amount of work → **thousands of tiny blocks,
  poor per-SM utilization**, plus per-cell in-bounds `scf.if` overhead. torch/cuDNN pooling
  uses a **flat grid-stride over ALL output elements** with many warps per SM. Fixes, in
  leverage order: (a) **flatten the grid** to total-output-elements with a grid-stride
  loop sized to `~2–4 × numSM × maxThreads` so every SM stays busy; (b) **vectorize the
  loads** — see below; (c) **tail-mask** the ragged remainder instead of a per-element
  branch. triu specifically hits 1.77 TB/s (vanity-overcounted) yet loses 2× because it
  writes the full N² including the zero triangle from one-block-per-row; a flat
  grid-stride writer + skipping fully-zero blocks closes it.

- **We do NOT vectorize (no ld.128/st.128).** Every load/store the emitters produce is a
  **scalar 32-bit `memref.load/store`** (grep confirms no `vector<4xf32>`/128-bit anywhere
  in the mover passes). Coalesced 32-bit loads saturate bandwidth on the *big* movers
  because there are enough threads to hide latency, but for **short-row / small-tile ops
  (pool OW=56, argmax, the compaction chunk read) the instruction/issue count is 4× what
  it needs to be**. Triton emits `tl.load` over a BLOCK that the backend widens to
  **128-bit vector (ld.global.v4.f32)**, cutting memory instructions 4× and improving MLP.
  Emitting `vector<4xf32>` loads (K a multiple of 4, aligned) is a cheap, broadly
  applicable win for every latency-sensitive op.

- **masked_select / nonzero (2.5–3.1×, only 6–12% HBM) — the biggest remaining mover
  gap.** `ConvertIntentCompactToGPU.cpp` is now 3-phase multi-block (count → scan →
  scatter) which fixed the ~1400× catastrophe, BUT **each block is `block=1 thread`**
  (header + code confirm). So phase-1 count and phase-3 scatter read their contiguous
  chunk **one element at a time on a single lane per block → fully un-coalesced, ~1 lane
  of memory-level parallelism per block**. That is exactly why it sits at 6–12% HBM. The
  fix is a **block-cooperative, coalesced compaction**: T threads/block, each block loads
  its chunk with **coalesced (ideally vectorized) reads**, does a **block-level
  Hillis-Steele/warp-scan** of the mask (we already have the warp-shuffle scan primitive
  from the sparsemax cumsum work), and scatters cooperatively. CUB `DeviceSelect::Flagged`
  is precisely this — a single-pass, warp-aggregated, coalesced select at ~full
  bandwidth. This alone should take masked_select from 3× to near parity.

**Bottom line:** on the square movers we are at the ~1.8 TB/s wall (no headroom). The 2–3×
losers are lost to **(i) one-block-per-row / one-thread-per-block under-occupancy and
(ii) scalar (non-128-bit) issue** — both are codegen tiling problems, not physics.

---

## 4. The general backend gap — no autotuning of block/tile/vectorize

**Confirmed:** there is **no autotuner** anywhere (`grep autotune|tune|schedule-select`
over `python/intentir/backend` and `lib/Intent/Conversion` → nothing). Block size is
either a fixed constructor arg (`blockThreads`) or a one-line `autoThreads(N)` heuristic;
BM/BN/BK in the contraction passes are **hard-pinned constants** (BM=BN=64, BK=32); no
vectorization width is ever chosen. Triton's entire performance story is
`@triton.autotune` sweeping `{BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages}` and
caching the best per shape; CUB/cuDNN ship offline-tuned heuristic tables. We currently
ship one fixed schedule per op → we win only where the fixed schedule happens to be right
(bandwidth-bound squares) and lose on every shape it isn't (pool OW=56, sort, small argmax).

**What a minimal autotuner/schedule-selection would need:**
1. **Expose the knobs as pass options** (most already are, just not swept): per-pass
   `{blockThreads (T), K/tile, vectorWidth ∈ {1,4}, gridStyle ∈ {one-block-per-row,
   flat-grid-stride}, and for contraction BM/BN/BK/numStages}`. This is the real work —
   several passes hard-code these.
2. **A candidate generator** = small discrete grid of legal configs per op class
   (respecting the divisibility constraints the passes already assert: N%T==0, K%4==0 for
   vec4, P a pow2 for bitonic, etc.).
3. **A measured selection loop.** The harness already exists — `bench_clusters.py` times a
   lowered cubin under `do_bench_cudagraph` and asserts correctness. Wrap it: for each op
   at each target shape, lower every candidate config, bench, keep the min, **persist a
   `{op, shape-bucket} → best-config` cache** (JSON, like Triton's autotune cache). This is
   a build-time / one-shot cost, not per-run.
4. **A default heuristic** for un-tuned shapes (occupancy-driven: pick T and gridStyle so
   `blocks ≈ 2–4 × numSM`, vectorWidth=4 when aligned) so cold shapes are reasonable.

Highest-leverage first cut: a **3-axis sweep (T, gridStyle, vectorWidth)** applied to the
pool/argmax/triu/compact movers using the existing bench harness — that is enough to turn
the four "tunable" 2× losers toward parity **without any new algorithm**, and it is the
prerequisite that makes the pool/compact fixes in §3 land at their best config rather than
another guessed constant.

---

## Priority ranking (impact × grounding)

1. **conv2d: implicit-GEMM onto the existing cp.async mma.sync pass + TF32/fp16 input mode**
   — the only 10× loser; reuses built, validated tensor-core infra. 10× → ~1.5–2×.
2. **masked_select/nonzero: block-cooperative coalesced compaction (CUB DeviceSelect
   pattern)** — 6–12% HBM today, one-thread-per-block is the cause; reuses the warp-scan
   primitive. 3× → near parity.
3. **pool/argmax/triu: flat grid-stride + vector<4> loads + tail-mask** — codegen tiling,
   no new algorithm. 2× → parity.
4. **Minimal measured autotuner (T, gridStyle, vectorWidth) over the existing bench
   harness** — unlocks #2/#3 at their real optimum and every future shape.
5. **sort: W-aware block top-k for the common top-k case; segmented-radix only if full
   large-N sort is truly required** — algorithmic, largest effort, narrowest payoff.
