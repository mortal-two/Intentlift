# R5 — Structural Critique of the Current IntentIR Backend

Self-critique the user demanded ("你后端那就是写的一塌糊涂"). Honest assessment of
`lib/Intent/Conversion/*.cpp` (~21 passes), `python/intentir/backend/{orchestrate,opgraph}.py`,
and `python/intentir/benchmarks/row_reduction.py`. Evidence is cited by file:line.

---

## TL;DR

The backend is a pile of **~21 hand-written, single-schedule op-recognizers**, each
re-deriving grid mapping / coalescing / block-reduction / smem-carving / epilogue by
hand, wired to a **runtime that regex-scrapes the fully-lowered LLVM text** to recover
grid/block/args. It is *correctness-solid* and the read-once register-tile idea is
genuinely good (it is why rms_norm/softmax hit HBM-peak parity). But it has **no schedule
abstraction, no autotuning, no cost model, no vectorized memory codegen, no occupancy
model, and no tail-masking** — every performance lever would have to be re-implemented
independently in each of the 21 passes. It cannot reach general high-perf by extension;
it needs a real schedule/codegen layer inserted between recognition and emission, plus
an autotuning driver. The front-end recognizers and the read-once emit primitives are
reusable; the forked single-schedule `rewrite()` bodies and the regex runtime are what
must be replaced.

---

## 1. The structural problems (concrete)

### 1a. "Everything through RowReductionKernel" — a misnamed universal launcher, not a codegen layer
`RowReductionKernel` (`benchmarks/row_reduction.py:54`) is not row-reduction specific: it
takes a `pass_name` and drives **any** pass. `opgraph.py` routes sort, gather, scatter,
rng, layout, elementwise, and compaction through it
(`opgraph.py:162,174,187,218,238,558,615,637,666,738`). It is a thin wrapper: compile →
`_parse_launches` → resolve grid/block/args → `cuLaunchKernel`. The grid/block it launches
are **whatever integer the C++ pass hard-coded into the IR**; the runtime never tunes
anything. So "the backend" is really: one fixed schedule baked per pass + a regex launcher.

### 1b. One-block-per-row, cols-over-T, hard `N % T == 0` (no tail mask)
- The row-reduction launch is literally `blocks (M,1,1) threads (T,1,1)`
  (`ConvertIntentRowReductionToGPU.cpp:1217`). **Exactly one block per row.**
- Geometry: `T = blockThreads ? blockThreads : autoThreads(N); if (T<=0 || N%T!=0) return failure();`
  (`:442-444`). **Hard divisibility requirement**; the comment admits "tail-mask
  (non-divisible) is a documented M1 follow-up" (`:444`). Any N not divisible by a
  pow2-≤256 thread count simply **fails to lower**.
- `autoThreads` caps at 256: `while (t*2 <= N && t*2 <= 256) t*=2` (`MapEmit.h:312`).
- Consequence: **under-occupancy on small-M shapes.** M=8 rows → 8 blocks on a ~170-SM
  Blackwell part; the GPU is 95% idle. There is no split-row (segmented) reduction, no
  grid-stride, no fusing multiple rows per block. The tiled/smem path (`rewriteTiled`,
  `:712`) caches a big row in shared memory but is still one-block-per-row.

### 1c. No vectorized load/store
Every access is a **scalar 32-bit** `memref::LoadOp` at `cols[k] = tx + k*T`
(`:1234-1244`). The interleaved `tx + k*T` layout is coalesced *across threads for a fixed
k* (good), but there is **no `ld.global.v4.f32` / 128-bit vectorization** — no `float4`,
no `vector<4xf32>`. The elementwise pass is identical (scalar SSA register tile,
`ConvertIntentElementwiseToGPU.cpp:19-27`). For the many kernels that are *not* already at
HBM peak (elementwise chains, sort, gather, layout), transaction efficiency is left on the
table. `grep` for `vector<`/`ld.global.v` across the passes: **none**.

### 1d. No autotuning, no cost model
- `block-threads` is a pass Option defaulting to 0 (`Passes.td:57` and 11 more). When 0 it
  falls to `autoThreads`; otherwise the **caller hand-passes the library kernel's number**
  — the docstring is explicit: "passes the library kernel's geometry (e.g. 128/256)"
  (`Passes.td:60`; `row_reduction.py:64`). `opgraph.py`'s `_div_threads`/`_threads_pow2`
  (`:116-132`) pick a divisor heuristically.
- The MMA contraction passes expose `num-stages`/`block-m`/`block-n`/`num-producer-warps`
  (`Passes.td:566-665`), but `MatmulKernel` requires the **caller to pass them by hand**
  (`row_reduction.py:302-328`); the defaults (BM=64,BN=128,stages=3) are fixed constants
  (`ConvertIntentContractionToGPUMMA.cpp:60,68`).
- There is **no search anywhere**: no `do_bench` sweep over configs, no cost model, no
  per-(op,shape,dtype) tuning cache. `grep` for autotune/best_config/sweep in the backend:
  none. **You can only ever tie the library's chosen geometry, never search past it.**

### 1e. No occupancy consideration
The register tile is **fully unrolled into SSA** — "K is static and small, so we fully
unroll over k and keep each Row input's K loaded values as SSA" (`:497-499`). K = N/T grows
with N; a large K means a large live register tile → register pressure → low occupancy, and
**nothing measures or bounds it**. `tileColsMax` splits K into tiles only when
`K > tileColsMax` (`:451-459`), a crude register cap, not an occupancy model. Grid is
literally M; register/smem/occupancy never feed back into grid or block choice.

### 1f. Correctness-only passes, single fixed schedule each
Each `match()` **rejects anything it cannot prove correct** and then emits exactly **one**
schedule: divisibility (`:443`), power-of-two T for the tiled/scan trees (`:478,489`),
single-reduce-only for the tiled path (`:465`), no-cross-row-fold for scan (`:487`). There
are no alternative schedules and no perf-driven selection between them. The design target,
stated in the comments, is to **byte-match a specific library kernel's geometry**
("byte-identical to liger", `:1085`), not to derive the best schedule for the shape.

### 1g. Per-op ad-hoc lowering instead of a principled schedule/codegen layer
- **~21 separate `ConvertIntentXToGPU.cpp` passes**, 250–1360 lines each, every one
  hand-writing gpu.launch emission with a bespoke `match()`+`rewrite()`.
- **RowReduction alone forks its schedule three ways**: `rewrite()` (`:1175`),
  `rewriteTiled()` (`:712`), `rewriteScan()` (`:942`) — each **re-implements** the launch
  scaffold, the shared-memory reduction tree, and the epilogue. The manual block-reduction
  tree is copy-pasted (`:844-859` vs `:1132-1147`).
- **Contraction is forked four ways**: WMMA, MMA, MMA-WS, LoopContraction (separate 547–718
  line files). Each re-derives tiling, K-loop, smem staging, epilogue by hand.
- Net: adding a new op = a new ~500-line pass; adding a new *schedule* to an existing op =
  forking its `rewrite()` (exactly what tiled/scan already did). There is **no shared tiling
  IR, no schedule object, no reusable coalescing/reduction/smem primitives.**

### 1h. The runtime is regex-scraping the lowered LLVM (a symptom, not a layer)
`orchestrate.py`/`row_reduction.py` recover grid, block, args, and intermediate buffers by
**regex over the fully-lowered LLVM-dialect text**: `_parse_launches` (`orchestrate.py:561`),
`_parse_constants` (`:373`), `_map_input_aligned_ptrs` assuming (alloc,align) pointer
pairing (`:350-370`), and `_parse_loose_buffers` that **string-walks add/ptrtoint/getelementptr
to recover a malloc's element count** (`:461-558`). This is brittle by construction: it
depends on `-canonicalize` folding grid extents to literal constants (`compile.py:51`), on
`insertvalue` field positions, and on source-order malloc↔inttoptr pairing. A real backend
would emit a structured launch plan; here the plan is reverse-engineered from text every run.

---

## 2. What a high-performance backend architecture needs (and what we lack)

Contrast with the refs OptGuide backend (`refs/TianchenForge/backends/cuda/`): it has
`lowering/lowering_contract.py`, per-arch `legality.py`, `realizer_selector.py`,
`target_support_contracts.py`, and ratio (cost) evidence — an explicit **admission →
selection → cost-attribution** structure. Our codegen side has none of that. A high-perf
backend needs:

1. **A schedule / tiling abstraction.** A first-class `Schedule` (grid mapping, block shape,
   per-thread tile, vector width, smem plan, pipeline depth, split-K/split-row factor)
   *separate from the op's math*. **One** parametric codegen consumes `(op-DAG, schedule) →
   gpu.launch`, replacing the 21 bespoke `rewrite()`s and the 3-way/4-way forks. Today math
   and schedule are welded together in each pass.

2. **Autotuning or a cost model.** Enumerate legal schedules, time them with `do_bench`
   (empirical, Triton-style) or rank by a bytes/occupancy cost model, and cache the winner
   per (op, shape, dtype). This is the only way to move from "match the library's number" to
   "beat it." Completely absent today.

3. **Vectorized coalesced memory codegen.** Emit `vector<4xf32>` / 128-bit `ld/st.global.v4`
   when the access is contiguous and aligned; fall back to scalar + tail mask otherwise. The
   single biggest bandwidth lever after read-once for the non-peak kernels.

4. **Occupancy-aware grid sizing.** Grid-stride loops so grid = f(SM count, achieved
   occupancy), not literally M; split large rows across blocks (segmented reduction with a
   second small combine kernel or atomics) when M underfills; bound the register tile K from
   an occupancy target instead of a raw `tileColsMax`.

5. **Tensor-core paths as schedule instances, not forked files.** WMMA/MMA/MMA-WS already
   exist and work, but as four hardcoded passes with fixed BM/BN/BK; they should be schedule
   points in the same codegen, with num-stages/block-m/-n **tuned**, not hand-set.

6. **A structured launch plan** emitted by the pass (grid/block/args/smem as an attribute or
   sidecar), replacing the regex scraping in `orchestrate.py`/`row_reduction.py`.

**Salvageable-by-extension, or a new layer?** Honest answer: **it needs a real
schedule/codegen layer** — it cannot reach general high-perf by extension. Every perf lever
(vectorization, tail-masking, occupancy, autotuning, split-row, TC tuning) currently requires
editing each of the ~21 passes independently, and the runtime is regex-scraping. BUT the
rewrite is not from zero: the **op-recognition front-ends** (each `match()`) and the
**read-once register-tile / smem-tree emit primitives** are good and reusable. The parts to
*replace* are the forked single-schedule `rewrite()` bodies (collapse into one parametric
codegen) and the regex runtime (replace with an emitted launch plan).

---

## 3. Most impactful restructuring, in priority order

**P1 — Tail-masking + split-row in the shared reduce/elementwise emitter.**
Removes the `N % T == 0` straitjacket (`:443`) so arbitrary N lowers, and adds a
second-level combine (or atomic) so small-M shapes fill the GPU. Biggest combined
correctness + coverage + occupancy win, and the precondition for any real autotuning
(you cannot sweep block sizes if only divisors are legal).

**P2 — Vectorized coalesced ld/st (float4 / `vector<4xf32>`).**
Biggest single bandwidth lever for the many kernels not already at HBM peak (elementwise
chains, gather, layout, sort staging). Read-once already maxed *traffic*; this maxes
*transaction efficiency*.

**P3 — A `Schedule` abstraction + one parametric codegen replacing the forked
`rewrite()`s** (rewrite/rewriteTiled/rewriteScan; WMMA/MMA/MMA-WS/LoopContraction).
Structural payoff: P1 and P2 get written **once**, not 21×; new schedules become data, not
new ~500-line files.

**P4 — An autotuning driver.**
`do_bench` sweep over {block-threads, tile-cols, vector-width, num-stages, split factor},
cache best per (op, shape, dtype). Turns "tie the library" into "search past it." Depends on
P1 (legal config space) and P3 (schedules as parameters).

**P5 — Replace the regex runtime with an emitted structured launch plan.**
Have each pass attach grid/block/args/smem as an attribute (or a small JSON sidecar) so
`orchestrate.py`/`row_reduction.py` stop string-walking lowered LLVM
(`orchestrate.py:461-589`). Removes brittleness and is the clean substrate P3/P4 read from.

---

## Bottom line

The user is substantially right. The backend *works* and is honestly measured (the
row_reduction docstring's do_bench discipline is good), but it is architecturally a
collection of one-off, single-schedule, scalar-memory, non-tuned recognizers glued to a
regex launcher. There is no schedule layer, no autotuner, no cost model, no vectorization,
no occupancy model, and no tail-masking. High performance in general requires inserting a
real schedule/codegen layer (P3) with tail-mask+split-row (P1), vectorized memory (P2), and
autotuning (P4) on top of the (reusable) recognizers and read-once primitives.
