# PRD — Real Coverage + High Performance (beat Triton / flag_gems)

Task: `07-07-coverage-and-perf-real` · Base: `master` · Priority: P2
Target HW: **RTX 5090D, GB202, sm_120a** (consumer Blackwell), triton 3.6.0, CUDA 13.0.

---

## 0. Why this PRD exists (what was rejected)

The prior coverage push was **correctness-only**: it lowered 162/231 flag_gems/triton
modules, verified them against a **torch reference** at `rel<1e-3`, and then used
*PRD-out-of-scope* labels to excuse (a) not benchmarking flag_gems, (b) not doing flash,
(c) shipping a **naive backend** (one-block-per-row, scalar 32-bit loads, no tensor cores
for conv, no autotuning). The measured result was honest but weak: on 20 benched ops we
win/parity on 11 memory-bound movers and **lose 1.3×–10.2×** on everything the codegen is
naive about (`python/tests/perf/REPORT.md`).

This PRD makes **performance the gate, not an afterthought**, while *keeping* the 162/231
coverage. "Beat Triton" here means the paper's honest bar — **~1.3× geomean, not orders of
magnitude** — and we are explicit about the one place where sm_120a hardware genuinely
caps us (~1.2–1.3× vs cuBLAS-class GEMM, no Hopper wgmma/TMA). Everywhere we cannot beat,
we target **parity with a named mechanism**, and everywhere we declare a wall we cite
evidence, not a PRD label.

---

## 1. Goal

Turn the correctness-complete IntentIR GPU backend into a **measured high-performance**
backend on sm_120a that:

1. **Benches fairly 3-way** — ours vs **torch (ATen/cuDNN/CUB)** vs **flag_gems (real
   Triton)** — all in-process on the same `do_bench_cudagraph` steady-state path.
2. **Beats or matches Triton/flag_gems** on the ops where the paper's mechanism applies
   (matmul, conv, attention/flash, norm, softmax): **≥1.0× (parity) required, ~1.2–1.3×
   geomean targeted** where we hold the lever.
3. **Erases the naive-codegen losses** on the movers: conv2d 10.2×→≤2×, masked_select/
   nonzero ~3×→≤1.3×, pool/argmax/triu ~2×→≤1.2×.
4. **Does flash honestly** — fwd to parity at all shapes, bwd to near-parity — via ordinary
   changes (NOT warp-spec, which is correctly abandoned).
5. **Restructures the backend** toward high-perf codegen: vectorized/coalesced memory,
   tail-masking + grid-stride occupancy, tensor-core reuse, and a **measured autotuner**
   with a per-(op,shape) cache — the only way to move from "tie the library" to "beat it."
6. **Keeps 162/231 verified-e2e coverage** with no correctness regression.

---

## 2. What we know (grounded baseline)

### 2.1 The paper's real mechanism (R1) — how IntentIR beats Triton
- GPU margin is **modest and real**: **1.45× H100 / 1.19× 5090D** on the 8-op suite,
  **1.34× / 1.35×** on 28 complex ops (`refs/intentir/paper/manuscript/main.tex §E5`).
- The GPU win comes from **(a)** hand-written specialized kernel templates (tensor-core
  WMMA matmul w/ `cp.async` pipeline; warp-shuffle + `float4` + CUB softmax/layernorm/
  reduce), **(b)** a **Layer-B contract-gated fast-path/fallback split** ("two versions per
  kernel") enabling **bounds-check removal + float4/cp.async vectorization**, **(c)**
  **host-side dispatch** among a *small* set of pre-tuned tile variants. **There is NO GPU
  autotuner in the paper** — `_schedule()` is a fixed per-op-family table
  (`matmul_conv→{64,128,32}`, else `{1,256,1}`, env-overridable).
- The only real *search* is **RVV-only**: an analytical GEMM **roofline cost model**
  (`backends/spmd_rvv/analysis/`). That is a CPU/RVV story; our GPU analog is the measured
  autotuner in §5 M6.

### 2.2 Assets we already have (do NOT rebuild — R1, R3, R4)
- Tensor-core matmul, both fp16→f32, cp.async-multistaged, validated to sm_120a cubin:
  `ConvertIntentContractionToGPU.cpp` (WMMA `m16n16k16`) and
  `ConvertIntentContractionToGPUMMA.cpp` (`nvgpu.ldmatrix`+`nvgpu.mma.sync` `m16n8k16`).
  **Neither has a TF32 (f32-input) variant yet.**
- Flash forward `ConvertIntentFlashMMAToGPU.cpp` (641 L, register-resident online-softmax
  mma.sync + cp.async KV staging) and flash backward
  `ConvertIntentAttentionBackwardToGPU.cpp` (617 L, 5-contraction, serial+atomic DQ modes,
  **on WMMA — a known gap**). Plus `ConvertIntentSplitKReduceToGPU.cpp`,
  `...SplitKReduceBackwardToGPU.cpp`, `ConvertIntentLoopContractionToGPU.cpp`.
- 20 `ConvertIntent*ToGPU.cpp` passes total; correctness-solid; the read-once register-tile
  / smem-reduction-tree emit primitives are good and reusable.

### 2.3 The naive-backend deficits (R3, R5) — measured, file:line-grounded
- **No vectorization**: every emitter uses scalar 32-bit `memref.load/store`; no
  `vector<4xf32>` / `ld.global.v4` anywhere (grep-confirmed).
- **One-block-per-row / hard `N%T==0`**: `ConvertIntentRowReductionToGPU.cpp:1217,442-444`;
  tail-mask is a "documented M1 follow-up." Under-occupancy on small-M and awkward shapes.
- **One-thread-per-block compaction**: `ConvertIntentCompactToGPU.cpp` is 3-phase multi-block
  but each block is `block=1 thread` → un-coalesced chunk read → 6–12% HBM.
- **No autotuner / cost model / tuning cache** anywhere (`grep autotune|tune|schedule-select`
  → nothing). Block/tile/vector-width are fixed constants or a one-line `autoThreads`.
- **Regex-scraped runtime**: `orchestrate.py`/`row_reduction.py` recover grid/block/args by
  string-walking lowered LLVM — brittle, and no structured launch plan.
- **Forked schedules**: RowReduction forks `rewrite`/`rewriteTiled`/`rewriteScan`;
  contraction forks WMMA/MMA/MMA-WS/LoopContraction — every perf lever must be re-added 4–21×.

### 2.4 Current measured perf (R3, `bench_clusters_results.json`) — ratio = ours/torch, <1 wins
| verdict | ops |
|---|---|
| WIN/parity (≈HBM-bound, keep) | gather 0.76, pad 0.85, dropout 0.88, embedding/scatter_add/concat/tile/flip/rand/upsample/polar ≈1.0 (11/20) |
| LOSE — tunable tiling | triu 1.98, argmax 2.25, avg_pool 1.96, max_pool 2.72 |
| LOSE — un-coalesced compaction | masked_select 2.6–3.1, nonzero 2.4–2.6 |
| LOSE — algorithmic | sort 1.29–1.36 (bitonic vs radix) |
| LOSE — no tensor core | **conv2d 10.2×** (6% of f32 peak vs cuDNN TF32) |

### 2.5 The flag_gems benchmarking bug (R2) — SOLVED, must be adopted
- Root cause: flag_gems `libentry.py` version gates (`:678,691,763`) cover triton 3.3–3.5
  and **exclude 3.6**; on 3.6 the cached fast-path drops the **constexpr placeholder slot**,
  so the launcher gets 31 args where triton-3.6 wants 32. (`launch_pdl` was a red herring.)
- Fix (validated 100/100 in-process, incl. `do_bench_cudagraph`): a **1-line monkeypatch**
  in *our* code — `flag_gems.utils.libentry.minor_version = 5` when `major==3 and minor>=6`,
  routing 3.6 to flag_gems' own constexpr-inclusive launch path. site-packages stay read-only.

### 2.6 The honest sm_120a wall (R3, R4) — declared, not chased
sm_120a is an **sm_80-class MMA machine**: it HAS `mma.sync` (m16n8k{8,16}), `ldmatrix`,
`cp.async`, FP8/FP4; it does **NOT** have `wgmma` (sm_90), `tcgen05`/tensor-memory (sm_100),
or TMA. Consequences we accept as hardware facts:
- Best `mma.sync` GEMM trails `tl.dot` by **~1.2–1.3× on large compute-bound shapes** —
  the irreducible tensor-core residual. We declare it; we do not chase it. Triton flash sits
  under the *same* wall, so flash-vs-Triton parity is unaffected.
- **Warp-specialized matmul is correctly abandoned**: `mbarrier` producer/consumer lowers &
  runs, but **deadlocks at occupancy ≥ 2** (no forward-progress guarantee; no Hopper
  warpgroup-MMA/TMA) and is *slower* single-wave. It is NOT flash's ceiling and NOT a path.
- **fp8_v attention** codegen (`tl.float8e4nv`) is unverified on sm_120a → gated OUT.

### 2.7 Coverage baseline to preserve
**162/231 verified-e2e** = 137 flag_gems (of 193) + 25 triton (of 38), per
`corpus/coverage.json`. This PRD must not regress it.

---

## 3. Requirements

### R-BENCH — Fair 3-way measurement (the gate's foundation)
- **R-BENCH-1** Add `python/intentir/backend/flaggems_shim.py::apply_flaggems_launch_shim()`
  (idempotent, guarded `major==3 and minor>=6`, sets `libentry.minor_version=5`). Call it
  immediately after every `import flag_gems`.
- **R-BENCH-2** `bench_clusters.py` must bench **ours, torch, AND flag_gems** in-process on
  the **same `do_bench_cudagraph`** path. Retire the `FG_BENCH` gate, the subprocess
  `fg_us()`/`fg_ref.py` escape hatch, and the stale "31 vs 32 = launch_pdl" comment.
- **R-BENCH-3** Report the **our/reference ratio** as load-bearing; keep the GB/s / %HBM
  columns only as clearly-labelled vanity (denominator/overcount noted). Correctness asserted
  at the benched shape for every contender (nothing a no-op).
- **R-BENCH-4** `nonzero` needs an `fg.nonzero` existence/signature check before use.

### R-PERF — Beat/parity gates (see §4 for numbers)
- **R-PERF-1** Every op in the **beat class** (matmul, conv, attention/flash, layernorm/
  rmsnorm, softmax) benches at **≤1.0× (parity) or better** vs the *faster* of torch/flag_gems,
  with a **~1.2–1.3× geomean target** where the lever exists. No op in this class regresses.
- **R-PERF-2** The **naive-loss** ops hit the closure targets in §4.2.
- **R-PERF-3** Perf claims are **mechanism-named**. Any "beat flag_gems" claim states the
  byte-saving/geometry lever; parity is the default honest bar for autotuned-Triton targets.

### R-CODEGEN — High-performance codegen capabilities (R3/R5)
- **R-CODEGEN-1 Vectorized coalesced memory**: emit `vector<4xf32>` / 128-bit ld/st when
  contiguous + 16B-aligned + `K%4==0`; scalar + **tail-mask** fallback otherwise. A shared
  emitter capability, not per-pass.
- **R-CODEGEN-2 Tail-masking + occupancy grid-stride**: remove the `N%T==0` straitjacket;
  add a **flat grid-stride** grid style sized `~2–4× numSM×maxThreads`, and split-row
  (segmented) reduction for small-M. Shared across reduce/elementwise/pool/arg/mover emitters.
- **R-CODEGEN-3 TF32 tensor-core input mode**: extend the existing MMA contraction pass to
  accept **f32 A/B**, round to tf32 (`cvt.rna.tf32.f32`, `m16n8k8`), matching cuDNN numerics.
  (Cheaper reuse path: cast to fp16 into the existing `m16n8k16` — legitimate because the
  cuDNN reference itself runs TF32.)
- **R-CODEGEN-4 Contract-gated two-version emission**: where Layer-B certifies in-bounds +
  alignment, emit a specialized fast path (bounds-check-removed, vectorized) with a
  conservative fallback — the paper's stated GPU win driver.
- **R-CODEGEN-5 Measured autotuner**: expose knobs as pass options
  `{blockThreads T, tile/K, vectorWidth∈{1,4}, gridStyle∈{one-block-per-row, flat-grid-stride},
  contraction BM/BN/BK/numStages}`; a legal-config candidate generator; a `do_bench` selection
  loop reusing `bench_clusters.py`; a persisted `{op, shape-bucket, dtype}→best-config` JSON
  cache; and an occupancy-driven default for cold shapes.

### R-FLASH — Attention/flash to the achievable bar (R4)
- **R-FLASH-1** Flash **fwd**: add a **cached launcher** (kill per-call Python arg rebuild) →
  **parity at all shapes** (paper 5090D = 0.984×). NOT a kernel change.
- **R-FLASH-2** Flash **bwd**: port matmuls **WMMA→mma.sync** (reuse the fwd recipe), raise
  **4→8 warps**, and **pipeline the 5-matmul barrier chain** → near-parity (paper ~1.03–1.06×).
- **R-FLASH-3** Attention coverage extensions (loop-shape only, doable): causal STAGE masking,
  additive `attn_mask` bias, batch/head dims, LSE output, xformers split-K fwd + splitK-reduce
  varargs. **fp8_v gated OUT.**

### R-COV — Keep coverage
- **R-COV-1** 162/231 verified-e2e preserved; correctness rel tol maintained (rel<1e-3
  general; flash rel<2e-2; conv/matmul TF32/fp16 legitimate vs TF32 cuDNN reference).
- **R-COV-2** sort/unique/nonzero/scatter/pooling remain **correctness+coverage** ops; where
  §4 sets a perf target it is a closure target, not a beat-Triton claim.

### R-RUNTIME — Structured launch plan (R5 P5)
- **R-RUNTIME-1** Each perf-critical pass attaches a **structured launch plan**
  (grid/block/args/smem as an attribute or sidecar); `orchestrate.py`/`row_reduction.py`
  consume it instead of regex-scraping lowered LLVM. (May land incrementally per §5.)

---

## 4. Acceptance criteria

### 4.1 Bench harness (gate 0 — blocks everything)
- flag_gems runs ≥100× in-process under `do_bench_cudagraph` with the shim; the 3-way table
  (ours/torch/flag_gems) is produced with correctness asserted per contender.
- `FG_BENCH` gate + subprocess path removed; results committed to `bench_clusters_results.json`
  and `python/tests/perf/REPORT.md` updated to the 3-way story.

### 4.2 Per-op closure targets (ratio = ours / faster-reference, <1 = we win)
| op | now | acceptance | mechanism |
|---|--:|--:|---|
| conv2d k3s1p1 | 10.2× | **≤2.0×** | implicit-GEMM onto mma.sync + TF32/fp16 (R-CODEGEN-3) |
| masked_select / nonzero | 2.4–3.1× | **≤1.3×** | block-cooperative coalesced compaction |
| avg_pool / max_pool+idx | 1.96–2.72× | **≤1.2×** | flat grid-stride + vector<4> + tail-mask |
| argmax | 2.25× | **≤1.2×** | flat grid-stride + tail-mask |
| triu | 1.98× | **≤1.2×** | flat grid-stride writer, skip zero blocks |
| sort ax1 | 1.29–1.36× | **improved, may stay >1.0** (scoped) | W-aware top-k; radix later |
| gather/pad/dropout/parity movers | 0.76–1.06× | **no regression** | keep |

### 4.3 Beat-class geomean (gate 1)
- Over {matmul, conv, layernorm, rmsnorm, softmax, flash-fwd, flash-bwd} the geomean vs the
  faster of torch/flag_gems is **≤1.0× (parity) required**, with **~1.2–1.3× targeted** on the
  subset holding the lever (norm/softmax/matmul-at-512³). Each op individually ≤1.3× unless
  covered by the declared cuBLAS wall (§2.6), which must be cited inline.

### 4.4 Flash (gate 2)
- flash **fwd**: `mine ≤ triton×1.1` at **all** benched shapes (128…1024), i.e. parity —
  the small-shape launch-overhead gap is closed by the cached launcher.
- flash **bwd** (atomic): `mine ≤ triton-best×1.2`, target ~1.06×; residual named if any.
- Attention extensions (causal/attn_mask/batch-head/LSE/splitK) PK-correct vs torch SDPA at
  rel<2e-2.

### 4.5 Autotuner (gate 3)
- For the tuned ops, the cached best-config beats the old fixed-schedule number on ≥1 real
  shape (demonstrates "search past the library," not just "tie it"); cache persists and is
  reused without re-searching.

### 4.6 Coverage (gate 4)
- `corpus/coverage.json` still shows 162 verified-e2e (137 fg + 25 triton); no PK regression.

---

## 5. Milestone plan (difficulty-ordered, dependency-aware)

Ordering is easiest-first / prerequisite-first, with the highest-value item (conv) pulled as
early as its dependencies allow.

**M0 — flag_gems shim + fair 3-way bench harness (R-BENCH).** *Easiest, unblocks the gate.*
The shim exists and is validated; wire it in, delete the subprocess/`FG_BENCH` workaround,
add flag_gems as a third in-process contender in `bench_clusters.py`. Without this there is no
Triton number to beat. **Exit: 4.1.**

**M1 — Vectorized coalesced memory + tail-mask + flat grid-stride (R-CODEGEN-1/2).** *Medium.*
Add `vector<4xf32>` ld/st (aligned, K%4==0) + tail-mask fallback + a flat grid-stride grid
style to the shared reduce/elementwise/pool/arg emitter primitives. This alone moves
pool/argmax/triu toward parity with **no new algorithm**. **Exit: pool/argmax/triu rows in 4.2.**

**M2 — Block-cooperative coalesced compaction (R-CODEGEN, R3 §2).** *Medium.*
Rewrite `ConvertIntentCompactToGPU.cpp` from `block=1 thread` to T-threads/block: coalesced
(ideally vector<4>) chunk loads, block-level warp-shuffle/Hillis-Steele scan (reuse the
sparsemax cumsum primitive), cooperative scatter — the CUB `DeviceSelect::Flagged` pattern,
preserving exact row-major order. **Exit: masked_select/nonzero row in 4.2.**

**M3 — conv2d as implicit GEMM + TF32/fp16 input mode (R-CODEGEN-3, R3 §1).** *Medium-hard;
highest value.* Add a TF32 (or fp16-reuse) input mode to the existing `...GPUMMA` contraction
pass, then re-lower conv2d as implicit GEMM (M=N·OH·OW, N=Cout, K=Cin·kH·kW) — keep the exact
OOB-guarded address math but emit it as the smem tile-fill (load predicate) of the tiled-K
GEMM. **Exit: conv2d ≤2.0× (4.2); the single 10× loser erased.**

**M4 — Flash to the achievable bar (R-FLASH).** *Medium-hard.* (a) cached launcher for
flash-fwd → parity all shapes; (b) flash-bwd WMMA→mma.sync + 8 warps + pipeline the 5-matmul
chain → near-parity; (c) attention coverage extensions (causal/attn_mask/batch-head/LSE/
splitK), fp8_v gated out. **Exit: gate 2 (4.4).**

**M5 — Measured autotuner + per-(op,shape) cache (R-CODEGEN-5, R5 P4).** *Hard.* Expose the
knobs already latent in the passes as swept options; discrete legal-config generator; wrap
`bench_clusters.py` as the lower→bench→min loop; persist a JSON cache; occupancy default for
cold shapes. First cut: 3-axis sweep (T, gridStyle, vectorWidth) over the M1/M2 movers so
they land at their *true* optimum, then extend to contraction BM/BN/BK/numStages. **Exit:
gate 3 (4.5); M1/M2/M3 re-tuned to best config.**

**M6 — Schedule abstraction + structured launch plan (R5 P3/P5).** *Hardest, structural.*
Introduce a first-class `Schedule` (grid mapping, block, per-thread tile, vector width, smem
plan, pipeline depth, split factor) separate from op math, consumed by one parametric codegen
that replaces the forked `rewrite/rewriteTiled/rewriteScan` and WMMA/MMA/MMA-WS/LoopContraction
bodies; and emit a structured launch plan replacing the regex runtime. *Scope note:* this is
the clean substrate M5 wants; if time-boxed, land the **launch-plan** half (R-RUNTIME-1) first
(it de-brittles M0–M5) and treat the full Schedule-unification refactor as a stretch that can
extend past this task without blocking the perf gates. **Exit: R-RUNTIME-1; forks collapsed
where landed.**

**M7 — sort W-aware top-k (R3 §5).** *Largest effort, narrowest payoff — last.* Implement a
block-wide O(N) top-k for the common W≪N case (per-thread register top-k → warp/block merge);
at minimum spread rows across more blocks so SMs aren't idle. Full large-N segmented radix
sort is a further, lower-priority effort. **Exit: sort improved (4.2, scoped).**

Dependencies: M0 → everything (no gate without it). M1 → M2 (shares vec/tail-mask), M5 (needs
legal config space). M3 reuses the M-none-new contraction infra (independent of M1/M2, can run
parallel). M5 depends on M1+M6-launch-plan. M6 underpins M5's cleanliness but the launch-plan
half is independently valuable.

---

## 6. Decision (ADR — resolved 2026-07-07)

**User chose MAXIMAL scope** ("推到 sm_120a 硬件天花板"): A (erase naive losses + match/beat
flag_gems) **+ B** (full backend restructure: measured autotuner M5 + Schedule-abstraction M6
with fork-collapse) **+ C** (push conv/flash to the absolute sm_120a ceiling, complete
segmented radix sort, re-attack the 14 irreducible coverage items, and justify every remaining
hardware wall with evidence not a label). Consequences:

1. **conv/matmul input mode: do BOTH.** fp16-reuse first to land conv fast, THEN TF32
   (`m16n8k8`, `cvt.rna.tf32.f32`) for cuDNN-exact numerics + the ceiling push.
2. **M6 Schedule unification: full fork-collapse in-task** (not just the launch-plan half) —
   the parametric-codegen refactor of the ~20 passes is in scope; it is the clean substrate M5
   needs and the "beat not tie" foundation.
3. **Beat-margin: parity-required, ~1.3× targeted where the lever exists, pushed to the
   hardware ceiling.** Every op that can hold a lever is driven to its sm_120a optimum; where
   physics caps us (cuBLAS-class GEMM residual, no wgmma/TMA) we state it with measured evidence.
4. **Pulled INTO scope** (was proposed out-of-scope): full large-N segmented radix sort (M7),
   Winograd conv as the ceiling push on top of implicit-GEMM, and re-attacking the 14 residual
   coverage modules (rwkv/DSA/complex-flash/attention family) to their real achievable bar.

---

## 7. Out of scope (with real justification, not excuses)

- **Chasing the ~1.2–1.3× cuBLAS-class GEMM residual.** Requires Hopper `wgmma`+TMA the
  sm_120a silicon lacks (R4). Declared as the hardware wall; Triton flash is under the same
  wall so flash-vs-Triton parity is unaffected. *Evidence, not a label.*
- **Warp-specialized matmul.** Lowers & runs but **deadlocks at occupancy ≥ 2** and is slower
  single-wave on sm_120a (no forward-progress guarantee). Correctly abandoned; not a path.
- **fp8_v attention.** `tl.float8e4nv` codegen unverified on consumer Blackwell — gated to
  fp16/bf16.
- **RVV/CPU roofline retune.** The paper's real search is RVV-only; this task is GPU-focused
  and the GPU analog is the measured autotuner (M5). RVV retune is a separate effort.

*(Pulled INTO scope by the maximal decision, no longer out: Winograd conv — the ceiling push on
top of implicit-GEMM; full large-N segmented radix sort — M7 now targets the complete radix, not
just W-aware top-k; the 14 residual coverage modules — re-attacked to their real achievable bar.)*
- **Auto-lifting flash via the LLM extraction pipeline.** Flash needs hand loop-authorship
  (the passes already exist); "flash = PRD-OOS" in `corpus/coverage.json` refers *only* to the
  auto-extraction harness, not to capability. We ship flash by hand-authored passes (M4).

---

## 8. Technical notes / pointers

- Shim: `flag_gems/utils/libentry.py:678,691,763,782-783` (buggy gate) →
  `:763-781` (correct constexpr-inclusive path we route to);
  `triton/backends/nvidia/driver.py:123` (`_BASE_ARGS_FORMAT` 13 fixed incl launch_pdl),
  `:186-207` (`format_of` returns `"O"` for constexpr). `~/.flaggems/code_cache/gather_rank_2.py`.
- Tensor-core to reuse: `ConvertIntentContractionToGPUMMA.cpp` (mma.sync m16n8k16, BM=64,
  BN=128, stages=3 at `:60,68`), `...ToGPU.cpp` (WMMA). Add TF32/fp16 input mode here.
- Conv: `ConvertIntentConvToGPU.cpp` — keep its OOB address math, re-emit as smem tile-fill of
  the tiled-K GEMM.
- Compaction: `ConvertIntentCompactToGPU.cpp` (block=1 thread today). Warp-scan primitive from
  the sparsemax cumsum work.
- Reduce/mover emitters + geometry: `ConvertIntentRowReductionToGPU.cpp:442-444,1217,1234-1244`,
  `MapEmit.h:312`, `ConvertIntentElementwiseToGPU.cpp`. Knob options in `Passes.td:57,566-665`.
- Flash: `ConvertIntentFlashMMAToGPU.cpp` (fwd, mma.sync — cached launcher target),
  `ConvertIntentAttentionBackwardToGPU.cpp` (bwd, WMMA→mma.sync target). Tests
  `python/tests/perf/test_flash_{fwd,bwd}_perf.py`; paper ref
  `refs/intentir/.../medium_only_realops_best_per_kernel.csv` (fwd 0.984/1.477, bwd 1.03–1.06).
- Bench harness: `python/tests/perf/bench_clusters.py` (do_bench_cudagraph loop, reusable as the
  autotuner's measurement loop), `bench_clusters_results.json`, `REPORT.md`.
- Runtime to de-brittle: `python/intentir/backend/orchestrate.py:350-589`,
  `benchmarks/row_reduction.py:54`, `backend/opgraph.py`.
- Streaming-attention atom reference (for bwd/splitK recipe):
  `refs/TianchenForge/backends/cuda/runtime_support/streaming_attention_*`.
- Coverage source of truth: `corpus/coverage.json` (162/231 = 137 fg + 25 triton verified-e2e).
