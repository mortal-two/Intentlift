# PRD — Systematic max performance: whole-universe + break the walls + full backend

Task: `07-07-max-perf-systematic` · Base: `master` · Target HW: **RTX 5090D, sm_120a**, triton 3.6, CUDA 13.

## 0. Why this PRD (what was still weak)

The prior task (`coverage-and-perf-real`) beat flag_gems/Triton on **22 benched micro-ops** (conv 10.4×→1.31×, movers WIN, flash-fwd beats, autotuner matmul 1.04×). Real, but NOT satisfying because:
1. **Perf was proven on 22 ops; the other ~146 verified ops were never measured or optimized.** A high-perf-backend claim must be **systematic** — a geomean over the whole op universe, like the paper's "28 complex ops 1.35×".
2. **The norm/softmax/reduce/activation families never got the paper's real templates** (warp-shuffle + CUB BlockReduce + float4 + `ex2.approx`). M1 only did movers/pool/argmax.
3. **The autotuner (M5) + Schedule layer (M6) are half-built** — demonstrated on a few ops / 3 passes, not wired into every op's build path, not fully fork-collapsed.
4. **The declared walls (flash-bwd 1.64×, sort GATE 1.2×, cuBLAS residual) were declared, not broken.**

**Decision (user, 2026-07-07):** 全都做,最高性能,最全面 — do ALL of: whole-universe systematic perf (A) + break the walls (B) + full backend restructure (D). **End-to-end model demonstration is explicitly deferred** (not now).

## 1. Goal

A **systematically measured, comprehensively optimized** IntentIR GPU backend on sm_120a that:
1. **Benches every verified-e2e op 3-way** (ours / torch / flag_gems) — 143 flag_gems + 25 triton, a full suite with a **geomean** vs the faster reference (not 22 micro-ops).
2. **Ports the paper's real high-perf kernel templates** (warp-shuffle reductions, CUB BlockReduce, float4/`vector<4xf32>`, `ex2.approx` fast-exp) to the **norm / softmax / reduce / activation / elementwise** families — the memory-bound majority — to WIN not tie.
3. **Breaks the three declared walls** to their real hardware limit: register-resident flash-bwd + transpose-`mma.sync` → near-parity; `gpu.ballot` (emulated or a custom op) → radix GATE beats CUB; cuBLAS residual pushed via the full `wmma_matmul.cuh` cp.async-multistage recipe.
4. **Fully deploys the beat-not-tie infra**: the M5 autotuner wired into **every** op's build path (per-(op,shape,dtype) cache consulted at build, real vectorWidth/gridStyle/tile sweep axes), and the M6 Schedule layer **fork-collapsed across all ~20 passes** into one parametric codegen.
5. **No correctness regression** (all PKs stay exact); coverage held at 168/231.

## 2. What we know (from the prior task's research R1–R5, refs/ read-only)

- Paper bar ~1.19–1.45× on 5090D via hand-written tensor-core templates + contract-gated float4/cp.async + pre-tuned dispatch. The gold: `refs/intentir/backends/cuda/runtime/kernels/{wmma_matmul,softmax,layernorm,reduce}.cuh` — study + port.
- Assets shipped: M1 occupancy/coalescing primitives (`MapEmit.h` flatThreads/flatGridBlocks/emitGridStride), M2 CUB-Flagged compaction, M3/M8 conv tensor-core (fp16+tf32), M4 CachedLauncher + flash fwd/bwd + causal/LSE, M5 `autotune.py`, M6 `Schedule.h` + `launch_plan.py`, the flag_gems shim (`flaggems_shim.py` + `conftest.py`).
- Honest walls (to break, not restate): flash-bwd smem 99KB cap (transpose dV/dK); no `gpu.ballot` in MLIR gpu dialect (radix register-ranking); no wgmma/TMA on sm_120a (cuBLAS residual).

## 3. Requirements

- [ ] **Full-universe 3-way benchmark** over all 143 flag_gems + 25 triton verified ops at realistic shapes; a committed geomean vs the faster ref + a per-op table; the regression harness so any new pass re-benches.
- [ ] **Reduction/norm/softmax templates**: warp-shuffle + CUB-style block reduce + `ex2.approx` fast-exp + float4 loads in the row-reduction/norm/softmax lowering; layernorm/rms_norm/softmax/log_softmax/reduce family WIN vs flag_gems.
- [ ] **float4 vectorization** in the contiguous elementwise/mover emitter (the deferred M1 blocker): aligned `vector<4xf32>` ld/st fast path + scalar tail, for the elementwise/activation/binary families.
- [ ] **Break flash-bwd**: register-resident bwd accumulators (free the dk/dv smem) + transpose-`ldmatrix`/`mma.sync` → ≤1.2× (the §4.4 gate).
- [ ] **Break sort GATE**: `gpu.ballot` (emulate via `gpu.shuffle`+ballot-reduce or add a lowering) → warp-match register-ranking radix → beat CUB at the pow2 GATE.
- [ ] **cuBLAS push**: the contraction/matmul mma.sync to its sm_120a ceiling (full cp.async multi-stage pipeline, padded smem, the wmma_matmul.cuh recipe) — close as much of the ~1.2–1.3× residual as the silicon allows.
- [ ] **Autotuner everywhere**: `config_for` consulted in every op's build path; real vectorWidth/gridStyle sweep axes; a persisted universe-wide tuned cache.
- [ ] **Schedule fork-collapse**: unify rewrite/rewriteTiled/rewriteScan + the contraction forks into one Schedule-consuming parametric codegen; retire the regex runtime.
- [ ] No per-kernel recognizer; Dataset/refs/site-packages/key read-only; no correctness regression.

## 4. Milestone plan (measurement-first, then systematic perf, then walls, then infra)

| N | milestone | why |
|---|---|---|
| **N1** | Full-universe 3-way benchmark suite + geomean baseline | measure ALL 168 — find where the losses are before optimizing (like M0 did for 22) |
| **N2** | Reduction/norm/softmax templates (warp-shuffle + CUB + ex2.approx + float4) | the biggest systematic lever — the memory-bound majority |
| **N3** | float4 vectorization for contiguous elementwise/activation/binary | the deferred M1 blocker; broad win across ~100 ops |
| **N4** | Break flash-bwd (register-resident + transpose-mma.sync) → ≤1.2× | wall 1 |
| **N5** | Break sort GATE (gpu.ballot register-ranking radix) → beat CUB | wall 2 |
| **N6** | cuBLAS push (full cp.async multi-stage contraction) | wall 3, compute-bound ceiling |
| **N7** | Autotuner wired into every build path + universe-tuned cache | full beat-not-tie infra |
| **N8** | Schedule fork-collapse across all passes + retire regex runtime | the structural finish |
| **Final** | Comprehensive geomean report; no regression; coverage held | the systematic claim |

Ordering: N1 first (measure). N2/N3 are the systematic perf body (parallelizable by family). N4/N5/N6 break the walls. N7/N8 finish the infra. Each milestone: measured before→after vs flag_gems, all PKs green, committed.

## 5. Decision (ADR) — maximal, systematic

Do A+B+D in full; defer C (end-to-end model). Perf is the gate, measured 3-way vs flag_gems (the real Triton target); WIN where the paper's mechanism + the hardware allow; the walls are to be BROKEN to their silicon limit, and only what is genuinely irreducible (declared with fresh evidence) remains.

## 6. Out of scope

- End-to-end model/workload demonstration (deferred by user — "端到端目前还不用").
- RVV/CPU backend. Multi-GPU. Winograd conv (implicit-GEMM already at ≤1.3×).

## 7. Technical notes

- Prior research: `.trellis/tasks/07-07-coverage-and-perf-real/research/r1-r5*.md`. Perf recipe: memory `tianchenir-beat-triton-recipe`.
- Bench: `python/tests/perf/bench_clusters.py` (extend to the universe). Coverage: `corpus/coverage.json` (143 fg + 25 triton verified).
- refs templates: `refs/intentir/backends/cuda/runtime/kernels/*.cuh` (read-only, borrow technique).
