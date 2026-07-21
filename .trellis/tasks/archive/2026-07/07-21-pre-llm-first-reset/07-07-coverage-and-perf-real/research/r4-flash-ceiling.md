# R4 — Flash / attention: the HONEST sm_120a ceiling (not an excuse, not an overclaim)

**Question (user):** "prd说oos就不做么" — does the PRD-OOS label mean we skip flash/attention?
**Short answer: NO. Flash was NOT skipped.** The 29-kernel task shipped real flash fwd + flash
bwd lowering passes, PK'd for correctness and benched head-to-head against the *real* Triton
flash-attn kernel. Two *different* "can't" statements were conflated into one excuse; this doc
separates them.

---

## 0. The two claims that got conflated (the honesty pivot)

| Claim | Scope | Verdict |
|---|---|---|
| **"flash/attention = PRD-OOS"** (`corpus/coverage.json:121,367,1063`) | the **automated LLM extraction→lower coverage pipeline** does not *auto-lift* flash | **True but narrow** — means "the auto-pipeline won't recover it," NOT "the project can't do it." Hand-authored passes DO cover it. |
| **"warp-spec deadlock ceiling on sm_120a"** (`test_matmul_mma_ws_pk.py:22-35`) | the **M7g producer/consumer warp-specialized *matmul* experiment** | **Sound, but about ONE optimization technique** — and flash never needed it. |

Neither claim says "flash is infeasible." Flash reaches **near-parity fwd** and **~1.8× bwd**
*without* warp-spec. The OOS label is about the auto-extraction harness, not about capability.

---

## 1. What the 29-kernel task ACTUALLY shipped (grep-verified)

Three real lowering passes + eight test/lit files:

- **`lib/Intent/Conversion/ConvertIntentFlashMMAToGPU.cpp`** (641 L) — M8c/M8d **register-resident
  flash forward**. Online-softmax state (O/m/l) stays in registers; only Q/K/V touch shared;
  S=Q@Kⱼᵀ and O+=P@Vⱼ on `nvgpu.mma.sync` (m16n8k16), per-row rowmax/rowsum via quad
  `gpu.shuffle`. M8d adds a `cp.async` N-stage (default 2) K/V staging pipeline.
- **`lib/Intent/Conversion/ConvertIntentAttentionBackwardToGPU.cpp`** (617 L) — M9 **flash
  backward**, the full 5-contraction fusion (recompute P, dV=Pᵀ@dO, dP=dO@Vᵀ, dS, dK=dSᵀ@Q,
  dQ=dS@K) as one `intent.loop`. **Dual DQ mode**: `serial` (one block/head, M9a baseline) and
  `atomic` (grid.x = cdiv(Sk,BN), per-key-block programs `atomic_rmw addf` into pre-zeroed DQ).
  Matmuls on **WMMA** (not mma.sync — a known gap, see §3).
- **`ConvertIntentLoopContractionToGPU.cpp`** — M8a shared-memory flash fwd (round-trips through
  shared; the M8c register-resident pass is the perf sibling).
- Tests: `test_flash_fwd_pk.py`, `test_flash_fwd_mma_pk.py`, `test_flash_fwd_perf.py`,
  `test_flash_bwd_pk.py`, `test_flash_bwd_dualmode_pk.py`, `test_flash_bwd_perf.py`,
  `test_flash_emit.py`, `test_flash_interpret.py`; MLIR lit in `test/lower-flash-*.mlir`,
  `test/attention.mlir`, `test/flashattention_fwd.intent.json`.

All correctness-gated: fwd rel < 2e-2 vs fp32 ref AND vs torch SDPA; bwd rel < 2e-2 on dQ/dK/dV
vs fp32 autograd.

---

## 2. Measured perf vs the REAL Triton flash-attn (committed, gated tests)

Both perf tests drive the **bare `@triton.jit` `_fwd_kernel` / `_bwd_kernel`** from
`Dataset/flash-attention/flash_attn/flash_attn_triton.py` (Dataset read-only; Triton-3.x
`trans_b=` patched into a local temp copy), fixed config, kernel-only `do_bench` median.

### Forward (`test_flash_fwd_perf.py`) — 5090D, honest committed trend
mine/triton gap **SHRINKS with size**: ~2–3× @128 → ~1.5–2× @256 → ~1.2–1.6× @512 →
**~1.0–1.2× @1024 (near-parity at the largest shape)**. The small-shape gap is **per-call Python
launch overhead** (`kern.run` rebuilds launch args each call; Triton caches its launcher),
inflating sub-30µs kernels — *not* a kernel-efficiency gap. The perf assert is gated only on
GPU-bound sq≥512 (loose `mine ≤ triton*2.5`, a regression guard, NOT a parity claim).
- **This does NOT reproduce the PRD's "parity small / 1.38–1.43× large"** — the trend is
  *inverted*; the committed file says so explicitly (lines 23-32).

### Backward (`test_flash_bwd_perf.py`) — 5090D
- serial (M9a single-SM baseline) → **atomic 6.7× @512, 14.5× @1024** faster (key-block parallel).
- **atomic vs Triton `_bwd_kernel`: ~1.4–1.6× off same-tile (64×64, 8-warp) config; ~1.8–1.9× off
  Triton's best (128×64)**. Loose gate `atomic ≤ triton*3.0`.
- Named residual (honest, file lines 23-27): inherited **wmma.load gap (M7)** + **5-matmul serial
  chain** (barrier between each) + **4 warps vs Triton's 8**. Grid size is NOT the problem
  (8–16 blocks either way).

### Paper reference numbers (`refs/intentir/.../medium_only_realops_best_per_kernel.csv`)
`ratio_5090d / ratio_h100` (>1 = speedup over Triton):
- flash `_fwd_kernel`: **0.984 / 1.477** → *parity on 5090D*, speedup on H100.
- `_flash_attn_backward`: 1.062 / 1.139 ; `_bwd_kernel`: 1.032 / 1.143 ; `_bwd_preprocess`: 1.94.
- flash `linear.kernel_fwd`: 1.290 / 1.597 ; xformers `_fwd_kernel_splitK`: 1.369 / 1.347 ;
  `_splitK_reduce_varargs`: 1.159 / 1.425.

**The paper's own 5090D flash-fwd number is 0.984 (parity).** That is the honest achievable bar
on this hardware — the current committed impl reaches it at large shapes and leaves launch
overhead on small shapes.

---

## 3. Is the warp-spec ceiling SOUND or an excuse? — SOUND, but mis-cited if used against flash

The M7g experiment (`ConvertIntentContractionToGPUMMAWS.cpp`, `test_matmul_mma_ws_pk.py`) is a
**matmul** producer/consumer warp-spec via `nvgpu.mbarrier` spin-waits. Findings:
- **GATE G5 passed**: mbarrier init/arrive/try_wait.parity + named `bar.sync` DO lower and execute
  on sm_120a — warp-spec is *expressible*.
- **Correct only at occupancy 1** (grid ≤ 170 SMs). At **occupancy ≥ 2 the spin-wait
  producer/consumer DEADLOCKS**: co-resident blocks' spinning consumer warps starve the producer's
  issue slots — **no hardware forward-progress guarantee on sm_120a; Blackwell consumer lacks
  Hopper's warpgroup-MMA / TMA async-proxy that make this schedule safe**. Threshold = SM count.
- **Even single-wave it's SLOWER**: 512³ WS ~16.4µs (1.49× tl.dot) vs non-WS mma.sync 10.2µs
  (0.93× tl.dot). One producer warp bottlenecks loads; the extra warp costs occupancy.

**Verdict:** the warp-spec ceiling is a *sound, correctly-abandoned* result for the Hopper-style
pipelining technique. **BUT it is NOT the reason flash trails** — flash fwd/bwd never use warp-spec;
they use plain-barrier register-resident mma.sync (fwd) / WMMA (bwd). Citing the warp-spec deadlock
as flash's ceiling **would be an excuse**. The real flash residual is ordinary impl choice, not a
hardware wall (§4).

### The GENUINE sm_120a compute ceiling (the honest wall)
M7f best-tuned mma.sync matmul vs `tl.dot`: **0.93 / 1.30 / 1.19** (512³ / large / large). It
*beats* tl.dot at 512³ and trails **1.19–1.30×** on large compute-bound shapes. That 1.2–1.3×
residual vs cuBLAS-class throughput **is** the honest tensor-core ceiling — closing it needs Hopper
wgmma+TMA the hardware doesn't have. Flash fwd inherits this per-tile; near-parity vs *Triton* flash
holds because Triton flash is itself not cuBLAS-optimal.

---

## 4. The honest achievable bar for flash on sm_120a / 5090D

| Path | Current committed | Achievable bar | What closes the gap (named mechanism) |
|---|---|---|---|
| **flash FWD** | near-parity @1024, launch-noise @small | **Parity** (paper: 0.984× on 5090D) | cached launcher (kill per-call arg rebuild) → parity at *all* shapes. NOT a kernel change. |
| **flash BWD** | atomic ~1.8–1.9× off Triton-best | **Parity** (paper: ~1.03–1.06×) | (1) bwd matmuls **WMMA→mma.sync** (port M8 fwd recipe); (2) **8 warps** not 4; (3) pipeline the 5-matmul chain (fewer full barriers). All ordinary — NOT warp-spec. |
| **compute residual vs cuBLAS** | mma.sync 1.19–1.30× on large GEMM | **Irreducible ~1.2–1.3×** | Would need Hopper wgmma+TMA. This is the real ceiling — declare it, don't chase it. |

**Bottom line:** parity with Triton flash-attn (fwd + bwd) is achievable on this hardware and the
paper reference proves it; the current impl is at parity on fwd-large and ~1.8× on bwd with a
*named, ordinary* closing path. The only irreducible wall is the ~1.2–1.3× tensor-core residual vs
cuBLAS-class GEMM throughput (missing Hopper async-MMA/TMA) — and Triton flash sits under the same
wall, so flash-vs-Triton parity is unaffected by it.

---

## 5. flag_gems attention / flash_api / flash_mla / scaled_softmax — cover + beat?

Inventory (FlagGems_sophgo `src/flag_gems`):
- **`ops/attention.py`** — ONE compute kernel `_attn_fwd` (Triton-tutorial fused attention **forward**;
  causal `STAGE` 1/2/3 diagonal masking, optional `attn_mask` bias, `fp8_v`, autotuned), exposed as
  `scaled_dot_product_attention`. **Forward-only** (no bwd kernel in flag_gems).
- **`fused/concat_and_cache_mla.py`** — this is the only "flash_mla"/"MLA" artifact: a **KV-cache
  write (scatter)**, NOT attention compute. Memory-bound.
- **No** separate `flash_api` / `flash_mla` compute kernel / `scaled_softmax` kernel exists here.
  "scaled_softmax" = softmax-with-scale = already covered by the M1 softmax pass.
- `flash_mla` / `attention` / `flash_kernel` appear in `corpus/coverage.json` only as **auto-pipeline
  OOS** markers (loop authorship), consistent with §0.

### Doable vs genuinely loop-authorship-hard
- **Already shipped (the hard loop-authorship part is DONE):** fused online-softmax fwd, 5-contraction bwd.
- **Incremental (doable, no new hardware capability):** causal diagonal masking (STAGE logic),
  `attn_mask` additive bias, batch/head dims, LSE output, xformers split-K + splitK-reduce
  (varargs). These are loop-shape extensions of passes that already exist.
- **Memory-bound / easy + beatable via fusion:** `concat_and_cache_mla` / `reshape_and_cache` KV-cache
  scatter, scaled-softmax.
- **Genuinely uncertain on sm_120a:** the `fp8_v` attention path (`tl.float8e4nv` codegen on Blackwell
  consumer is unverified — gate to fp16/bf16, same call the 29-kernel RISKS doc makes).
- **Irreducible:** the last ~1.2–1.3× vs cuBLAS-class GEMM (tensor-core ceiling). Not closable; also
  not needed to reach *flag_gems/Triton* parity, since `_attn_fwd` is an autotuned Triton kernel under
  the same ceiling.

**To beat flag_gems `_attn_fwd`:** it's an autotuned Triton fused-attention fwd, so parity is the
realistic honest bar; a *beat* must be named — fewer HBM bytes via deeper fusion (e.g. folding a
following op) or better launch geometry, not raw tensor-core throughput.
