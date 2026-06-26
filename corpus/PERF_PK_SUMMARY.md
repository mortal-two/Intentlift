# PK-the-library campaign — head-to-head vs the REAL kernels (RTX 5090 D, sm_120a)

The real bar (Forge E1): generate CUDA for the **same** op and match/beat the
library's **own** high-performance kernel. NOT the elementwise-fusion direction
(that exploits a cross-op gap the library left — a different comparison, retired
as off-target). Baselines are the **real** liger 0.8.0 `@triton.jit` kernels
(`pip install --no-deps liger_kernel`), called via their forward launchers /
raw kernels and JIT-run on sm_120.

## What's built (the machinery)

- **Block-cooperative reduction lowers to sm_120a** via the real MLIR toolchain
  (`gpu.all_reduce add` / `maxnumf` → mlir-opt → nvvm → ptxas). New capability —
  the flash path used only scalar carries.
- **Read-once block-cooperative row kernel**: one `gpu.launch` block/row,
  `T = num_warps*32` mirroring liger's `calculate_settings` (Layer-C hint),
  per-thread `memref.alloca<Kxf32>` caches the row ONCE (K=next_pow2(cols)/T,
  promoted to registers — no local spill), reduction(s) via `gpu.all_reduce`,
  second pass writes from the cache. HBM traffic = parity with liger.
- **NVVM intrinsics for transcendentals**: `rsqrt.approx.f` (math.sqrt/rsqrt are
  libdevice dead-ends), `ex2.approx` for exp (existing peephole). `maxnumf` is the
  f32 max kind (no `gpu.all_reduce max` in MLIR-20).
- **The backend GENERATES from intent** (`backend/rowreduce_codegen.py`):
  `recognize_row_reduction(IntentFunction)` → emit the single high-perf kernel,
  vs the naive `-convert-intent-to-linalg` path (8–9 kernels that even fail
  ptxas). 3-way mutually-exclusive recognition by reduce-count + affine role.
- **Honest measurement**: KERNEL-ONLY primary metric (both sides preallocated,
  sustained-clock warmup, interleaved do_bench, median); alloc-inclusive is a
  labeled secondary (it flatters us via allocator churn).

## Results — row-reduction family (all PARITY, all bit-faithful)

| kernel | reductions | from-intent | kernel-only ours/liger | rel vs liger | task |
|---|---|---|---|---|---|
| rms_norm | 1 (mean) | ✅ 1 kernel | ~1.00–1.03× (parity) | ~1e-7 (Y+RSTD) | 06-26-rms-norm-pk |
| softmax | 2 (max,sum) | ✅ 1 kernel | 1.015× / 1.000× (parity) | ~1e-7 | rowreduce-codegen |
| layer_norm | 2 (mean,var) | ✅ 1 kernel | 1.060× / 1.074× (parity*) | ~1.6e-7 (Y+Mean+RSTD) | rowreduce-codegen |

\* ratios within the card's ±~8% run-to-run clock variance (sub-ms kernels on the
throttled "D" part, ~290–365 GB/s band). **Parity IS the win** for memory-bound
ops on identical block geometry — no win claimed (an early rms_norm 0.77× probe
was a transient-clock artifact, retracted). Both move the same bytes; a beat
would have to be explainable, never from a weakened baseline.

`corpus/PERF_RMSNORM.md`, `PERF_SOFTMAX.md`, `PERF_LAYERNORM.md` for per-kernel
detail. Tests: `test_{rmsnorm,softmax,layernorm}_pk.py` + `test_rowreduce_codegen.py`.

## Remaining 20-kernel dataset (the campaign)

The family codegen covers the **normalization (7) + softmax (1)** core. Reuse /
extend next:
- **group_norm, poly_norm** — normalization variants, reuse the reduction codegen.
- **losses: cross_entropy, kl_div, jsd, tvd** — row reductions over logits (sum /
  KL-style); reuse the multi-reduction codegen + a different post.
- **activations: swiglu, geglu** — elementwise (the fusion machinery already
  handles these shapes; PK head-to-head vs liger's fused `silu_and_mul`/`geglu`).
- **position embeddings: rope, qwen2vl_mrope, llama4_rope** — NOT row reductions
  (per-element rotation); a different codegen family.
- **xformers (rms_norm, rms_norm_add, splitK fwd/reduce), flash_attn linear** —
  rms_norm/rms_norm_add reuse this codegen; splitK/flash are attention-family.

Sequencing per advisor: depth proven (backend generates one kernel from intent);
breadth now reuses the proven codegen. eps is currently baked per-kind (validated
against caller) — runtime-eps = parameterize the emitter (tracked, not blocking).
