# R1 — The Paper's High-Performance Backend (how IntentIR beats Triton)

Scope: `refs/intentir` (the paper repo — has `paper/`), `refs/TianchenForge` (sibling/earlier
project with the most explicit tensor-core paths), `refs/IntentIR` (trimmed public artifact).
The user's claim — "写死了很多高性能路径，至少能够 beat Triton" (hardcoded many high-perf
paths that at least beat Triton) — is **literally true but modest**: the GPU margin is
**~1.2–1.45× geomean over Triton**, not orders of magnitude. It is achieved by
**hand-written specialized kernel templates + contract-gated fast paths + host-side dispatch
among pre-tuned variants**, NOT by a TVM/Ansor-style autotuner on GPU.

---

## 0. TL;DR — the actual mechanism

There are **two different codegen strategies**, and the confusion in our project comes from
conflating them:

- **GPU (CUDA) backend = hand-written high-perf TEMPLATES + contract-gated specialization
  + host-side dispatch.** No search on GPU. Schedule (tile) is a *fixed default per op-family*,
  env-overridable. Speed comes from (a) hand-written `.cuh` tensor-core / warp-shuffle kernels,
  (b) pattern-matched specialized MLIR emitters for fused norm/reduce, (c) Layer-B certificates
  that let it drop bounds checks and vectorize (float4 / `cp.async`), (d) a runtime picking among
  a *small pre-tuned* set of tile variants.
- **RVV backend = the only place with real "retune"/search.** An **analytical GEMM roofline
  cost model** (`gemm_roofline_v0`) evaluates a small candidate tile grid and picks the best;
  Layer-C source hints are a *prior/constraint*, not a commitment. This is what the paper's
  "schedule decoupling / retune-over-freeze" story is built on.

Headline measured numbers (from `paper/manuscript/main.tex`):
- RVV: **4.48× single-thread, 3.00× 16-thread** geomean over a Triton-CPU→LLVM→RV64GCV baseline
  (8 AI-Benchmark kernels). Retune vs freeze-tiling: **1.13× geomean, up to 4.38×**, 0 regressions
  (dot tier 1.49×, reduce 1.11×, copy 1.03×).
- GPU 8-op suite (MatMul, Dropout, Softmax, LayerNorm, Correlation, Resize, RoPE, Warp):
  **1.45× H100, 1.19× RTX 5090 D** over Triton.
- GPU 28 complex ops (FlashAttention / xFormers / Liger): **1.34× H100, 1.35× RTX 5090 D**.
- Evidence (Layer-B contract) shrinks the host-dispatch candidate set **2.04×** and dispatch
  time **2.00×** (`tables/e5_cuda_evidence_off.tex`).

---

## 1. HOW the GPU code beats Triton (codegen strategy)

### 1a. Hand-written high-perf kernel templates (`.cuh`)
`refs/intentir/backends/cuda/runtime/kernels/*.cuh` — parameterized-by-C++-template, genuinely
optimized:
- `wmma_matmul.cuh` (565 lines): **tensor-core matmul** via `#include <mma.h>`, a **multi-stage
  `cp.async` software pipeline** (`CpAsyncPolicy CA/CG`, `STAGE_K`, double/triple-buffered
  `As`/`Bs` shared tiles with `AS_PAD`/`BS_PAD` bank-conflict padding), `float4`-vectorized global
  loads, template params `<TILE_M, TILE_N, STAGE_K, ...>`. `matmul_fallback.cuh` is the
  non-tensor-core path.
- `softmax.cuh` (667 lines): warp-shuffle `warp_reduce_max/sum`, `BLOCK_THREADS`-templated
  block all-reduce, and a **PTX `ex2.approx.f32`** fast-exp (`exp(x)=exp2(x·log2e)`).
- `layernorm.cuh`, `reduce.cuh` (CUB `BlockReduce` with a shuffle fallback for sm≥100),
  `rope.cuh`, `dropout.cuh`, `resize.cuh`, `correlation.cuh`, `warp.cuh`. `reduce.cuh` even
  gates CUB off on Blackwell for portability.
- `layernorm.cuh` shows the **contract-gated fast path** concretely: an `aligned` branch
  (16-byte aligned + `N%4==0`) using `float4` loads + a `FULL_M` template flag that removes the
  `if (m>=M) return;` bounds check. Misaligned/partial → scalar fallback.

### 1b. Pattern-matched specialized MLIR emitters
`refs/intentir/intent_ir/mlir/passes/lower_intent_to_cuda_gpu_kernel.py` (~28k lines) is the real
lowering. It **recognizes op subgraphs** and emits hand-tuned, vectorized GPU-dialect MLIR bodies:
`_emit_layer_norm_full_row_vector_body`, `_emit_rms_norm_full_row_vector_body`,
`_emit_fused_add_rms_norm_full_row_vector_body`, plus `early_*_v1` recognizers
(`cfg_masked_row_reduce`, `fused_linear_jsd_row_reduce`, `tvd_row_reduce`, `ai_bench_layernorm`).
Output goes MLIR → NVVM/LLVM → `mlir-translate` → PTX.

### 1c. Contract-gated specialization + host-side dispatch (the paper's stated cause)
From `main.tex` §E5 verbatim: *"The speedup arises from contract-certified in-bounds access
enabling boundary check removal and vectorized memory operations, combined with host-side dispatch
selecting among pre-tuned tile variants. Lowering generates two versions per kernel: an in-contract
specialized fast path and a conservative fallback."* Layer-B certificates (in-bounds, alignment)
are the enabler; **evidence prunes the variant search 2.04×**. This is a *dispatch-time selection
among a handful of precompiled variants*, not runtime autotuning.

### 1d. Execution path (strict, contract-first)
`backends/cuda/pipeline/driver.py` runs `legalize → shape_infer → schedule → emit → compile →
launch` over an **MLIR backend contract carrying a prebuilt PTX executable** (`cuda_ptx`, origin
must be `llvm_llc` under strict mode). Most of driver.py is PTX-ABI reconstruction (memref
descriptor arg expansion, inferring grid from `%ctaid.x` bound predicates, block from
`.reqntid`/`.maxntid`). `runtime.py` launches via bare-pointer cuLaunchKernel.

**There is NO GPU autotuner / cost model / schedule search.** The `_schedule()` stage
(driver.py:1432) is a fixed table: `matmul_conv → {64,128,32}`, else `{1,256,1}`,
`tile_n→128` when rewrite candidates exist, plus `CUDA_*_TILE_*` env overrides. That's it.

---

## 2. The hardcoded high-performance paths (enumeration)

`refs/intentir/backends/cuda/opset.py` `CUDA_SUPPORTED_OPS` (op classes with dedicated lowering):
- **Dense/contraction:** `matmul` (WMMA tensor-core + fallback), `conv1d/2d/3d`,
  `conv_depthwise2d`.
- **Attention:** `scaled_dot_product_attention` (+ TileLang `flash_attention2d`,
  `masked_attention2d` reference kernels in `kernels/tilelang/ops/`).
- **Norm/reduce/softmax:** `softmax`, layernorm/rms-norm (via specialized emitters + `.cuh`),
  `reduce_{sum,prod,max,min,any}`, `std`, `cumsum/cummax/cummin`, `argmax/argmin`.
- **Pooling:** `max_pool2d_with_indices`, `avg_pool2d`.
- **Sort/indexing/scatter:** `sort`, `unique`, `nonzero`, `gather`, `scatter`,
  `index_add/index_put`, `masked_select/masked_scatter`, `select_scatter/slice_scatter`.
- **Resampling / positional:** `resize`, `upsample_nearest{1,2}d`, `upsample_bicubic2d_aa`,
  `rope`, `warp`, `correlation`, `glu`, `polar`, `kron`, `pad`, `concat`.
- **Losses / quant:** `mse_loss`, `nll_loss_forward`, `nll_loss2d_forward`,
  `per_token_group_quant_fp8`, `quantile`, `dropout`.

`refs/TianchenForge/backends/cuda/runtime_support/` is the **most explicit hardcoded high-perf
library**, a registry of runtime "atoms": `streaming_attention_forward_runtime_support.py`,
`streaming_attention_splitk_runtime_support.py`,
`streaming_attention_tensorcore_atom_source_fragments.py` (real **mma / ldmatrix operand staging /
online-softmax flash-attention** with `block_m=16, block_k=16, fragment_n=8`, `half2` shared
query-tile, 2/4/8 query-rows-per-CTA reuse, `cp.async` KV-tile pipeline),
`dense_linear_runtime_support.py`, `rotary_position_runtime_support.py`,
`group_normalization_runtime_support.py`, `row_softmax_runtime_support.py`,
`attention_backward_runtime_source.py`. Selected via `realizer_selector.py` +
`target_support_contracts.py`.

Measured perf-vs-Triton (all geomean, over Triton baselines):
| Suite | H100 | RTX 5090 D | Source |
|---|---|---|---|
| 8-op (MatMul/Dropout/Softmax/LayerNorm/Correlation/Resize/RoPE/Warp) | 1.45× | 1.19× | main.tex §E5 |
| 28 complex ops (FlashAttn/xFormers/Liger) | 1.34× | 1.35× | main.tex, `fig/e5_cuda_complex_ops.pdf` |
| RVV 8 AI-Bench (vs Triton-CPU→LLVM) | 4.48× (1T) / 3.00× (16T) | — | main.tex abstract+§E5 |

Contract-off ablation hurts MatMul most; host-dispatch-off hurts tile-sensitive Warp/Resize.

---

## 3. Backend ARCHITECTURE — how a schedule is chosen

### GPU: template + fixed schedule + contract-gated variant dispatch (NO search)
- **Grid/block:** inferred from PTX (`%ctaid.x` bound predicate → grid.x; `.reqntid`/`.maxntid`
  → block) in `driver.py`, or output-numel fallback. High-perf kernels themselves fix
  block=`BLOCK_THREADS` template arg; row-kernels are one-block-per-row.
- **Tiling:** fixed defaults by op-family (`_schedule()`), env-overridable. Tensor-core tiles
  are compile-time C++ template params in the `.cuh`.
- **Vectorization / shared-mem / occupancy:** hand-coded in templates (`float4`, `cp.async`
  stages, padded shared tiles, CUB/shuffle reductions).
- **"Tuning":** *offline* — a handful of pre-tuned variants compiled per kernel; **host-side
  dispatch** picks one at launch; Layer-B evidence prunes the candidate set (2.04×). No online
  cost model.

### RVV: real analytical-model retune (the paper's decoupling story)
`refs/intentir/backends/spmd_rvv/analysis/`:
- `cost_model.py` → `GEMMCostModel` = **roofline** `min(compute_peak, mem_bw·AI)` with
  cache-level penalty (L1/L2/L3/DRAM working-set classification) and vec-efficiency
  `min(1, tile_n/vector_lanes)`; `search_best_tile()` picks argmax over candidates.
- `tiling_search.py` → `choose_tiles_with_bindings()` builds a small grid
  `tm∈{16,32,64} × tn=vec_lanes·{1,2,4} × tk∈{16,32,64}`, filters by L2 footprint, evaluates
  each with the cost model. Non-matmul ops fall back to backend defaults (matches paper:
  *"focuses this search on matmul-heavy kernels … other operators follow backend defaults"*).
- `tuning.py` → parses Layer-C hints into constraints (e.g. `"tile_n in (64,128)"`) so source
  schedule is a **prior**, not a commitment (freeze vs retune).
- `hardware_profile.py`, `baseline_freeze_tile.py` (the freeze baseline it beats 1.13×).

So: **there is a cost model + schedule sketch search, but only on RVV and only for GEMM-like
tiles.** No TVM/Ansor/Triton-autotuner; no learned model; deterministic analytical roofline.

### Contrast with OUR current backend (`TianchenIR`)
- Our reduction/elementwise/row ops all funnel through `RowReductionKernel`
  (`python/intentir/benchmarks/row_reduction.py`, driven by `backend/opgraph.py`,
  `backend/orchestrate.py`): **one block per row, T threads over columns, no autotuning, no
  per-op-family schedule table**. Confirmed by comments in `ConvertIntentRowReductionToGPU.cpp`,
  `ConvertIntentGatherToGPU.cpp`, `ConvertIntentLayoutToGPU.cpp`, `ConvertIntentPolarToGPU.cpp`
  ("one block per row").
- **BUT we are not purely naive:** we already have MLIR conversion passes
  `lib/Intent/Conversion/ConvertIntentContractionToGPUMMA.cpp`,
  `...GPUMMAWS.cpp` (warp-specialized producer/consumer `mma.sync`),
  `ConvertIntentFlashMMAToGPU.cpp`, `ConvertIntentSplitKReduceToGPU.cpp`, and a tensor-core
  matmul kernel in `row_reduction.py` (`nvgpu.ldmatrix + nvgpu.mma.sync`, WMMA tile, sm_120a).
- **Gaps vs paper:** (1) no per-op-family fixed-schedule table; (2) no contract-gated
  fast-path/fallback split or float4/`cp.async` vectorization gated on Layer-B alignment/in-bounds
  certificates; (3) no host-side dispatch among pre-tuned variants; (4) no analytical
  cost-model retune (our RVV/CPU path lacks the `GEMMCostModel` grid search); (5) reductions/norms
  go through generic one-block-per-row instead of warp-shuffle + `float4` + CUB like `.cuh`.

---

## 4. What the paper did for conv / sort / attention-flash / pooling

- **conv (1d/2d/3d, depthwise):** first-class in `CUDA_SUPPORTED_OPS`, `matmul_conv` op-family
  gets the tensor-core tile schedule `{64,128,32}` in `_schedule()`. Treated like a contraction.
- **attention / flash:** `scaled_dot_product_attention` supported; TileLang reference kernels
  (`flash_attention2d.py`, `masked_attention2d.py`, `_attn_fwd.py`). The real hardcoded high-perf
  flash lives in **TianchenForge** `streaming_attention_*_runtime_support.py` — a tensor-core
  (`mma`/`ldmatrix`) online-softmax flash-forward atom family with KV-tile `cp.async` pipeline,
  split-K, and multi-row-per-CTA reuse. This is what carries the **1.34–1.35× over Triton** on the
  28 complex ops (which include FlashAttention/xFormers/Liger fused kernels).
- **pooling:** `max_pool2d_with_indices`, `avg_pool2d` are dedicated ops (elementwise/reduction
  family, `{1,256,1}` default schedule) — not a headline beat-Triton claim, just covered.
- **sort / unique / nonzero / scatter:** covered as dedicated ops but **no perf-vs-Triton claim**;
  these are correctness/coverage, not the speed story.

Caveat: some of these appear as **lifting/recoverability OOS** in the eval (e.g. CUDA
`matmul_fused_epilogue2d`, `matmul_relu2d` marked `OUT_OF_SCOPE — sync/barrier witness not
implemented`; `masked_attention2d` LLM lift failure — `docs/EXPERIMENTS_RESULTS_DATA.md`). Those
are *front-end lifting* limits, not backend-perf limits, but they show the flash/fused-epilogue
paths were the hardest to certify.

---

## Key file map (for recovery)
- GPU templates: `refs/intentir/backends/cuda/runtime/kernels/{wmma_matmul,softmax,layernorm,reduce,rope,dropout,resize,correlation,warp,matmul_fallback}.cuh`
- GPU pattern-emitters: `refs/intentir/intent_ir/mlir/passes/lower_intent_to_cuda_gpu_kernel.py`, `.../emit_cuda_contract.py`
- GPU driver/schedule/launch: `refs/intentir/backends/cuda/pipeline/driver.py` (`_schedule` @1432), `backends/cuda/runtime.py`, `backends/cuda/opset.py`
- RVV retune/cost-model: `refs/intentir/backends/spmd_rvv/analysis/{cost_model,tiling_search,tuning,hardware_profile}.py`, `baseline_freeze_tile.py`
- Tensor-core flash library: `refs/TianchenForge/backends/cuda/runtime_support/streaming_attention_*` + `realizer_selector.py`
- Perf evidence: `refs/intentir/paper/manuscript/main.tex` §E5 + `tables/e5_cuda_evidence_off.tex`, `kernel_suite.tex`
