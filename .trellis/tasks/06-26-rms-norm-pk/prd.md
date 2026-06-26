# rms_norm: block-reduction row kernel via real MLIR, PK liger head-to-head

Parent: `06-26-pk-library-row-reductions`. First end-to-end PK number.

## Goal

Generate a **high-performance, Triton-like rms_norm CUDA kernel via the real
MLIR-20 toolchain** (one block per row, block-cooperative reduction over the
feature dim, coalesced loads) and PK it **head-to-head against the real liger
rms_norm kernel** on the same op, same device (5090D / sm_120a). Report the
honest `ours/Triton` latency ratio and GB/s vs HBM peak.

## The op (llama casting, what liger does)

Per row x (length N), weight w (length N), eps:
1. `ms = mean(x^2)` over N   ← the ONE block reduction (sum of squares / N)
2. `rstd = rsqrt(ms + eps)`
3. `y = (x * rstd) * w`       ← elementwise, second pass over the row

liger computes the reduction in fp32 (llama/gemma casting). `rms_norm_forward`
launches `_rms_norm_forward_kernel[(n_rows,)]` — grid = (n_rows,), BLOCK_SIZE and
num_warps from `calculate_settings(n_cols)`. That IS the baseline.

## Approach

- **Hand-emit first** (retire toolchain risk, get the number): emit MLIR with a
  `gpu.launch` row kernel parameterized by (rows, cols, block_threads):
  thread-local `scf.for` partial sum-of-squares over a strided slice →
  `gpu.all_reduce add` block sum → `rstd` → second `scf.for` writes
  `y = x*rstd*w`. Static shapes (bare-ptr requires it); lower via
  `lower_memref_module_to_kernels`; launch via `backend/launch.py`.
  - PROVEN: `gpu.all_reduce` lowers to sm_120a and is correct (the row-sum probe).
  - exp peephole irrelevant here; rsqrt → check `math.rsqrt`/`arith` lowers (no
    libdevice — mirror the exp→ex2.approx rule if rsqrt needs an intrinsic).
- Then **wire into codegen from intent**: the backend should EMIT this MLIR from
  the rms_norm intent (the hand-emit is the target spec, not the deliverable).

## Definition of Done

1. Our MLIR-generated rms_norm kernel runs on 5090D, **certified vs torch fp32
   reference rel < 1e-3** (expect ~1e-6) AND vs the real liger output.
2. **Head-to-head PK vs real liger `rms_norm_forward`**: honest median-of-reps
   latency (warm JIT both sides, working set > L2 ~101MB, launch-N-then-one-sync,
   NO per-iter sync), reported as `ours/liger` ratio at ≥2 shapes (e.g.
   (32768,1024), (16384,4096)).
3. **Honest GB/s vs HBM peak** for both — show both saturate (or explain the gap).
   Memory-bound ⇒ parity (~1.0×) is the expected, legitimate win; a beat must be
   explainable, never from a weakened baseline.
4. Deterministic GPU-gated test (correctness strict; perf lenient/skipped if GPU
   absent). Results in `corpus/PERF_RMSNORM.md`. trellis-check green.

## Non-goals
OptGuide schedule search; auto-tuning BLOCK_SIZE (use liger's `calculate_settings`
policy as the Layer-C hint); the other 19 kernels (follow-on tasks).

## Research
See `research/liger-rms_norm-kernel.md` (extracted from installed liger 0.8.0).
