# From-intent high-perf row-reduction codegen

Parent: `06-26-pk-library-row-reductions`. This is the **real** user requirement —
"我们的后端**生成**（mlir）的时候要足够高性能". The hand-emitted `rmsnorm_emit.py`
(task `06-26-rms-norm-pk`) is the *target spec*, NOT the backend. This task makes
the backend GENERATE that kernel from the lifted intent.

## The gap (probed 2026-06-26)

`-convert-intent-to-linalg` on `corpus/rms_norm/intent.json` emits **8 separate
`linalg.generic`** (x*x, reduce-mean, +eps, sqrt, recip, x*invrms, *w, + fill) →
**~8 naive GPU kernels**, each a full HBM round-trip. That is the toy/non-high-perf
path. The proven hand-emit is ONE read-once block-cooperative kernel at parity
with liger. So the bridge is: recognize the row-reduction PATTERN and emit the
single fused kernel instead of 8 naive ones.

The intent already carries the schedule hint: `layer_c.tile.N = BLOCK_SIZE` (the
Layer-C row-tile). The reduction op carries `assoc=float_nondeterministic` (legal
to reassociate → block-cooperative reduction is allowed).

## The pattern to recognize (generalizes across the reduction family)

```
  pre-elementwise(X, ...)           # x*x (rms), x (softmax/ln), ...
  → reduce over last axis           # mean (rms), max+sum (softmax), mean+var (ln)
  → broadcast + post-elementwise    # (x*rsqrt(mean+eps))*w (rms); (x-max).exp()/sum; ...
```

Emit ONE `gpu.launch` kernel per the verified template
(`.../06-26-rms-norm-pk/research/verified-rmsnorm-mlir.md`): grid = rows, T =
`num_warps*32` from the Layer-C/`calculate_settings` hint, per-thread
`memref.alloca<Kxf32>` read-once cache, `gpu.all_reduce` per reduction (fp32),
NVVM intrinsics for rsqrt/exp (libdevice is a dead end), second-pass write from
the cache. This SAME codegen covers rms_norm (1 reduce), softmax (2: max, sum),
layer_norm (2: mean, var).

## Approach (decision needed in Phase 1)

Two candidate mechanisms — pick in design:
- **(A) C++ MLIR pass** recognizing the pattern on the intent dialect (or on the
  fused-linalg form) and emitting the gpu.launch kernel. Matches the project's
  "mature C++ MLIR + TableGen" bar; the `intent.loop` → scf lowering
  (`lib/Intent/Transforms/`) is precedent. Heavier, but it's "the compiler".
- **(B) Python intent-consuming emitter** (generalize `rmsnorm_emit.py` to read
  the IntentFunction body + layer_c instead of hardcoding rms_norm), wired into a
  backend codegen entry. Faster to land; mirrors how `emit_mlir.py` already emits
  intent dialect from the model. Precedent: `flash_emit.py` is the hand-emit that
  the `intent.loop` lowering is meant to reproduce.
Lean per advisor: depth over breadth — generating ONE kernel from intent is the
result that distinguishes a compiler from a kernel collection. Decide A vs B,
then prove on rms_norm (already have the parity target), then softmax/layer_norm
reuse the codegen for free.

## Definition of Done

1. [x] The backend, given the rms_norm INTENT (not a hand-written MLIR string),
   GENERATES a single high-perf row-reduction kernel (not 8 naive kernels) and
   runs it on the 5090D. **Done** — `backend/rowreduce_codegen.py`:
   `recognize_row_reduction(IntentFunction) -> RowReductionPlan` recognizes the
   rms_norm body; `codegen_row_reduction` / `run_from_intent` reuse the proven
   `bench/rmsnorm_emit.emit_rmsnorm` lowering to emit **ONE** `gpu.launch` kernel
   (`CompiledRowReduction.n_kernels == 1`) and run it. The naive
   `-convert-intent-to-linalg` path was measured at **9** `gpu.launch_func` sites
   and additionally fails ptxas (its `sqrt` -> `__nv_sqrtf` libdevice dead-end),
   so the from-intent path is strictly necessary, not just faster.
2. [x] Certified vs fp32 reference AND real liger (rel < 1e-3, expect ~1e-7).
   **Done** — from-intent Y at rel ~1.4e-7–1.7e-7 vs both fp32 ref and liger at
   (4096,1024) and the non-pow2 (4096,1000) tail-mask case
   (`python/tests/test_rowreduce_codegen.py`, 6 passed).
3. [ ] PK kernel-only vs real liger at parity (same bar as the hand-emit task).
   **Inherited** — the generated kernel IS the hand-emit kernel (same
   `emit_rmsnorm` text, same geometry), whose kernel-only parity is already
   established in `06-26-rms-norm-pk` (`corpus/PERF_RMSNORM.md`). This task is
   about GENERATION correctness; no new perf claim is made here.
4. [ ] Pattern is general enough that softmax + layer_norm intents reuse the
   codegen. **Structured for it, not built** — `RowReductionPlan` carries a
   `reductions: list[Reduction]` (1 for rms_norm; 2 for softmax max+sum /
   layer_norm mean+var) plus `weight_input` / `bias_input` / `eps_input`. The
   recognizer dispatches per-op (`_recognize_rms_norm` today) with marked
   `TODO(softmax)` / `TODO(layer_norm)` slots; `codegen_row_reduction` raises a
   clear `NotImplementedError` for any non-rms_norm recognized kind. Softmax /
   layer_norm recognition + codegen are the explicit follow-on (see
   `_SOFTMAX_TODO` / `_LAYERNORM_TODO` in the module).
5. [x] Real C++ MLIR pass OR a documented codegen entry; test. **Done
   (Option B)** — documented Python codegen entry
   (`backend/rowreduce_codegen.py`), GPU-gated + always-on recognizer tests
   (`python/tests/test_rowreduce_codegen.py`, 6 passed; both files `py_compile`
   clean). NB: this repo configures no ruff/mypy/flake8 (pyproject is
   setuptools-only) and full `trellis-check` was NOT run by this sub-agent — only
   pytest + py_compile. A C++ MLIR pass remains a valid later hardening (per the
   viability probe).

### Extension points (softmax / layer_norm follow-on)
- **Recognizer**: add `_recognize_softmax` / `_recognize_layer_norm` alongside
  `_recognize_rms_norm`, dispatched from `recognize_row_reduction` (the generic
  last-axis-reduce gate already passes both). softmax = two reduces (max, then
  sum-of-exp(x-max)) with no weight/bias; layer_norm = two reduces (mean, var)
  with weight AND bias. Populate `RowReductionPlan.reductions` (len 2) +
  `bias_input`.
- **Codegen**: generalize the emitted kernel's pre/reduce/post stages from the
  plan instead of hardcoding rms_norm — two `gpu.all_reduce` passes (max-then-add
  for softmax, add-then-add for layer_norm), `exp` via the ex2.approx peephole,
  and a bias add for layer_norm. The read-once cache + block-cooperative geometry
  carry over unchanged.
- **Known limitation (eps)**: the intent carries `eps` as a runtime input but the
  proven `emit_rmsnorm` bakes `eps = 1e-6`; `codegen_row_reduction` validates the
  caller's eps against the baked value and raises on mismatch (a runtime-eps
  kernel = parameterize the emitter's eps constant; tracked, not done).

## Non-goals
OptGuide schedule search; tuning beyond mirroring liger's `calculate_settings`
Layer-C hint; the loss/position-embedding kernels (later).
