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

1. The backend, given the rms_norm INTENT (not a hand-written MLIR string),
   GENERATES a single high-perf row-reduction kernel (not 8 naive kernels) and
   runs it on the 5090D.
2. Certified vs fp32 reference AND real liger (rel < 1e-3, expect ~1e-7).
3. PK kernel-only vs real liger at parity (same bar as the hand-emit task).
4. Pattern is general enough that softmax + layer_norm intents reuse the codegen.
5. Real C++ MLIR pass OR a documented codegen entry; trellis-check green; test.

## Non-goals
OptGuide schedule search; tuning beyond mirroring liger's `calculate_settings`
Layer-C hint; the loss/position-embedding kernels (later).
