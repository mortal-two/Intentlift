# From-intent codegen — viability probe (2026-06-26, PROVEN)

Settles the Phase-1 A-vs-B decision: **Option B (Python emitter consuming the
IntentFunction, wired as a backend codegen entry) is viable and proven.** A C++
MLIR pass remains a valid later hardening, but B lands the user's "backend
GENERATES the kernel from intent" requirement now and matches the established
Python-orchestration pattern (`emit_mlir.py`, `flash_emit.py`, `compile.py`).

## What the probe did

Loaded `corpus/rms_norm/intent.json` → `IntentFunction.from_json`. A minimal
structural recognizer (`recognize_rms_norm`) checks the body shape: exactly one
`reduce{fn=mean, axes=[last]}`, plus `map_unary sqrt`, `map_unary recip`, and
≥3 `map_binary mul` (x·x, x·inv_rms, ·W). On match, `codegen_from_intent` pulls
`rows,cols` from `dims` + the `layer_c.tile.N` hint → thread count via
`calc_settings`, and emits the PROVEN `emit_rmsnorm` kernel.

## Result

| shape | from-intent output | rel vs fp32 ref | rel vs liger |
|---|---|---|---|
| (32768,1024) | **1 `gpu.launch` kernel** (vs 8 naive `linalg.generic`) | 1.3e-7 | 1.3e-7 |
| (16384,4096) | **1 `gpu.launch` kernel** | 1.2e-7 | 1.2e-7 |

The backend, given only the rms_norm INTENT (not a hand-written MLIR string),
generates the single high-perf row-reduction kernel and runs it correctly on the
5090D. This is the wiring the task needs.

## What's left to productionize (the actual task)

1. **Generalize the recognizer** beyond rms_norm to the row-reduction family:
   - softmax: pre = identity; reduces = {max, then sum of exp(x-max)}; post =
     exp(x-max)/sum. (Two `gpu.all_reduce`: max then add; exp via the existing
     ex2.approx peephole.)
   - layer_norm: reduces = {mean, var}; post = (x-mean)*rstd*w + b. (Two
     all_reduce-add passes; weight AND bias.) See the campaign's
     `research/liger-softmax-kernel.md` / `liger-layer_norm-kernel.md`.
   Drive the emitted kernel's pre-elementwise / reduction(s) / post-elementwise
   from the parsed intent body rather than hardcoding rms_norm.
2. **Wire as a backend codegen entry** (e.g. `compile_row_reduction(intent)` that
   the orchestrator selects when the pattern matches; fall back to the naive
   `-convert-intent-to-linalg` path otherwise). Keep it honest: only the
   recognized pattern takes the fast path.
3. Certify + PK each (rms_norm done at parity; softmax/layer_norm to follow) and
   add deterministic tests. Optionally harden into a C++ pass later.

## Decision
Proceed with **B**. rms_norm is the proven anchor; softmax + layer_norm reuse the
same codegen once the recognizer is generalized.
