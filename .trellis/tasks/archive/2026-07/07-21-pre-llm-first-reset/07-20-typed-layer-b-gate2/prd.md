# PRD — Source-anchored Gate2 and typed Layer-B lowering readiness

## Goal

Make the five active high-performance kernels proceed through the canonical
pipeline without weakening Gate1:

```text
fresh real LLM Lift/Relift
  -> strict Gate1 source/replay hard closure
  -> Gate2 source <-> final-J replay
  -> Gate2 source <-> typed IntentProgram(A+B+ABI) scalar lowering
  -> independent per-target backend conformance
```

The first priority is the Gate topology itself: source runtime must remain in
Gate2, and scalar handoff/setup must not become a Gate0/Gate1.5 that suppresses
source/replay evidence. Typed coordinate/shape binding remains an important
portable-lowering capability, but it is not an earlier admission gate and is not
the first milestone.

## Non-negotiable invariants

- Do not remove or soften any Gate1 hard category, threshold, or comparison.
- Do not increase numeric tolerances to make a kernel pass.
- Do not infer missing predicate/fill/alias/conflict fields or add kernel-name
  special cases.
- Official live remains fresh, uncached, `gpt-5.6-sol/xhigh`, with zero paired
  semantic interventions and no pi ablations. `wrong_fill_delta` is not on the
  acceptance path; masked-J remains a terminal anti-leakage attack.
- Gate2 has exactly three execution legs and two required comparisons:
  `source↔replay` and `source↔scalar`. It cannot import or fall back to
  CUDA/ROCm production lowering.
- Typed predicate, IntentProgram handoff and scalar shape binding failures are
  recorded inside Gate2's scalar leg; they never prevent source/replay from
  executing and never create a new gate verdict.
- Production backend conformance happens only after Gate2.
- Obsolete compatibility/umbrella paths are not restored.
- Every coherent change is tested and committed.

## Requirements

1. Execute source, final-J replay and scalar IntentProgram lowering for every
   Gate2 case, with independent `source_replay` and `source_scalar` evidence.
2. Remove typed semantic admission from `Gate1Artifact` construction and the
   Gate1 replay/fingerprint loop. Keep ordinary structural schema validation.
3. Preserve scalar handoff/parser failures as structured Gate2 evidence while
   still executing source/replay; never repair or rewrite accepted J in Gate2.
4. Keep one canonical typed Layer-B grammar for IntentProgram/lowering and
   improve Lift prompt quality, without turning it into a pre-Gate rejection.
5. Add generic tests for all four completed dual-comparison outcomes and for
   scalar handoff unavailable with source/replay evidence preserved.
6. Re-run fresh real LLM Gate1/Gate2 for the five kernels as capability becomes
   available. Old source/replay v1 reports remain diagnostic only.
7. Make scalar execution scalable for legitimate captured shapes without a
   JSON/NumPy interpreter or kernel-name profiles. Any small profile must be a
   declared, ABI-valid specialization shared by replay and scalar.
8. After Gate2 passes, run independent target-specific lowering conformance,
   keeping common, CUDA, CPU/reference, and future backends physically separate.

## Current evidence

- `05876fe` put typed predicate parsing in `Gate1Artifact` construction. Review
  determined that this was a hidden pre-Gate admission and must be removed;
  parser failure belongs to the Gate2 scalar leg.
- `a191e97` replaced source/replay Gate2 with replay/scalar-only Gate2. The
  corrected protocol restores source and adds scalar as the second independent
  source-anchored differential.
- The active scalar reference now uses a value-independent MLIR module, one
  ranked-memref C entry for all outputs, a content-addressed toolchain-bound
  cache, and a crash/timeout-isolated worker. The old dense-input,
  per-output `mlir-runner` path was deleted rather than retained as a fallback.
  Execution provenance is report-only and does not alter Gate2 comparison.
- GeGLU: Gate1 closed; typed handoff succeeds; full capture is `[65536,256]`;
  unresolved B obligations and scalar execution scale remain.
- KLDiv: Gate1 closed after `reduction_preheader_staging`; typed handoff rejects
  the natural-language domain; old Gate2 is v1 and not current evidence.
- TVD: capture ABI and specialized unused input modeling are fixed; old J has a
  natural-language domain and requires fresh Lift.
- LayerNorm: Gate1 still has op-multiset/dataflow hard differences and its J has
  a natural-language domain; multi-output scalar execution exists.
- Rope: last real provider call returned HTTP 408; no valid final J exists.

## Completion criteria

- Three Gate2 legs and both source-anchored comparisons land with unit and
  integration tests.
- Invalid observed domain forms do not prevent Gate1 replay/fingerprint or
  Gate2 source/replay evidence; they fail the scalar leg honestly.
- No Gate1 comparator/tolerance relaxation appears in the diff.
- At least KLDiv is re-lifted fresh through Gate1 and both Gate2 comparisons;
  remaining kernels have honest terminal evidence or explicit capability gaps.
- Gate2 architecture tests prove no production CUDA dependency.
- Target lower results, when run, are reported separately from Gate2.
