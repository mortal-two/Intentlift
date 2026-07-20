# PRD — Typed Layer-B domains and replay/scalar Gate2 closure

## Goal

Make the five active high-performance kernels proceed through the canonical
pipeline without weakening Gate1:

```text
fresh real LLM Lift/Relift
  -> Gate1 named A/B/interface/π admission
  -> executable J-only + all-causal-step π necessity
  -> strict Gate1 source/replay hard closure
  -> typed IntentProgram(A+B+ABI) handoff
  -> final-J replay/scalar Gate2 v2
  -> independent per-target backend conformance
```

The shared blocker was an ownership error: extraction parsed semantic fields
before Gate1, while some Layer-B strings still escaped to a later typed handoff.
Extraction must now retain only transport-level constructability; Gate1 must own
all semantic admission and turn precise failures into bounded source-aware
Relift, so the LLM produces a genuinely executable J without a hidden Gate0.

## Non-negotiable invariants

- Do not remove or soften any Gate1 hard category, threshold, or comparison.
- Do not increase numeric tolerances to make a kernel pass.
- Do not infer missing predicate/fill/alias/conflict fields or add kernel-name
  special cases.
- Official live remains fresh, uncached, `gpt-5.6-sol/xhigh`, with zero paired
  semantic interventions, required executable π ablations and terminal masked-J.
  `wrong_fill_delta` is not on the acceptance path.
- Gate2 v2 has exactly two required legs: final-J replay and scalar MLIR
  reference. It cannot import or fall back to CUDA/ROCm production lowering.
- Production backend conformance happens only after Gate2.
- Obsolete compatibility/umbrella paths are not restored.
- Every coherent change is tested and committed.

## Requirements

1. Extraction emits a raw `LiftCoreProposal` and checks only JSON serialization,
   closed top-level fields/container kinds and provenance. It must not parse A/B,
   infer missing fields, validate opaque ids/interface, or compose π.
2. Gate1 owns named hard admission categories for constructability, Layer A,
   Layer B, opaque identity, interface/compiler view and π capability/completeness.
3. Use one canonical typed Layer-B grammar in Gate1 and IntentProgram handoff;
   fail closed on assignment/prose/source-local syntax and drive bounded Relift.
4. Formal Gate1 executes J-only and every RuleCard-classified causal π dropout
   through a fresh writer, compile and score-free hard comparison. Canonicality
   pre-rejection is not evidence; unexpected closure rejects; incomplete
   infrastructure yields `NO_CLOSURE`.
5. Add generic tests for observed KLDiv/TVD/LayerNorm forms and the complete π
   verdict matrix.
6. Re-run fresh real LLM Gate1/Gate2 for the five kernels as capability becomes
   available. Old source/replay v1 reports remain diagnostic only.
7. Make scalar execution scalable for legitimate captured shapes without a
   JSON/NumPy interpreter or kernel-name profiles. Any small profile must be a
   declared, ABI-valid specialization shared by replay and scalar.
8. After Gate2 passes, run independent target-specific lowering conformance,
   keeping common, CUDA, CPU/reference, and future backends physically separate.

## Current evidence

- Raw proposal and Gate1-owned admission are implemented on this task branch;
  observed binder/prose forms become `layer_b_semantics` evidence and enter
  bounded Relift rather than failing in extraction or after Gate1.
- π necessity now reaches a fresh writer/compile/hard compare. RuleCards classify
  causal vs administrative steps; `canonicality_rejected` is historical only.
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

- Typed-domain validation lands with unit and integration tests.
- Invalid observed domain forms fail during Gate1 named admission with
  actionable repair diagnostics.
- No Gate1 comparator/tolerance relaxation appears in the diff.
- At least KLDiv is re-lifted fresh through Gate1 and replay/scalar Gate2 v2;
  remaining kernels have honest terminal evidence or explicit capability gaps.
- Gate2 architecture tests prove no production CUDA dependency.
- Formal live reports prove J-only plus all causal-step π probes were executed;
  every probe has one writer attempt and a compile nonclosure or hard mismatch.
- Target lower results, when run, are reported separately from Gate2.
