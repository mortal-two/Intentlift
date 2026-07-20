# PRD — Typed Layer-B domains and replay/scalar Gate2 closure

## Goal

Make real LLM lifting produce a complete executable `J=A+B` and causal replay
construction evidence without weakening Gate1 or creating a semantic Gate0:

```text
fresh real LLM LiftCore proposal
  -> Gate1 typed A/B/interface admission + canonical pi construction
  -> bounded source-aware Relift for admission or hard-closure evidence
  -> strict Gate1 source/replay hard closure
  -> executable pi ablations + actual-pi masked-J terminal checks
  -> typed IntentProgram(A+B+ABI) handoff
  -> final-J replay/scalar Gate2 v2
  -> independent per-target backend conformance
```

The immediate architecture defect is that semantic predicate/interface/J/pi
checks currently reject or repair a candidate inside extraction before Gate1,
while direct pi dropout is counted as evidence without reaching the writer.
These checks must become named Gate1 hard admission/causal evidence.  Extraction
keeps only enough closed JSON shape and primitive types to preserve and audit a
proposal; it must not become a second acceptance authority.

## Non-negotiable invariants

- Do not remove or soften any Gate1 hard category, threshold, or comparison.
- Do not increase numeric tolerances to make a kernel pass.
- Do not infer missing predicate/fill/alias/conflict fields or add kernel-name
  special cases.
- Official live remains fresh, uncached, `gpt-5.6-sol/xhigh`, with zero paired
  semantic J/B interventions. `wrong_fill_delta` is not on the acceptance path.
  Executable pi ablations and masked-J are required terminal proof arms; neither
  may feed Relift or rescue a candidate.
- Gate2 v2 has exactly two required legs: final-J replay and scalar MLIR
  reference. It cannot import or fall back to CUDA/ROCm production lowering.
- Production backend conformance happens only after Gate2.
- Obsolete compatibility/umbrella paths are not restored.
- Every coherent change is tested and committed.

## Requirements

1. Keep LiftCore parsing limited to closed JSON/object/array/primitive
   constructability and provenance capture. Move portable predicate typing,
   compiler-view/interface agreement, semantic J references, scope/storage,
   iota extent, C strategy and pi capability/completeness into Gate1 admission.
2. Emit structured Gate1 admission evidence and route semantic admission
   failures through the same bounded source-aware Relift policy. Provider
   transport failures remain terminal and never masquerade as semantic repair.
3. Treat RuleRegistry only as an executable pi mechanism catalogue. Compose the
   canonical pi inside Gate1; zero/ambiguous rule coverage is a Gate1 capability
   failure with actionable Relift evidence, not an extraction-time whitelist.
4. Keep one canonical typed portable grammar for Layer-B predicates and use it
   in Gate1 and IntentProgram handoff. Assignment/prose/source-local syntax must
   fail closed in Gate1; deterministic code must not rewrite or infer it.
5. Separate canonical pi validation from ablation execution. A structurally
   executable step dropout/substitution must enter a fresh source-blind writer,
   compile, and undergo the same score-free hard comparison. Writer-before
   rejection, schema invalidity or provider failure is inconclusive, not a kill.
6. Require every applicable causal pi ablation to be killed by compile
   nonclosure or hard fingerprint nonclosure. A surviving ablation rejects the
   normal candidate. Classify administrative ABI/schedule steps separately so
   the causal claim is neither vacuous nor overstated.
7. Preserve actual-pi masked-J as an independent terminal attack after normal
   hard closure and pi ablation proof. It cannot modify normal J, enter Relift,
   or compensate for any other failed arm.
8. Add generic tests for valid and invalid predicates, admission-driven Relift,
   compose capability gaps, executable dropout/substitution, inconclusive
   evidence and mutation survival.
9. Re-run fresh real LLM Gate1/Gate2 for the active kernels, then extend the same
   evidence contract to all 28 high-performance kernels. Old source/replay v1
   reports remain diagnostic only.
10. Make scalar execution scalable for legitimate captured shapes without a
   JSON/NumPy interpreter or kernel-name profiles. Any small profile must be a
   declared, ABI-valid specialization shared by replay and scalar.
11. After Gate2 passes, run independent target-specific lowering conformance,
   keeping common, CUDA, CPU/reference, and future backends physically separate.

## Current evidence

- Typed predicate parsing landed in `05876fe`, but it currently executes in
  `Gate1Artifact` construction and therefore rejects before Gate1. This is now
  an identified architecture defect to migrate, not the desired final design.
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

- Extraction no longer performs semantic acceptance or canonical pi coverage.
- Invalid observed domain forms and J/interface/pi capability gaps appear as
  structured Gate1 admission evidence and can drive bounded source-aware Relift.
- Direct pi dropout/substitution reaches a fresh writer and compiler; formal
  evidence never counts pre-writer canonicality rejection as a successful kill.
- Official live requires applicable executable pi ablations plus masked-J, while
  retaining zero paired semantic interventions and no `wrong_fill_delta` path.
- No Gate1 comparator/tolerance relaxation appears in the diff.
- The active five kernels are re-lifted fresh through Gate1 and replay/scalar
  Gate2 v2 under the revised evidence protocol; remaining campaign kernels have
  honest terminal evidence or explicit capability gaps until all 28 close.
- Gate2 architecture tests prove no production CUDA dependency.
- Target lower results, when run, are reported separately from Gate2.
