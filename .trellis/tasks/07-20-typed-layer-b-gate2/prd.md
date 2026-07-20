# PRD — Typed Layer-B domains and replay/scalar Gate2 closure

## Goal

Make the five active high-performance kernels proceed through the canonical
pipeline without weakening Gate1:

```text
fresh real LLM Lift/Relift
  -> strict Gate1 source/replay hard closure
  -> typed IntentProgram(A+B+ABI) handoff
  -> final-J replay/scalar Gate2 v2
  -> independent per-target backend conformance
```

The immediate shared blocker is that Gate1 currently accepts free-form Layer-B
domain strings which later fail the typed contract parser. KLDiv's accepted
artifact contains `b = grid_axis(0), 0 <= b < 2048`; TVD and LayerNorm have
equivalent natural-language spellings. This must be fixed at the Lift/schema
boundary so the LLM produces a genuinely executable J.

## Non-negotiable invariants

- Do not remove or soften any Gate1 hard category, threshold, or comparison.
- Do not increase numeric tolerances to make a kernel pass.
- Do not infer missing predicate/fill/alias/conflict fields or add kernel-name
  special cases.
- Official live remains fresh, uncached, `gpt-5.6-sol/xhigh`, with zero paired
  semantic interventions and no pi ablations. `wrong_fill_delta` is not on the
  acceptance path; masked-J remains a terminal anti-leakage attack.
- Gate2 v2 has exactly two required legs: final-J replay and scalar MLIR
  reference. It cannot import or fall back to CUDA/ROCm production lowering.
- Production backend conformance happens only after Gate2.
- Obsolete compatibility/umbrella paths are not restored.
- Every coherent change is tested and committed.

## Requirements

1. Define one canonical, typed portable grammar/model for Layer-B domain
   predicates and use it in both Gate1 validation and IntentProgram handoff.
2. Update the LLM prompt/schema/repair diagnostics so invalid natural-language
   predicates are rejected early and repaired by the LLM, not rewritten by
   deterministic code.
3. Preserve compatibility only for expressions that are already valid under
   the canonical grammar; fail closed on assignment/prose/source-local syntax.
4. Add generic tests for valid logical predicates and invalid forms observed in
   KLDiv/TVD/LayerNorm.
5. Re-run fresh real LLM Gate1/Gate2 for the five kernels as capability becomes
   available. Old source/replay v1 reports remain diagnostic only.
6. Make scalar execution scalable for legitimate captured shapes without a
   JSON/NumPy interpreter or kernel-name profiles. Any small profile must be a
   declared, ABI-valid specialization shared by replay and scalar.
7. After Gate2 passes, run independent target-specific lowering conformance,
   keeping common, CUDA, CPU/reference, and future backends physically separate.

## Current evidence

- Typed predicate admission landed in `05876fe`: the three observed binder/prose
  forms now fail before Gate1, enter ordinary bounded Lift repair, and cannot be
  reintroduced through post-construction mutation. The canonical specs now
  describe this current behavior; explicit typed free-symbol binding remains a
  follow-up capability rather than an inferred compatibility rule.
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
- Invalid observed domain forms fail during Lift/Gate1 artifact validation with
  actionable repair diagnostics.
- No Gate1 comparator/tolerance relaxation appears in the diff.
- At least KLDiv is re-lifted fresh through Gate1 and replay/scalar Gate2 v2;
  remaining kernels have honest terminal evidence or explicit capability gaps.
- Gate2 architecture tests prove no production CUDA dependency.
- Target lower results, when run, are reported separately from Gate2.
