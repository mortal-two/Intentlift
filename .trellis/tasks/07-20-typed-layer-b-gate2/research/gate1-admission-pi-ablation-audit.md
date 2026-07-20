# Gate1 admission and executable pi-ablation audit

Date: 2026-07-20

## Finding

The current implementation has two acceptance mechanisms outside the intended
Gate1 proof boundary:

1. `python/intentir/extraction/lifters/scrc.py::extract_scrc_artifact` parses a
   fully semantic `Gate1Artifact`, validates iota extents and opaque IDs,
   compares J/interface/C against the trusted compiler view, materializes pi,
   and runs `RuleRegistry.validate_strict_plan`. Any failure is treated as an
   extraction repair before Gate1 sees the proposal.
2. `python/intentir/verify/gate1/pi_ablations.py::run_pi_ablations` handles
   `j_only` and `plan_step_dropout` by calling exact compose equality and never
   invokes the writer. `CANONICALITY_REJECTED` is nevertheless counted by
   `supports_pi_necessity`, so the present evidence proves tamper detection,
   not executable causal necessity.

`Gate1Runner.run` re-parses the artifact and validates the strict registry plan,
but an admission exception returns `LIFT_REJECTED` immediately. It does not
enter the bounded outer Relift loop. Consequently, merely moving a check into
this `try` block would still not improve J.

## Required boundary

```text
LiftCore closed JSON proposal + provenance
  -> Gate1 admission
       typed A/B/interface/C
       trusted compile contract
       canonical pi composition/capability
  -> bounded source-aware Relift on semantic admission failure
  -> normal source/replay hard closure
  -> executable causal pi ablations
  -> actual-pi masked-J
  -> Gate2 replay/scalar
```

The extraction boundary may still reject malformed JSON, missing closed fields,
wrong container/primitive types, unknown keys and values that cannot be retained
as a proposal. It must not decide whether a retained predicate is portable,
whether J matches the compiler view, or whether RuleRegistry can construct pi.

## Pi ablation verdict contract

- Canonical mode: full artifact pi must equal deterministic composition.
- Ablation mode: the intentionally incomplete/substituted plan must be
  structurally serializable and writer-consumable, but must not be required to
  equal canonical composition.
- Every ablation uses a new source-blind provider generation ID and receives no
  normal replay source, compiler error or comparator feedback.
- `compile_nonclosure` and score-free hard mismatch kill an ablation.
- Hard closure means the mechanism is not proven necessary and rejects the
  normal candidate.
- Provider/generation/evaluation failure is incomplete evidence and cannot pass.
- A dropped rule/card must not remain in the ablation packet in a form from which
  the writer can reconstruct the omitted construction step.
- Only causal construction steps are required. ABI declarations and frozen C
  bookkeeping are reported as administrative and excluded with an explicit,
  deterministic classification.

## Existing evidence preserved

- Gate1 hard categories and exact comparison remain unchanged.
- `wrong_fill_delta` remains absent from official acceptance.
- Masked-J remains a separate terminal anti-leakage arm and never feeds Relift.
- Gate2 remains final-J replay versus scalar MLIR only.
