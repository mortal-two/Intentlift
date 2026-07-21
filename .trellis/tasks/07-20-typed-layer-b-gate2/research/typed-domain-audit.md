# Typed Layer-B domain audit

## Observed soundness gap

Current flow permits:

```text
LLM free-form boundary.domains string
  -> Gate1Artifact strict JSON shape validation
  -> replay can hard-close
  -> intent_program_from_gate1
  -> parse_contract_expression fails for the first time
```

Concrete accepted KLDiv final J from
`retry_liger_kl_div_after_staging_fix_01` contains:

```text
b = grid_axis(0), 0 <= b < 2048
```

It is Gate1 `replay_closed` under the old run but the current typed handoff
raises `ContractError: invalid predicate expression: invalid syntax`.

Other observed rejected candidates use:

```text
row = grid_coordinate(0) where 0 <= row < 2048
row := coordinate(grid_axis=0) with 0 <= row < 2048
```

These strings mix a logical predicate, a coordinate binder, source execution
mapping, and prose. They must not be accepted or regex-rewritten.

## Required design

- Reuse `intentir.ir.contracts.parse_contract_expression` as the only grammar.
- Add a predicate-specific check so domain/access predicates are boolean-shaped,
  not arbitrary arithmetic expressions.
- Validate domain and access predicate syntax during Lift/relift admission so a
  bad provider response enters ordinary bounded Lift repair.
- Keep a defense-in-depth check at Gate1 artifact acceptance and typed handoff.
- Do not parse source-realization `index_expression` as portable semantic index;
  logical index ownership remains an explicit Layer-A node/typed contract.
- Reject assignments, `where`/`with`, `program_id`, `grid_axis`, unknown calls,
  range prose, and other source-local syntax.
- Do not extract the suffix, drop the predicate, substitute `true`, or add a
  kernel-specific repair.

## Compatibility decision

Existing test fixtures with strings such as `program_id(0) < 1` or
`global_ordinal in [0, N)` encode the old free-text contract. Migrate them to
portable predicates rather than widening the new grammar. Legacy reports remain
JSON-readable as reports, but cannot be reaccepted as current Gate1 artifacts
without satisfying current semantic validation.

## Tests

- Parser accepts canonical chained comparisons and boolean combinations.
- Parser rejects the three real natural-language/binder examples.
- Gate1 artifact admission rejects invalid domain and access predicate paths.
- Lifter repair receives the precise semantic validation diagnostic and a
  repaired full artifact can proceed with zero interventions.
- Gate1 accepted fixtures remain convertible by `intent_program_from_gate1`.

## Phase 2.1 implementation status (2026-07-20)

Commit `05876fe` implements canonical boolean-role checking in
`ir.contracts`, applies it at Gate1 domain/access construction and typed
handoff, adds exact prompt/schema guidance, and revalidates the current mutable
artifact before every Gate1 admission round. The focused plus Gate1/Gate2/live
contract suite passes 450 tests; no comparator, hard category, tolerance,
CUDA/ROCm path, or intervention budget changed.

This closes the observed assignment/prose/source-call syntax gap. Full free
symbol resolution is still a separate capability: a mature domain contract
needs explicit logical-coordinate binders and a typed environment for tensor
shape/scalar symbols. Do not solve that gap with identifier blacklists or by
inferring binders from expression position.
