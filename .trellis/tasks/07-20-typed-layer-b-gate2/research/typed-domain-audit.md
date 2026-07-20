# Typed Layer-B domain audit

## Observed lowering gap

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
mapping, and prose. They cannot be consumed by the typed scalar lower and must
not be regex-rewritten. That does not retroactively invalidate Gate1's
source/replay structural evidence.

## Required design

- Reuse `intentir.ir.contracts.parse_contract_expression` as the only typed
  IntentProgram/lowering grammar.
- Add a predicate-specific check so domain/access predicates are boolean-shaped,
  not arbitrary arithmetic expressions.
- Do not invoke this parser in `Gate1Artifact` construction or before Gate1
  replay/fingerprint; doing so creates a hidden Gate0/Gate1.5.
- Parse at `intent_program_from_gate1()` for the scalar leg. On failure, retain a
  structured unavailable scalar request inside Gate2 and still execute
  source/replay.
- Improve the Lift prompt so the LLM is more likely to emit canonical predicates,
  but do not use a post-hoc parser failure as semantic feedback or repair.
- Do not parse source-realization `index_expression` as portable semantic index;
  logical index ownership remains an explicit Layer-A node/typed contract.
- Reject assignments, `where`/`with`, `program_id`, `grid_axis`, unknown calls,
  range prose, and other source-local syntax.
- Do not extract the suffix, drop the predicate, substitute `true`, or add a
  kernel-specific repair.

## Compatibility decision

Existing artifacts with strings such as `program_id(0) < 1` or
`global_ordinal in [0, N)` remain structurally readable by Gate1. They are not
silently widened into the typed grammar and cannot complete the scalar leg until
a fresh Lift produces a portable predicate. This is a Gate2 coverage gap, not a
new Gate1 artifact verdict.

## Tests

- Parser accepts canonical chained comparisons and boolean combinations.
- Parser rejects the three real natural-language/binder examples.
- Gate1 artifact construction retains invalid domain/access strings rather than
  rejecting before replay/fingerprint.
- IntentProgram handoff reports the precise path and Gate2 preserves successful
  source/replay evidence when the scalar leg is unavailable.
- Gate1 accepted fixtures remain convertible by `intent_program_from_gate1`.

## Review correction (2026-07-21)

Commit `05876fe` correctly introduced a reusable boolean-role parser, but
incorrectly applied it in `Gate1Artifact` construction. That placement rejects
the artifact before real Gate1 replay/fingerprint and therefore acts as an
unnamed extra gate. The parser and prompt guidance remain useful; the pre-Gate
admission does not.

Full free-symbol resolution remains a later portable-lowering capability: a
mature domain contract needs explicit logical-coordinate binders and a typed
environment for tensor shape/scalar symbols. It is not the first architecture
priority, and it must not be approximated by identifier blacklists, inferred
binders, or a new admission stage.
