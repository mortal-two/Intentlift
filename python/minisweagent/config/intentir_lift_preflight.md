# IntentIR candidate preflight

Use this short source-neutral checklist before submitting. It does not replace
the artifact schema or external TypeCheck.

1. Emit the complete top-level object with exactly `schema`, `j`, `c`, and
   `pi`; never emit an incremental repair.
2. Keep handle families separate and declaration ordered. In particular,
   `t*` denotes declared J tensors, `v*` denotes body values, `x*` denotes
   symbolic dimensions, `u*` denotes J realization authorizations, and `i*`
   denotes pi instances.
3. A symbolic `x*` is a signature/type handle. It may occur in shapes and
   logical index/domain expressions, but it is not an SSA operand or a scope
   capture/value.
4. Every `J.signature.logical_ref` names only a declared J input/output `t*`
   or symbolic dimension `x*`, never a body/local `v*`. A scalar or semantic
   constexpr used as logical data gets a dtype-matching rank-0 J input with
   shape `[]`; only an integral extent-only handle binds an `x*`. Never bind a
   value parameter to a J output just to reuse an existing handle.
5. A body operand must already be in the current lexical scope. Do not use an
   output tensor handle as if it were an already-defined intermediate; use the
   producing node's declared value handle where the grammar requires one.
6. Direct dense `index_expression` uses one bare axis for rank 1, `[]` for
   rank 0, or an ordered bracketed list such as `[k1, k0]` for rank N. Never
   use an unbracketed comma list such as `k1,k0`.
7. If a pure-functional boundary contains any predicate/fill, the observable
   store producer has an explicit `type` exactly matching the J output. When
   that op directly produces the output handle, keep `output_mappings` empty;
   declare a mapping only for a matching explicit function `returns` rebinding.
8. Every sensitive pi refinement `auth_ref` names an actually declared J
   authorization handle `u*`. It never names a node `n*`, instance `i*`, or
   prose authorization kind.
9. C contains only the exact trusted scheduling slots and values from
   `harness_view.json`. It contains no operator, predicate, formula, strategy,
   source/API name, or semantic constant.
10. Pi contains anonymous realization relations only. It contains no dtype,
   shape, operator, formula, predicate, source fragment, frontend API, C value,
   or executable text.
11. Run the supplied static schema validator. Passing it only means the JSON
   envelope is well formed; the external runner still performs canonical
   parsing, dependent TypeCheck, Replay, and Gates in a new phase.
