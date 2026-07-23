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
4. A body operand must already be in the current lexical scope. Do not use an
   output tensor handle as if it were an already-defined intermediate; use the
   producing node's declared value handle where the grammar requires one.
5. Every sensitive pi refinement `auth_ref` names an actually declared J
   authorization handle `u*`. It never names a node `n*`, instance `i*`, or
   prose authorization kind.
6. C contains only the exact trusted scheduling slots and values from
   `harness_view.json`. It contains no operator, predicate, formula, strategy,
   source/API name, or semantic constant.
7. Pi contains anonymous realization relations only. It contains no dtype,
   shape, operator, formula, predicate, source fragment, frontend API, C value,
   or executable text.
8. Run the supplied static schema validator. Passing it only means the JSON
   envelope is well formed; the external runner still performs canonical
   parsing, dependent TypeCheck, Replay, and Gates in a new phase.
