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
7. `fill_value` is legal only on a load with a non-null `predicate`. An
   unconditional access omits both fields (or leaves both null); never attach a
   redundant fill to it. Consumers reject rather than normalize an orphan fill.
8. If a pure-functional boundary contains any predicate/fill, the observable
   store producer has an explicit `type` exactly matching the J output. When
   that op directly produces the output handle, keep `output_mappings` empty;
   declare a mapping only for a matching explicit function `returns` rebinding.
9. Preserve every Source program/grid-coordinate occurrence as a distinct
   domain-anchored `realize` path in a program region, even when its current grid
   extent is one. Its program-uniform path enters `address_base`; the lane/C path
   enters `address_lane`. If the two integer contributions merge before one
   pointer addition, keep both roles and their route without inventing a second
   pointer add. Never replace the program path with constant zero.
10. Every sensitive pi refinement `auth_ref` names an actually declared J
   authorization handle `u*`. It never names a node `n*`, instance `i*`, or
   prose authorization kind.
11. A C slot may optionally attach its prior to one declared J `n*`, `d*`,
   `a*`, `q*`, or `b*` using `subject_ref`. Add `axis_ref` only with a subject
   and only when that subject declares the referenced non-negative logical axis
   or `x*`; do not guess either attachment.
12. C otherwise contains only the exact trusted scheduling slots and values
   from `harness_view.json`. It contains no operator, predicate, formula,
   strategy, source/API name, or semantic constant.
13. For `numeric_cast`, authorization `subject_ref` names the explicit unary
   cast node; `source_dtype` equals its operand dtype and `target_dtype` equals
   its result dtype. The matching pi instance uses that same semantic ref and
   at most one refinement may target an instance.
14. A floating output depending on a transcendental op, floating reduction, or
   reassociation declares `result_relation.kind=tolerance`, never default or
   explicit `exact`. J declares only the class; numeric thresholds remain in
   external Q.
15. Pi contains anonymous realization relations only. It contains no dtype,
   shape, operator, formula, predicate, source fragment, frontend API, C value,
   or executable text.
16. Run the supplied static schema validator and then
   `intentir_candidate_check /output/candidate.json`. CandidateCheck is bounded
   diagnostics only; you must make every edit, and the external Runner still
   repeats canonical Decode and dependent TypeCheck before Replay and Gates.
   If the exact submitted bytes lack a prior PASS, submission invokes this same
   read-only preflight and returns any failure to you; it never repairs fields.
