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
   value parameter to a J output just to reuse an existing handle. A Harness
   argument with `kind=none` is exactly `kind=opaque`, `mode=value`,
   `dtype=none`, with no rank, alias, logical ref, J tensor/node/effect, or pi
   instance; never substitute a dummy pointer or scalar.
   A runtime scalar used only for representation/address realization is a
   rank-0 `mode=value` scalar with public dtype but no `logical_ref` or J input;
   pi may reference its p* carrier. Decide from Source use, never name/position.
   Classify across all uses of one scalar. If it both bounds a predicate/index
   and enters arithmetic or boolean dataflow, bind one dtype-matching rank-0
   J input `t*` and reuse that same `t*` in every such use. Do not split it into
   `x*` plus `t*`/null, specialize it to a constant, or synthesize a count.
   Bind `x*` only when every use is extent/index/domain-only.
   This null-carrier form is runtime-scalar-only. Every
   `bound_constexpr_values` entry is logical: bind logical data to a
   dtype-matching rank-0 J input or an integral extent-only value to `x*`.
   A pure schedule constexpr belongs to C; never leave a bound constexpr as a
   representation-only pi carrier. If it is neither logical nor pure schedule,
   the Harness case is invalid rather than something the artifact can repair.
   Every rank-0 tensor bound by `logical_ref` must contribute to an observable
   J output dependency or an explicit boundary/shape semantic relation;
   CandidateCheck rejects rather than auto-unbinding a dead logical input.
5. A body operand must already be in the current lexical scope. Do not use an
   output tensor handle as if it were an already-defined intermediate; use the
   producing node's declared value handle where the grammar requires one.
6. Direct dense `index_expression` uses one bare axis for rank 1, `[]` for
   rank 0, or an ordered bracketed list such as `[k1, k0]` for rank N. Never
   use an unbracketed comma list such as `k1,k0`. In the rectangular functional
   profile, declare every logical axis bound explicitly in `boundary.domains`.
   A padding predicate `k0 < x0` must index `k0` directly and have a matching
   domain `0 <= k0 and k0 < x0`; tensor shape, C, pi, Source, and runtime values
   never fill a missing/different domain. Preserve general logical predicates
   rather than deleting them to force this profile.
7. `fill_value` is legal only on a load with a non-null `predicate`. An
   unconditional access omits both fields (or leaves both null); never attach a
   redundant fill to it. Consumers reject rather than normalize an orphan fill.
   If a guard-free direct padding predicate and a complete zero-based
   rectangular domain use the same-axis bound as the typed tensor extent, and
   that masked axis is eliminated by `reduce(mean)`, zero fill proves only the
   numerator, not the logical valid denominator. The current boundary schema
   has no valid-count contract: preserve Source's explicit logical stages (for
   example `reduce(sum)` followed by explicit division) instead of collapsing
   them to padded `mean`. Dense/unpredicated mean, guard-bearing or cross-axis
   predicates, unrelated non-reduced masks, and unknown-op propagation are not
   this TypeCheck error; consumers still fail closed when they cannot project
   those boundaries.
   This stage rule is general: explicit Source `sum` then division remains two
   J nodes even when the divisor equals an extent. Never rewrite it to `mean`
   or replace the divisor with a generated `sum(ones)` count.
8. If a pure-functional boundary contains any predicate/fill, the observable
   store producer has an explicit `type` exactly matching the J output. When
   that op directly produces the output handle, keep `output_mappings` empty;
   declare a mapping only for a matching explicit function `returns` rebinding.
9. Preserve every Source program/grid-coordinate occurrence as a distinct
   domain-anchored `realize` path in a program region, even when its current grid
   extent is one. Its program-uniform path enters `address_base`; the lane/C path
   enters `address_lane`. If the pointer/tensor `p*` is applied at the base and
   that pointer result then receives the lane, route `p*` directly into
   `address_base` and preserve the second lane pointer/address stage. If base and
   lane merge as integers before the only pointer application, route only the
   integer/domain/C contributions through `address_base`; the pointer `p*`
   bypasses it and enters `address_lane` or the final address stage. Never decide
   from the base-to-lane edge alone. Never replace the program path with constant
   zero. Repeated occurrences anchored to the same `d*` are repeated realizations
   of the same program-axis identity, not evidence for another axis. Use a distinct
   J domain anchor only when Source really contains a distinct program coordinate.
   If one Source SSA coordinate/address intermediate is defined once and reused by
   several accesses, emit one producer with routed consumers and one-producer
   sharing; syntactic uses do not create repeated producer instances. Repeat a
   semantic ref only for distinct compiler-visible Source definitions/stages.
10. A J `contraction` uses only canonical `attrs.subscripts`; never emit
   `lhs_contract_axes`/`rhs_contract_axes`. A direct Source contraction primitive
   represented by one canonical J `contraction` node gets one
   contraction-anchored `realize` instance. If Source
   instead has separately compiler-visible outer-product/multiply/reduction stages,
   preserve them as multiple ordered instances anchored to that node. Never collapse
   the latter topology or invent the latter stages for a direct primitive.
11. If Source has a compiler-visible semantic recurrence, encode it with the
   existing structured J `loop`: declare trip/scan, every carry and initial value,
   the ordered body update, and matching next/yield/result values. Preserve the
   canonical contraction subscripts and result-axis orientation consumed by the
   update; add a transpose only for a real logical transformation. Never replace
   the recurrence with an algebraically equivalent global formula and expect pi or
   Replay to recover it. Pi has topology/routes only and cannot carry or infer an
   initialization, update formula, yield, or contraction orientation.
12. Every C-tiled physical region realizing a structured J `loop` has exactly
   one `pi.iterations` relation. Its `loop_instance` anchors that J loop and its
   `tile_instance` anchors the matching C tile slot. Use `extent_step` for a
   zero-to-logical-extent induction space stepped by the tile; use
   `tile_ordinal` for a zero-based unit-step tile ordinal. Preserve the basis
   even when the induction variable is unused. Never place bounds, values,
   formulas, Source spelling, or frontend APIs in this relation, and do not
   invent it for an untiled loop.
   An explicit relation may use any slot role when the slot is required,
   positive-integer, and attached to the same loop/trip axis; axis ordinal `0`
   and that loop's declared `trip.axis=x*` are equivalent. The reserved roles
   `tile_extent` and `reduction_tile_extent` only let TypeCheck detect a wholly
   missing relation. A custom/misclassified role does not invalidate an
   explicit relation, but complete omission remains visible only to Gate1.
13. Every sensitive pi refinement `auth_ref` names an actually declared J
   authorization handle `u*`. It never names a node `n*`, instance `i*`, or
   prose authorization kind. Attach it only to the instance where that exact
   Source conversion occurs; never copy it to sibling parameters, strides, or
   same-dtype instances. A signed/unsigned widen includes explicit source and
   target integer dtypes of matching signedness with strictly increasing width;
   missing or equal dtype pairs are not widenings.
14. A C slot may optionally attach its prior to one declared J `n*`, `d*`,
   `a*`, `q*`, or `b*` using `subject_ref`. Add `axis_ref` only with a subject
   and only when that subject declares the referenced non-negative logical axis
   or `x*`; do not guess either attachment.
15. C otherwise contains only the exact trusted scheduling slots and values
   from `harness_view.json`. It contains no operator, predicate, formula,
   strategy, source/API name, or semantic constant.
16. For `numeric_cast`, authorization `subject_ref` names the explicit unary
   cast node; `source_dtype` equals its operand dtype and `target_dtype` equals
   its result dtype. The matching pi instance uses that same semantic ref and
   at most one refinement may target an instance.
17. A floating output depending on a transcendental op, floating reduction, or
   reassociation declares `result_relation.kind=tolerance`, never default or
   explicit `exact`. J declares only the class; numeric thresholds remain in
   external Q.
18. For a J reduction, `boundary.reductions.tile_axis` is the Source frontend's
   local tile axis after program/grid axes select the program instance. It is
   not `reduce.attrs.axes`' whole-tensor logical axis ordinal; do not copy or
   offset one to obtain the other.
19. Pi contains anonymous realization relations only. It contains no dtype,
   shape, operator, formula, predicate, source fragment, frontend API, C value,
   or executable text.
20. Run the supplied static schema validator and then
   `intentir_candidate_check /output/candidate.json`. CandidateCheck is bounded
   diagnostics only; you must make every edit, and the external Runner still
   repeats canonical Decode and dependent TypeCheck before Replay and Gates.
   If the exact submitted bytes lack a prior PASS, submission invokes this same
   read-only preflight and returns any failure to you; it never repairs fields.
