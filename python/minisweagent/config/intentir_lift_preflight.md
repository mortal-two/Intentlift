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
   Classify across all uses of one scalar. If it both bounds an active-domain
   predicate/index and enters arithmetic or boolean dataflow, bind one
   dtype-matching rank-0 J input `t*` and reuse that same `t*` in every such
   use. Do not split it into `x*` plus `t*`/null, specialize it to a constant,
   or synthesize a count. If Source also makes this exact integral value the J
   shape dimension `x*`, author one `J.value_dimension_relations` item exactly
   `{value_ref: t*, dimension_ref: x*, relation: eq}`. The one physical `p*`
   still binds only `t*`; the relation records the Source-owned equality.
   One physical Harness `p*` has exactly one `logical_ref`; never duplicate or
   alias that scalar into a second `p*` so one copy owns `t*` and another owns
   `x*`. Do not infer equality from a current Harness value, dtype, position, or
   observed shape. A separately bound capacity `x*` may coexist with an
   active-length rank-0 `t*` without a relation. Emit the complete relation array,
   including `[]`; fixed code, Replay, C, and pi never add or repair its members.
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
   Every `mode=inout` logical_ref names the same tensor declared with identical
   type in both J inputs and J outputs; an input-only or output-only ref is
   rejected rather than repaired. Tensor ids advance by first declaration:
   the output occurrence repeats the input `t*` handle and does not allocate a
   second handle or use `alias_of` (which only relates physical `p*` parameters).
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
7. `fill_value` is legal only on a load with an authored predicate or
   `access_guard_relation`. An
   unconditional access omits both fields (or leaves both null); never attach a
   redundant fill to it. Consumers reject rather than normalize an orphan fill.
   A computed Layer-A `i1` access guard belongs in exactly one
   `boundary.access_guard_relations` item, not copied into the access predicate.
   Its non-negative axes are strictly increasing access-target broadcast axes;
   for example a guard shaped `[M]` used by an access shaped `[M,N]` declares
   `axes:[1]`. This relation is the select guard's unique axes authority: never
   repeat the same access-node/value pair in `boundary.broadcasts`. A
   relation-guarded load has an explicit fill. Every relation-guarded store
   declares the target tensor identically in both J inputs and outputs and binds
   it once with `mode=inout`; never use an unrelated same-typed input as its old
   value. Outside a loop, anchor an explicit select whose inactive branch is the
   target tensor and map/return the output from that select result. Inside a
   loop, `carry.init_value` is exactly the target tensor, the inactive branch is
   `body_argument`, and the select result is exactly `carry.next_value`.
   Positive polarity is `select(guard,candidate,current)`; negative polarity is
   `select(guard,current,candidate)`.
   One carry has at most one guarded write in this version. CandidateCheck only
   rejects bad declarations: it never inserts select, changes output to inout,
   derives axes/polarity, or asks pi to supply any missing semantic fact.
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
   otherwise author the non-identity destination in function `returns` or
   boundary `output_mappings`. Either one is sufficient; if both are present,
   they must bind the same output to the same value. Fixed code checks this
   unique binding and never chooses or inserts its value.
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
12. A Source binary conditional that controls whether accesses/effects occur
   and then reaches a parent continuation is one Layer-A `effect_if`, not a
   value `select`. Its already-visible predicate is scalar `i1`; positive and
   negative arms are both explicit and bind distinct sibling `branch_region`
   scopes, even when either arm body is empty. The conditional itself has no
   result or yield. Arm-local pure values may feed effects in that arm but never
   escape to the sibling or parent; an arm may contain nested structured loops.
   Do not repeat effect/access fields on an arm: the bound scope's `effects` and
   matching `EffectSpec.access_ref` are the only membership authority. The same
   access cannot also have an `access_guard_relation`. Pi anchors each arm scope
   with one `realize` instance in a distinct sibling `placement=branch` region;
   it never copies predicate or polarity. CandidateCheck only rejects missing or
   contradictory authorship and never inserts an arm, scope, predicate, polarity,
   or select.
13. A Source `scf.if` that yields one value is one Layer-A `value_if`, not a
   `select` or `effect_if`. Author its visible scalar-`i1` predicate, one result
   with a closed type, and explicit positive/negative
   `{scope_id,body,yield_value}` arms. Either body may be empty and yield a
   captured outer value. Both yields exactly match the result dtype/shape; only
   that result escapes. Arms bind distinct sibling `branch_region` scopes and
   contain only known pure ordinary ops—no access/effect, unknown op, or nested
   region. Pi realizes both scopes in distinct sibling `placement=branch`
   regions and never copies predicate/polarity/yields. Fixed code never fills or
   converts this structure.
14. Every C-tiled physical region realizing a structured J `loop` has exactly
   one `pi.iterations` relation. Its `loop_instance` anchors that J loop and its
   `tile_instance` anchors the matching C tile slot. Use `extent_step` for a
   zero-to-logical-extent induction space stepped by the tile. Use
   `rounded_extent_step` for a zero-based tile-stepped range whose upper bound
   is `ceil_div(trip,tile)*tile`. Use `tile_ordinal` for a zero-based unit-step
   tile ordinal. Preserve the basis
   even when the induction variable is unused. Never place bounds, values,
   formulas, Source spelling, or frontend APIs in this relation, and do not
   invent it for an untiled loop.
   An explicit relation may use any slot role when the slot is required,
   positive-integer, and attached to the same loop/trip axis; axis ordinal `0`
   and that loop's declared `trip.axis=x*` are equivalent. The reserved roles
   `tile_extent` and `reduction_tile_extent` only let TypeCheck detect a wholly
   missing relation. A custom/misclassified role does not invalidate an
   explicit relation, but complete omission remains visible only to Gate1.
15. Every sensitive pi refinement `auth_ref` names an actually declared J
   authorization handle `u*`. It never names a node `n*`, instance `i*`, or
   prose authorization kind. Attach it only to the instance where that exact
   Source conversion occurs; never copy it to sibling parameters, strides, or
   same-dtype instances. A signed/unsigned widen includes explicit source and
   target integer dtypes of matching signedness with strictly increasing width;
   missing or equal dtype pairs are not widenings.
16. A C slot may optionally attach its prior to one declared J `n*`, `d*`,
   `a*`, `q*`, or `b*` using `subject_ref`. Add `axis_ref` only with a subject
   and only when that subject declares the referenced non-negative logical axis
   or `x*`; do not guess either attachment.
17. C otherwise contains only the exact trusted scheduling slots and values
   from `harness_view.json`. It contains no operator, predicate, formula,
   strategy, source/API name, or semantic constant.
18. For `numeric_cast`, authorization `subject_ref` names the explicit unary
   cast node; `source_dtype` equals its operand dtype and `target_dtype` equals
   its result dtype. The matching pi instance uses that same semantic ref and
   at most one refinement may target an instance.
19. A floating output depending on a transcendental op, floating reduction, or
   reassociation declares `result_relation.kind=tolerance`, never default or
   explicit `exact`. J declares only the class; numeric thresholds remain in
   external Q.
20. For a J reduction, `boundary.reductions.tile_axis` is the Source frontend's
   local tile axis after program/grid axes select the program instance. It is
   not `reduce.attrs.axes`' whole-tensor logical axis ordinal; do not copy or
   offset one to obtain the other.
21. Pi contains anonymous realization relations only. It contains no dtype,
   shape, operator, formula, predicate, source fragment, frontend API, C value,
   or executable text.
22. Run the supplied static schema validator and then
   `intentir_candidate_check /output/candidate.json`. CandidateCheck is bounded
   diagnostics only; you must make every edit, and the external Runner still
   repeats canonical Decode and dependent TypeCheck before Replay and Gates.
   If the exact submitted bytes lack a prior PASS, submission invokes this same
   read-only preflight and returns any failure to you; it never repairs fields.
