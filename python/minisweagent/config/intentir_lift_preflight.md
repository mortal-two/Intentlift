# IntentIR candidate preflight

Use this compact checklist before submitting. The supplied JSON Schema and
external TypeCheck remain authoritative; CandidateCheck diagnoses but never
constructs or repairs semantic fields.

1. Submit one complete `{schema,j,c,pi}` object. Use `j.kernel_id = scrc_k0`
   and `j.intent.intent_function.name = scrc_j0`; these fixed names are not
   interchangeable. Keep all anonymous handle
   families declaration-ordered and distinct. `x*` is a symbolic dimension, not
   an SSA value; `p*` is an ABI parameter, never a logical value reference.
2. Use `shape_contract`, `shape_coordinate`, and Source-established relations to
   parameterize shapes. Never infer symbolic equality from a concrete value,
   grid, name, C, pi, or coincident dimensions. Logical scalar data is a rank-0
   `t*`; an extent-only integer may bind `x*`; a value that is also an extent uses
   one `t*` plus an explicit `value_dimension_relations` equality.
   `dimension_constraints` contains only boolean ContractExpr strings over x*
   dimensions; it never contains prose, t* values, or a value-to-dimension
   equality. ContractExpr comparisons use `==`, `!=`, `<`, `<=`, `>`, or `>=`;
   `=` is assignment syntax and is invalid. If there is no independent x*-only
   domain restriction, use `dimension_constraints: []`. A scalar used only for pointer/address/index construction or
   physical stride is representation-only and keeps `logical_ref: null` unless
   it contributes to a J output, shape, predicate, or logical operation.
   When that representation-only scalar is actually used by Source, add the
   matching `pi.access_representation_bindings` row with
   `relation:address_recomputed` (or `pointer_advanced`), its `access_ref`,
   `carrier_ref:p*`, and the exact non-negative tensor `axis_ref`. This keeps
   the p* ABI carrier out of logical J while making its address ownership
   explicit for source-blind Replay.
3. Every boundary access has one matching effect. Predicate-bearing loads keep
   their actual Source `other` value and all required consumer extensions;
   guarded writes use the real inout tensor and explicit select/carry semantics.
   Every `mode=inout` handle explicitly closes a pre-state load, post-state
   store, their read/write effects, and one canonical output binding.
4. In an ordinary selected-frontend Lift, C contains the complete typed
   scheduling declarations and `c.values` must exactly match the trusted
   selected frontend/Harness view. Preserve every declaration supplied by
   `layer_c_declarations`; map its source `slot` key to candidate C field
   `name` and never emit a candidate `slot` field. C values remain
   implementation/realization inputs and never move into J or pi. Never guess
   a value from a different frontend or target.
   Trusted local BindSource validates the selected frontend values after
   authorship. Optional
   attachments name only declared J subjects/axes and never add semantics,
   Source topology, or target recipes. Copy `subject_ref` and `axis_ref` only
   when those fields are present in the trusted declaration row; if absent,
   keep them absent. Never derive them from arange, program_id, a Source node,
   or a physical axis.
5. Pi always contains the required `program_axes` array. Keep all pi families
   empty unless anonymous Source semantics directly establish the relation and
   an existing J/C anchor proves it. A direct `program_axes` row is exactly
   `{domain_ref:d*,axis:int}`. A complex `program_bindings` row is exactly
   `{axis:int,relation:<closed relation>,logical_axes:[{domain_ref:d*,axis_ref:int|x*}]}`;
   `label` and `value` are not legal fields. Every program/iteration/control/
   materialization/access row names existing J axes/values/accesses and any
   required attached C slot. It never contains physical SSA ids, formulas,
   regions, instances, routes, target recipes, numeric permission, free text,
   allowed diffs, or semantic exceptions. A row whose fixed unary recognizer
   cannot prove remains hard. Triton Source-TTIR/observer, CuTe MLIR/PTX,
   CUDA native compiler projection, and TileLang compiler projection are
   frontend-specific compiler evidence. They may be queried for the selected
   branch, but cannot create J semantics, replace another frontend's
   realization, or authorize a pi row without a directly grounded existing
   J/C anchor.
6. Floating outputs involving transcendental functions, reductions, or
   reassociated arithmetic use `result_relation.kind=tolerance`; `exact` is
   reserved for contracts that explicitly prove cross-backend exact equality.
7. Run the static schema validator and `intentir_candidate_check` on the exact
   candidate bytes. Fix every reported issue yourself; final Decode, dependent
   TypeCheck, fresh Replay, Gate1, and Gate2 remain external and authoritative.
