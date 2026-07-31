# IntentIR candidate preflight

Use this compact checklist before submitting. The supplied JSON Schema and
external TypeCheck remain authoritative; CandidateCheck diagnoses but never
constructs or repairs semantic fields.

1. Submit one complete `{schema,j,c,pi}` object. Keep all anonymous handle
   families declaration-ordered and distinct. `x*` is a symbolic dimension, not
   an SSA value; `p*` is an ABI parameter, never a logical value reference.
2. Use `shape_contract`, `shape_coordinate`, and Source-established relations to
   parameterize shapes. Never infer symbolic equality from a concrete value,
   grid, name, C, pi, or coincident dimensions. Logical scalar data is a rank-0
   `t*`; an extent-only integer may bind `x*`; a value that is also an extent uses
   one `t*` plus an explicit `value_dimension_relations` equality.
3. J must contain the complete logical dataflow, predicates, fills, domains,
   effects, recurrences, carries, control, casts, and output relations. Preserve
   explicit Source stages such as `sum` followed by division. Pi and C cannot
   repair omitted J semantics.
4. Every boundary access has one matching effect. Predicate-bearing loads keep
   their actual Source `other` value and all required consumer extensions;
   guarded writes use the real inout tensor and explicit select/carry semantics.
5. C contains scheduling slots and values only. Optional attachments name only
   declared J subjects/axes and never add semantics, Source topology, or target
   recipes.
6. Pi is only:
   `{"schema":"intentir.pi.normalization-witness.v1","program_axes":[]}`.
   Add a program-axis row only from an observed Source/TTIR launch axis for one
   declared J domain. Mixed-radix and other relations remain hard until a fixed
   unary normalizer is implemented. Pi never contains regions, instances,
   routes, order, sharing, carriers, terminal
   control, materialization, refinements, numeric permission, or exceptions.
7. Floating arithmetic outputs normally use `result_relation=tolerance`; exact
   is reserved for contracts that truly require cross-backend exact equality.
8. Run the static schema validator and `intentir_candidate_check` on the exact
   candidate bytes. Fix every reported issue yourself; final Decode, dependent
   TypeCheck, fresh Replay, Gate1, and Gate2 remain external and authoritative.
