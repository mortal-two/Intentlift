# Protocol-v3 Swiglu Gate1 No-Closure Audit（冻结证据）

## Frozen outcome

- case: `liger_swiglu_forward`
- attempt: `protocol_v3_recovery`
- receipt SHA-256:
  `ae69d449cc70d7e812dbd4de4382aa164e837b6bcd8b1246211fc982de8df7c1`
- Lift: one HTTP attempt, completed and accepted by the full parser
- normal Replay: one HTTP attempt, generated and compiled
- strict Gate1: `no_closure`
- masked-J: zero calls because normal Gate1 did not close

This proves Task06's wire/parser mismatch is repaired.  The new failure is later and independent:
the accepted artifact does not carry enough source-realization information for a source-blind writer
to reproduce the source TTIR topology.

## Exact source/replay topology difference

The frozen source specialization has `stride=1024`, `n_cols=1024`, `BLOCK_SIZE=1024`, but the source
still preserves the runtime row-stride construction:

```text
program_id:i32 -> extsi:i64
p3(stride):i32 -> extsi:i64
row_base = muli(program_id_i64, stride_i64)
scalar tt.addptr(base, row_base)              # three pointers
vector tt.addptr(row_pointer, col_offsets)    # three pointers
mask = col_offsets < c0
```

The source-blind Replay instead flattened the same specialization:

```text
program_id * 1024 -> splat
flat_offsets = addi(splat_block_base, arange)
vector tt.addptr(base, flat_offsets)           # three pointers
mask = flat_offsets < 131072
```

The op-multiset delta is exact:

```text
replay extra:   arith.addi +1, arith.constant +1, tt.splat +1
replay missing: arith.extsi -2, tt.addptr -3
```

The strict comparison also records 81 dataflow differences and 6 predicate/effect differences;
interface, schedule-only and reduction components pass.  Overall score is
`0.42689393939393944`, and hard closure correctly fails.

## Why the current artifact cannot determine the source topology

- J correctly states the whole-tensor Swiglu semantics and logical accesses at `x0`.
- C correctly freezes the constexpr/compiler values.
- interface correctly retains anonymous scalar `p3`.
- current pi states regions, J/access instances, routes, order and sharing, but it has no anonymous
  relation connecting `p3` to per-program pointer bases, no scalar/vector address decomposition, no
  integer-width cast topology, and no predicate-construction placement.

Therefore the accepted `J + C + pi + interface` admits both TTIR realizations above.  A source-blind
LLM cannot uniquely recover which one the source used.  Prompting harder cannot restore information
that the wire does not contain; strict Gate1 must not normalize the difference away.

## Architectural consequence

The next task must enrich pi itself as a source-aware, anonymous realization witness.  It must not
move address/predicate realization into J or C, weaken Gate1, add a composer/RuleRegistry, or add a
Swiglu/kernel-name rule.  Lift remains the LLM that directly writes pi; Replay remains the LLM that
consumes it.  Deterministic code may validate the global shape and reference closure only.

The minimal contract must be evaluated beyond this case: runtime-stride row/column address
decomposition, mask attachment before/after a compute node (the LayerNorm counterexample), copy/
atomic/async realization, and masked-J information ablation.  Pi may carry realization arithmetic,
but tests must show that pi without J cannot recover the kernel's semantic value graph.
