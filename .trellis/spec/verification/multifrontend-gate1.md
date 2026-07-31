# Multi-frontend SCRC contract: Triton, CuTe, TileLang and CUDA

## Scope

The SCRC proof objects remain exactly `J + C + pi`. Frontend selection is a
harness decision (`U_f`) and must not add frontend syntax, source names or
compiler artifacts to any proof object. The shared lifecycle is:

```text
Source_f ──compile(profile_f)──> Observation_f(Source)
   │
   └──Lift──> J + C + pi ──source-blind Replay_f(J,MaskC)──> Replay_f
                                                │
                     compile(profile_f)─────────┘
                                                v
                               Observation_f(Replay)
                                                │
                         profile-scoped pi normalization
                           and J/C structural equality
```

The active registry contains four audited public frontends:

| Frontend | Replay language | Level-2 compiler observation |
|---|---|---|
| Triton | Python/Triton | normalized TTIR |
| CuTe | Python/CuTe DSL | normalized MLIR plus PTX topology |
| TileLang | Python/TileLang | normalized TIR operation topology plus normalized PTX |
| CUDA | standalone CUDA C++ | normalized PTX semantic topology |

Source and Replay are always compiled with the same frontend compiler case,
target and compiler options. Schedule values remain owned by `C`; the
fingerprint records them for audit but does not reinterpret them as `J`.

## CUDA semantic fingerprint

Raw CUDA source and raw PTX text are not semantic fingerprints. They contain
source spelling, debug paths, virtual-register allocation and label choices
that are irrelevant to SCRC closure. The CUDA observer compiles both arms to
PTX and builds the following hard views:

1. Positional kernel ABI: PTX parameter types/alignment/extents, plus the
   anonymous `J.signature` kind/dtype/rank/mode projection.
2. Alpha-normalized instruction/dataflow topology.
3. Opcode multiset including semantic modifiers (rounding, saturation,
   approximation, vector width and data type).
4. Memory effects, retaining load/store/atomic/reduction opcode, address
   space, width, address expression and instruction predicate.
5. Collective and synchronization topology: barriers, `mbarrier`, async copy,
   fences, shuffle/vote/match, MMA/WGMMA and election operations.
6. Control topology: labels, conditional/unconditional branches, calls,
   returns and exits.
7. Thread/block coordinate use: `%tid`, `%ctaid`, `%ntid`, `%nctaid`, lane,
   warp, SM and related special registers.
8. Target profile (for example `sm_120`).

The following presentation-only information is normalized away:

- kernel and ordinary helper names;
- PTX virtual-register spelling and branch-label spelling;
- `.file`, `.loc`, `.pragma`, comments, compiler banners and source paths;
- register declaration order when it only changes alpha names;
- `.version`, `.address_size` and target banner text (the selected target is
  recorded separately by the compile case).

The observer must fail closure when memory space, predication, atomic/barrier
topology, warp/MMA operation, control structure, coordinate use, ABI or target
changes. This makes PTX a compiler observation of public CUDA semantics, not a
backend-support test and not a source-text similarity metric.

## TileLang semantic fingerprint

TileLang uses two linked views:

- the public TIR/TileLang operation sequence (`T.Kernel`, loop forms, copy,
  allocation, reduction, GEMM and other public operations), with identifiers
  removed; and
- the same normalized PTX fingerprint used by native CUDA.

TIR prevents target code alone from erasing a frontend realization difference;
PTX prevents a shallow list of TileLang API calls from accepting a different
memory/control implementation. Both views are hard Gate1 evidence.

## Replay safety and compiler isolation

- CUDA Replay accepts one `extern "C" __global__ scrc_k0`, optional device
  helpers and a fixed public-header allowlist. Host functions, launch code,
  inline PTX, file/process/network I/O and arbitrary preprocessor directives
  are rejected before `nvcc`.
- TileLang Replay accepts one direct `@T.prim_func scrc_k0`, the canonical
  public `T` import and public `T.*` operations. Host I/O, reflection, dynamic
  execution, arbitrary imports and dunder access are rejected before a fresh
  compile worker imports the module.
- Neither compiler path launches a kernel during Gate1.

## Gate2 execution

The shared three-edge Gate2 judge can receive native CUDA or TileLang runners.
CUDA launches the Gate1-profile PTX through the CUDA driver on isolated Torch
tensors. TileLang compiles the accepted PrimFunc to a callable JIT kernel.
Source and the exact Gate1-accepted Replay each receive cloned inputs, then the
existing external-Q comparisons check Source↔Replay, Source↔J and Replay↔J.

## External Dataset

Source datasets live outside the code repository. The default root is the peer
`/home/kingdom/phdworks/Dataset`, overridable with `INTENTIR_DATASET_ROOT`.
`TianchenIR-MultiFrontend/` contains 31 TileLang kernel factories, four
handwritten standalone CUDA kernels and 31 CUDA source/PTX snapshots exported
from those TileLang factories. The export metadata records the origin case,
compiler version, target, launch topology, parameter shapes/effects and content
digests, so the generated CUDA corpus is traceable rather than counted as an
independent handwritten corpus.

The catalog stores source evidence plus the deliberately versioned CUDA/PTX
compiler snapshots. Replay source, certificates, compiler caches, probe logs
and run reports remain ephemeral artifacts and must not be written back into
Dataset.

Dataset membership is not itself a pass claim. A case is runnable only under a
recorded compatible compiler profile and a complete Harness descriptor. This
distinguishes frontend support from version-specific kernel readiness.

`tools/frontend_dataset.py --probe-tilelang` and `--probe-cuda` perform the
corresponding compile-only readiness audits. The Dataset is pinned to TileLang
0.1.8; no package upgrade is required. Fourteen factories were repaired at the
Dataset boundary by capping invalid thread layouts, replacing unstable
fragment-to-global copies with direct stores, using unambiguous serial loop
variables and retaining semantics-first serial attention realizations where
0.1.8 cannot lower the original parallel fragment schedule. The repaired
attention kernels still implement causal masking and stable/online softmax;
their parallel schedules remain future `C` choices rather than changes to `J`.

On the integration host, all 31 TileLang factories compile under TileLang
0.1.8. CUDA 12.8 then recompiles all 35 CUDA sources (31 traceable TileLang
exports plus four handwritten kernels) for `sm_120`. Compile readiness is still
not an SCRC closure claim: an individual Source↔Replay result additionally
requires its accepted `J + C + pi`, Harness binding and strict Gate1/Gate2
evidence.

## J/C structural observer boundary (Task 20)

The common π wire syntax, relation-kind names and ownership labels remain closed.
Their applicability and observer denotation are profile-relative rather than
frontend-independent. Each observer profile may implement a versioned,
deterministic partial unary normalizer for only a declared subset of π relation kinds:

```text
Normalize_pi(profile, observation, J, C, pi)
    -> normalized_observation | undefined
```

The same `Normalize_pi` is called independently on Source and Replay; one call
cannot inspect the peer observation or their diff. It is idempotent and fail-closed
on ambiguous anchors or unsupported relations. Its fixed profile rules cannot be
extended by a candidate or by a per-kernel prompt. J-owned value/dataflow,
predicate/fill, reduction, effect, control/post-state and output facts remain hard
observation; π cannot normalize them. A receipt may report the shorthand
applied/unsupported relation list for debugging, but the formal Gate1 verdict is
definedness and equality of the two normalized observations.

A candidate relation is not effective merely because it is well typed. Each
observation must independently match the exact typed anchor and fixed local canonical
form, and every rewrite must preserve the profile's fixed J/C-owned hard projection.
A failed match leaves the observation hard or returns `undefined`.

An active profile must publish a candidate-independent `ProfileOK` contract: its hard
projection covers every J/C-owned atom class the profile claims to observe, unknown
atoms remain hard, and each enabled unary rewrite has a mechanical hard-preservation
check or an independently audited rule. TypeCheck only checks the candidate π
instance; it does not establish this profile property.

Every relation kind is non-normalizing by default for a new profile. Promotion
requires a profile-wide unary canonical form and hard-preservation contract;
all other and all unknown observation atoms remain hard. Rule selection cannot
inspect a kernel identity, benchmark ordinal or source path. Generic typed
rank/dtype and J-authored symbolic-shape premises are permitted, but equality of
the current concrete fixture alone is not a relation premise.

The four profiles need not share the same raw diff serialization. They must each
record the supported π relation subset, observer/canonicalizer versions and
unsupported cases. A relation unsupported by a profile remains hard or produces
`observer_unsupported`; it is never borrowed from another frontend. This keeps
multi-frontend work per observer profile rather than per kernel, without claiming
that the four profiles have identical observation strength. The common wire is a
serialization and ownership contract only; an identically named relation kind is
not the same observer function across profiles, and certificates do not transfer.
