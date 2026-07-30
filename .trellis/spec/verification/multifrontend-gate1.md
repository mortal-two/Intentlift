# Multi-frontend SCRC contract: Triton, CuTe, TileLang and CUDA

## Scope

The SCRC proof objects remain exactly `J + C + pi`. Frontend selection is a
harness decision (`U_f`) and must not add frontend syntax, source names or
compiler artifacts to any proof object. The shared lifecycle is:

```text
Source_f ──compile(profile_f)──> Observation_f(Source)
   │
   └──Lift──> J + C + pi ──source-blind Replay_f──> Replay_f
                                                │
                     compile(profile_f)─────────┘
                                                v
                               Observation_f(Replay)
                                                │
                                  strict Gate1 comparison
```

The active registry contains four audited public frontends:

| Frontend | Replay language | Level-2 compiler observation |
|---|---|---|
| Triton | Python/Triton | normalized TTIR |
| CuTe | Python/CuTe DSL | normalized MLIR plus PTX topology |
| TileLang | Python/TileLang | alpha-normalized structural TIR; normalized PTX diagnostic |
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
2. Value-producing opcode topology traced backward from stores, atomics and
   memory/control predicates. Integer value arithmetic remains hard; pointer
   arithmetic used only to form an address does not.
3. Memory-effect summaries retaining load/store/atomic/reduction kind, address
   space, width/type, multiplicity and predication while erasing cache/eviction
   hints and exact virtual-register address expressions.
5. Collective and synchronization topology: barriers, `mbarrier`, async copy,
   fences, shuffle/vote/match, MMA/WGMMA and election operations.
6. Control topology: labels, conditional/unconditional branches, calls,
   returns and exits.
7. Thread/block coordinate use: `%tid`, `%ctaid`, `%ntid`, `%nctaid`, lane,
   warp, SM and related special registers.
8. Target profile (for example `sm_120`).

The following presentation/lowering-choice information is normalized away from
the verdict but retained in the diagnostic raw topology:

- kernel and ordinary helper names;
- PTX virtual-register spelling and branch-label spelling;
- `.file`, `.loc`, `.pragma`, comments, compiler banners and source paths;
- register declaration order when it only changes alpha names;
- `.version`, `.address_size` and target banner text (the selected target is
  recorded separately by the compile case).
- cache and eviction hints such as `.nc`, `.ca` or `.cg`;
- equivalent pointer-address algebra spelling such as addition versus bitwise
  composition. Such integer operations are relaxed only when they feed a memory
  address; the same operations remain hard when they feed an observable value.

The observer must fail closure when value operations, memory kind/space/type/
predicate/multiplicity, atomic/barrier topology, warp/MMA operation, control
structure, coordinate use, ABI or target changes. Exact normalized instruction,
raw opcode and raw memory strings remain evidence but do not force equivalent
CUDA programs to reproduce cache policy or address-strength reduction. This
makes PTX a compiler observation of public CUDA semantics, not a backend-support
test and not a source-text similarity metric.

## TileLang semantic fingerprint

TileLang uses one authoritative view and one linked diagnostic view:

- the hard public TIR semantic structure, alpha-normalized while retaining
  buffer shapes, loop bounds, indices, operators, calls and control nesting;
  `serial`/`parallel`/`vectorized`/`unroll` iterator spellings normalize to one
  loop form only when the bound and body remain identical, because those
  spellings are realization schedule rather than mathematical structure; and
- the same normalized PTX fingerprint used by native CUDA, retained for audit
  and a hard schema/ABI/target safety envelope, while exact instruction,
  multiplicity, memory, collective, control and coordinate topology is
  diagnostic rather than a TileLang closure veto.

TileLang closure is therefore decided at its TIR compiler boundary. PTX can
explain a downstream lowering difference but cannot force two structurally equal
TIR programs to reproduce backend unrolling, address spelling or instruction
selection. The v3 structural AST replaces the former shallow operation-name
list as the decisive evidence. It preserves loop count, nesting, bounds,
indices, dataflow, operators and effects while erasing only iterator schedule
spelling, so relaxing PTX does not weaken mathematical loop/index/dataflow
coverage.

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

## Episode-local syntax Composer

Codex Replay v36 installs the same source-neutral Composer core in each fresh
Triton, TileLang or CUDA workspace. The only frontend profile choice is
Python-like indentation (`triton`, `tilelang`) versus C-like braces (`cuda`). The
core receives only agent-authored `line/block/raw_block` nodes and local edit
commands; it does not import artifact models or read `J`, `C`, `pi`, Source,
compiler output or Gate evidence, and it contains no op-to-API, realization-to-
skeleton or kernel-family mapping.

Composer plan, source map, checkpoint and telemetry are proof-irrelevant episode
state. The accepted object remains the exact complete Replay source after the
existing frontend ingress, compile and Gate checks. Direct source authoring and
`raw_block` remain escape paths, so extending frontend syntax does not require a
new host grammar. Normal and Mask-J arms use distinct writer/workspace/session/
Composer instances; neither arm receives the other's source, diagnostics or
checkpoint.

If a normal arm does not close, shared relift feedback selects the exact prior
Replay compiler artifact from `observer_profile.compiler_stage`, rather than
dereferencing a Triton-specific field: TTIR for Triton, TIR for TileLang, PTX for
CUDA, and MLIR plus PTX for CuTe. This is read-only counterexample transport to
the source-aware Lift owner; it does not reinterpret one IR as another or expose
the artifact to the next source-blind Replay.

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
