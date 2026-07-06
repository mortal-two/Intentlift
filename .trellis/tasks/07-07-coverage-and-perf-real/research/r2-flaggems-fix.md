# R2 — Fix flag_gems benchmarking (triton 3.6 / sm_120a launcher ABI mismatch)

**Status: SOLVED and VALIDATED.** flag_gems now runs 100x+ in-process (including under
`do_bench_cudagraph`) with a 1-line monkeypatch shim in our code.

---

## 1. Diagnosis — the missing arg is a **constexpr**, NOT `launch_pdl`

### Symptom (reproduced)
```
call 0: OK
call 1: FAIL TypeError: function takes exactly 32 arguments (31 given)
  .../flag_gems/utils/libentry.py:783   kernel[grid[0:3]](*k_args.values())
  .../triton/compiler/compiler.py:498   runner -> self.run(...)
  .../triton/backends/nvidia/driver.py:713   self.launch(... self.launch_pdl, ...)
```
The **first** call per process succeeds; every subsequent call throws. That is why the
committed correctness PKs pass (they call each flag_gems op exactly once) and why the
perf harness could never get a steady-state number.

### Root cause (exact)
Two version gates in `flag_gems/utils/libentry.py` `LibEntry.run` only cover triton
**3.3–3.5** and wrongly exclude **3.6**:

- Line 678 / 691 — building `k_args`:
  `if major_version == 3 and 3 <= minor_version <= 5:` → **constexpr args are added to
  `k_args` only for 3.3–3.5**. On 3.6 `k_args` omits every constexpr param.
- Line 763 — choosing the launch path:
  same gate. On 3.6 it falls to the `else` at line 782–783:
  `kernel[grid[0:3]](*k_args.values())` — i.e. it launches with the constexpr-less
  `k_args`.

Meanwhile triton **3.6's** nvidia launcher expects the constexpr params to still be
present as placeholder `PyObject*` (`"O"`) slots:
- `triton/backends/nvidia/driver.py:123` `_BASE_ARGS_FORMAT = "iiiKKppOOOOOO"` → **13
  fixed** args = gridX/Y/Z, stream, function, `launch_cooperative_grid`, **`launch_pdl`**
  (the 2nd `p`), global_scratch, profile_scratch, kernel_metadata, launch_metadata,
  enter_hook, exit_hook.
- `format_of()` (line 186–207) returns `"O"` for **every** signature entry **including
  `constexpr`** (line 192–193). So the PyArg_ParseTuple format length = `13 + len(full
  signature incl. constexpr)`.

For the failing `gather_rank_2` kernel: 18 non-constexpr params + **1 constexpr**
(`BLOCK_SIZE_N`, from `@triton.heuristics`). Launcher wants `13 + 19 = 32`; the cached
path passes `13 + 18 = 31`. **The one missing arg is the constexpr `BLOCK_SIZE_N`.**

`launch_pdl` is a **red herring** — it is injected on the Python side by
`CudaLauncher.__call__` (`driver.py:700–714`, `self.launch(..., self.launch_pdl, ...)`)
and is *always* supplied. The user-facing hypothesis ("31 vs 32 = new launch_pdl param")
was directionally right about "a new arg triton 3.6 wants" but wrong about *which* arg:
the deficit is a **constexpr placeholder slot**, caused by flag_gems' stale version gate.

Why call 0 works: it goes through the full JIT path
`kernel = self.fn.run(*args, **kwargs)` (`libentry.py:709`), which assembles the complete
argument list itself. Only the *cached* fast-path (call ≥ 1) uses the buggy `k_args`.

The "CUDA-context poisoning / hang" noted in `fg_ref.py` was a **downstream symptom**: a
partially-issued/failed launch left the context dirty. With the shim there is no failed
launch, so there is nothing to poison — flag_gems is fully co-resident.

---

## 2. The fix — 1-line monkeypatch in OUR code (site-packages stay read-only)

flag_gems' own 3.3–3.5 branch already contains a **correct, constexpr-inclusive** launch
path (`libentry.py:763–781`): it rebuilds the full ordered arg list from
`k_args ∪ tune_constexprs ∪ heur_constexprs ∪ constexprs` and calls
`kernel[grid[0:3]](*all_args)`. We just need 3.6 to take that branch. Both gates key off
the module global `minor_version`, so **one assignment flips both**:

```python
# apply BEFORE the first flag_gems op runs (idempotent)
import importlib
_le = importlib.import_module("flag_gems.utils.libentry")   # NB: attr `libentry` is the
                                                            # decorator fn, so import the
                                                            # module object explicitly
if _le.major_version == 3 and _le.minor_version >= 6:
    _le.minor_version = 5   # -> LibEntry.run uses its constexpr-inclusive launch branch
```

Safety of lying `minor_version = 5`:
- Only `flag_gems.utils.libentry`'s copy is mutated; every other flag_gems module computes
  its own `major/minor` from `triton.__version__`.
- The only other consumer in this module is `LibTuner.__init__` line 236
  `if major_version == 2 or (major_version == 3 and minor_version <= 1)` — stays False at
  5 (identical to 6). Line 41 is parse-only. No behavioural change beyond the intended
  two gates.

Recommended home: a tiny helper, e.g. `python/intentir/backend/flaggems_shim.py`
exposing `apply_flaggems_launch_shim()`, called immediately after `import flag_gems` in
both `python/tests/perf/bench_clusters.py` (the `_flag_gems()` helper) and
`python/tests/perf/fg_ref.py`.

---

## 3. Validation (all on RTX 5090D / sm_120a, triton 3.6.0, cuda 13.0)

| Test | Result |
|---|---|
| `fg.gather` ×100 in-process, correctness vs `torch.gather` | **100/100 OK, allclose** |
| 16-op fg_ref roster ×20 each, one process | **16/16 ops, 20/20 each, 0 failures** |
| autotuned ops (mm, softmax, layer_norm) ×30 | **OK** |
| `do_bench_cudagraph` in-process (gather/softmax/argmax) | **OK** (168 / 69 / 11 µs) |

`do_bench_cudagraph` succeeding in-process is the load-bearing result: flag_gems can now
be timed on the **same fair CUDA-graph steady-state path** as our cluster kernels and
torch — no subprocess, no `FG_BENCH` gate, no plain-`do_bench` downgrade.

---

## 4. flag_gems ops confirmed benchable as the REAL Triton target

All ops in the current `fg_ref.py` roster, verified running repeatedly in-process:

`gather, embedding, scatter_add, cat, pad(constant_pad_nd), tile, flip, triu, argmax,
sort, avg_pool2d, max_pool2d_with_indices, upsample_nearest2d, polar, conv2d,
masked_select`

Additionally verified: `add, mul, softmax, sum, layer_norm, mm`.

Notes:
- `nonzero` is referenced in `bench_clusters.py` C8 but is **not** in the fg_ref roster —
  confirm `fg.nonzero` exists / has a matching signature before relying on it (not
  exercised here).
- The shim is op-agnostic: it fixes the launcher dispatch itself, so any libentry-cached
  flag_gems triton kernel benefits.

---

## 5. Concrete integration steps for the perf task

1. Add `python/intentir/backend/flaggems_shim.py` with `apply_flaggems_launch_shim()`
   (the snippet in §2), idempotent and guarded on `major==3 and minor>=6`.
2. In `bench_clusters.py`: call it inside `_flag_gems()` right after `import flag_gems`;
   then time flag_gems **in-process** with the existing `bench()` /
   `do_bench_cudagraph` path (same as ours + torch). Delete the subprocess `fg_us()` /
   `FG_BENCH` gate and the stale "31 vs 32 = launch_pdl" comment (root cause was the
   constexpr slot, and it is now fixed).
3. Keep `fg_ref.py` only if a subprocess fallback is still wanted; if kept, apply the
   shim there too so it can use `do_bench_cudagraph` instead of plain `do_bench`.

### Evidence pointers (read-only site-packages)
- `flag_gems/utils/libentry.py:678,691,763,782-783` — the version gates + buggy fast-path.
- `flag_gems/utils/libentry.py:763-781` — the correct constexpr-inclusive path we route to.
- `triton/backends/nvidia/driver.py:123` — `_BASE_ARGS_FORMAT` (13 fixed, incl launch_pdl).
- `triton/backends/nvidia/driver.py:186-207` — `format_of` returns `"O"` for constexpr.
- `triton/backends/nvidia/driver.py:700-714` — `CudaLauncher.__call__` injects launch_pdl.
- `triton/compiler/compiler.py:490-501` — `CompiledKernel.__getitem__.runner`.
- `~/.flaggems/code_cache/gather_rank_2.py` — the concrete kernel (1 constexpr BLOCK_SIZE_N).
