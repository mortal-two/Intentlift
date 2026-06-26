# softmax head-to-head PK vs real liger -- RTX 5090 D (sm_120a), f32

Our MLIR-generated, **read-once** block-cooperative row kernel (`bench/softmax_emit.py`)
PKs the REAL liger softmax forward kernel (`_softmax_single_block_forward_kernel`)
head-to-head, same op, same device, fp32. One `gpu.launch` block per row; `T = num_warps*32`
threads from liger's `calculate_settings` ladder; `K = next_pow2(cols)/T` columns/thread,
coalesced `tx + k*T`.

**This GENERALIZES the row-reduction codegen** from ONE reduction (rms_norm: mean) to TWO
(softmax: max then sum): PASS 1 loads each `x` lane ONCE into a per-thread `memref.alloca<Kxf32>`
cache and reduces the row max via `gpu.all_reduce maxnumf` (the verified sm_120a-lowering analog
of `add` -- MLIR-20 has no `max` op kind; `maxnumf` is the f32 max); PASS 1b computes `e = exp(cache - m)`
(`math.exp` -> `ex2.approx` peephole), reduces the denominator via `gpu.all_reduce add`, and
writes `e` back into the cache; PASS 2 writes `y = cache / s` from the cache. ONE global X load,
ONE global Y store -- no second global reload (faithful to liger's live single-block kernel).

## Primary metric: KERNEL-ONLY (the true codegen comparison)

Both sides preallocate `Y` (rows,cols) ONCE; the timed thunk only LAUNCHES the kernel. liger is
driven by calling its RAW Triton kernel `_softmax_single_block_forward_kernel` directly with the
preallocated output (same geometry as `calculate_settings`). A per-call `torch.empty` of a large
`Y` churns the caching allocator and drags BOTH sides down to the same level, masking the
kernel-level picture (see the secondary table). Timed with `triton.testing.do_bench` (per-window
warmup + L2-flush + median) over interleaved windows after a ~1s sustained clock warmup; per-side
median, median over several full PK runs. GB/s = `(x + y)*4` bytes / time.

Geometry: at both PK shapes ours and liger use the IDENTICAL block geometry (`calculate_settings`
gives num_warps=4 = 128 threads at cols=1024 and num_warps=8 = 256 threads at cols=4096; ours
mirrors the ladder), so there is **no thread-geometry edge to exploit** -- parity (~1.0x) is the
theoretically-expected result for one memory-bound op on identical geometry. These are sub-ms
kernels on a clock-throttling card, so the ratio is variance-dominated.

| shape (rows,cols) | T | ours ms (GB/s) | liger ms (GB/s) | ours/liger | rel Y (ref / liger) | row-sum err |
|---|---|---|---|---|---|---|
| (32768,1024) | 128 | 0.787 (341) | 0.765 (351) | **1.015x** | 1.1e-07 / 1.1e-07 | 2.4e-07 |
| (16384,4096) | 256 | 1.966 (273) | 1.966 (273) | **1.000x** | 9.6e-08 / 9.6e-08 | 2.4e-07 |

## Secondary metric: end-to-end WITH allocation (conflates allocator overhead)

Each timed call allocates a fresh `Y` (matching liger's `_softmax_forward` launcher, which
allocates internally). This is NOT the headline: the per-call `torch.empty` churn dominates and
drags both sides to the same level. Kept for completeness only.

| shape (rows,cols) | T | ours ms (GB/s) | liger ms (GB/s) | ours/liger |
|---|---|---|---|---|
| (32768,1024) | 128 | 0.778 (345) | 0.785 (342) | **0.992x** |
| (16384,4096) | 256 | 1.844 (291) | 1.965 (273) | **1.001x** |

## Honest reading

- **Headline (kernel-only): PARITY.** softmax is a single memory-bound op (two reductions, but
  still read-x-once / write-y-once over HBM) and ours runs liger's own block geometry, so both
  move the same bytes and there is no fusion/traffic or geometry edge to exploit -- parity (~1.0x)
  is the ceiling, not a loss. Both sides sit in the same GB/s band. We do NOT claim a win; a beat
  must be explainable and never from a weakened baseline.
- **The codegen GENERALIZES.** The same read-once row-kernel skeleton that matched liger on
  rms_norm (one mean reduction) now matches liger on softmax (max + sum). The only deltas are the
  reduction op kinds (`maxnumf` then `add` vs a single `add`), the `-inf`-init / `-inf`-tail-fill
  for the max, and the in-cache `exp` rewrite -- the block layout, read-once cache, generic launch
  resolver, and PK harness are shared. This is the DoD's "general enough that softmax reuses the
  codegen" item, proven empirically.
- **Correctness is the primary DoD bar and is met cleanly:** every output is bit-faithful to the
  fp32 reference AND to liger at rel ~1e-6/1e-7, every row sums to ~1.0, INCLUDING the
  non-power-of-2 tail-mask path (the `-inf` load -> never wins max -> `exp`->0 -> 0 to sum ->
  masked store chain). The result is bandwidth parity, not a reduced-work shortcut.
