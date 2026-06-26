# rms_norm head-to-head PK vs real liger -- RTX 5090 D (sm_120a), f32

Our MLIR-generated, **read-once** block-cooperative row kernel (`bench/rmsnorm_emit.py`)
PKs the REAL liger rms_norm forward kernel head-to-head, same op, same device,
`casting_mode="llama"`, `offset=0`, fp32. One `gpu.launch` block per row; `T = num_warps*32`
threads from liger's `calculate_settings` ladder; `K = next_pow2(cols)/T` columns/thread,
coalesced `tx + k*T`.

**Read-once is the whole game:** PASS 1 loads each `x` lane ONCE into a per-thread
`memref.alloca<Kxf32>` cache and reduces sum-of-squares via one `gpu.all_reduce add`
(fp32); PASS 2 writes `y` from the cache. HBM traffic = read x once + read w + write y
= parity with liger (a 2-global-pass version pays 1.5x traffic -> ~1.5x slower). `rstd`
uses the NVVM `rsqrt.approx.f` intrinsic (rel ~1.3e-7; `math.rsqrt` is a libdevice dead end).

## Primary metric: KERNEL-ONLY (the true codegen comparison)

Both sides preallocate `Y` (rows,cols) + `RSTD` (rows,) ONCE; the timed thunk only LAUNCHES
the kernel. liger is driven by calling its RAW Triton kernel `_rms_norm_forward_kernel`
directly with the preallocated outputs (same geometry as `calculate_settings`). This is the
right comparison: a per-call `torch.empty` of a ~128 MB `Y` churns the caching allocator and
drags BOTH sides down to the same level, masking the kernel-level picture (see the secondary
table). Timed with `triton.testing.do_bench` (per-window warmup + L2-flush + median) over
interleaved windows after a ~1s sustained clock warmup; per-side median, median over several
full PK runs. GB/s = `((x + y)*4 + w*4)` bytes / time.

Geometry: at both PK shapes ours and liger use the IDENTICAL block geometry (`calculate_settings`
gives num_warps=4 = 128 threads at cols=1024 and num_warps=8 = 256 threads at cols=4096; ours
mirrors the ladder), so there is **no thread-geometry edge to exploit** -- parity (~1.0x) is the
theoretically-expected result for one memory-bound op on identical geometry. These are sub-ms
kernels on a clock-throttling card, so the ratio is variance-dominated (run-to-run ~+/-8%).

| shape (rows,cols) | T | ours ms (GB/s) | liger ms (GB/s) | ours/liger | rel Y (ref / liger) | rel RSTD (ref / liger) |
|---|---|---|---|---|---|---|
| (32768,1024) | 128 | 0.785 (342) | 0.777 (345) | **1.032x** | 1.4e-07 / 2.1e-07 | 1.1e-07 / 1.1e-07 |
| (16384,4096) | 256 | 1.592 (337) | 1.591 (337) | **1.000x** | 1.6e-07 / 1.1e-07 | 1.1e-07 / 1.1e-07 |

## Secondary metric: end-to-end WITH allocation (conflates allocator overhead)

Each timed call allocates a fresh `Y`+`RSTD` (matching liger's launcher, which allocates
internally). This is NOT the headline: the per-call `torch.empty` churn dominates and drags
both sides to the same level, so it masks the kernel comparison. Kept for completeness only.

| shape (rows,cols) | T | ours ms (GB/s) | liger ms (GB/s) | ours/liger |
|---|---|---|---|---|
| (32768,1024) | 128 | 0.775 (347) | 0.755 (356) | **1.027x** |
| (16384,4096) | 256 | 1.594 (337) | 1.590 (338) | **1.003x** |

## Honest reading

- **Headline (kernel-only): PARITY.** rms_norm is a single memory-bound op and ours runs
  liger's own block geometry, so both move the same bytes and there is no fusion/traffic or
  geometry edge to exploit -- parity (~1.0x) is the ceiling, not a loss. At (32768,1024) the
  ratio is variance-dominated (swings ~0.97x-1.12x across runs on this throttling card); at
  (16384,4096) it is stable parity (~1.00x). Both sit in the same ~337-367 GB/s band. The
  result is robust to the metric choice: the secondary alloc-inclusive table lands at parity
  too (~1.03x / ~1.00x), so kernel-only was not picked to flatter. We do NOT claim a win; a
  beat must be explainable and never from a weakened baseline.
- **Scalar kernel retained; vectorization cannot help here.** The op is HBM-bandwidth-bound,
  not issue-bound (measured: clocks pinned at 2985 MHz / 187 W of 600 W, parity regardless of
  geometry), so wider loads relieve instruction issue -- which a bandwidth-bound kernel does
  not need. A separate probe (NOT reproduced in this deliverable) tried emitting `vector<4xf32>`
  `vector.load`/`vector.store` (Triton layout, `memref.assume_alignment ..., 16` inside the
  `gpu.launch` region): it reported correct lowering to `ld.global.v4.f32` / `st.global.v4.f32`
  but neutral-to-slightly-slower kernel-only time -- consistent with the bandwidth-bound
  conclusion above. The scalar kernel is the deliverable.
- **Correctness is the primary DoD bar and is met cleanly:** every output is bit-faithful to
  the fp32 reference AND to liger at rel ~1e-7 (Y and RSTD), including the non-power-of-2
  tail-mask path. The result is bandwidth parity, not a reduced-work shortcut.
- **Deviation note vs the task's recorded probe (retracted).** An earlier probe recorded ours
  at 0.77x / 461 GB/s at (32768,1024) and attributed it to liger's `num_warps=4` geometry being
  BW-suboptimal. That is FALSE and the win is retracted: ours and liger use the IDENTICAL
  128-thread geometry at this shape, so there is no mechanism for a geometry-driven win. With
  clocks pinned at steady boost (2985 MHz, no throttle) the 0.77x does not reproduce -- it was a
  transient high-clock window. We report the reproducible parity, never a faked peak.
