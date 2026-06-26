# VERIFIED rms_norm MLIR design + PK numbers (probe, 2026-06-26, 5090D)

This is the **proven** hand-emitted MLIR template the productionized emitter must
reproduce. Every claim below was run end-to-end on the 5090D (sm_120a, system
MLIR 20.1.8, ptxas 12.8). Probe: `scratchpad/rmsnorm_probe.py` (session-local;
this file is the durable record).

## The verified MLIR (read-once, block-cooperative, parameterized R,C,T; K=C/T)

```mlir
module {
  func.func @rmsnorm(%X: memref<RxCxf32>, %W: memref<Cxf32>,
                     %Y: memref<RxCxf32>, %RSTD: memref<Rxf32>) {
    %c1 = arith.constant 1 : index
    %R  = arith.constant R : index
    %Tg = arith.constant T : index
    gpu.launch blocks(%bx,%by,%bz) in (%gx=%R,%gy=%c1,%gz=%c1)
               threads(%tx,%ty,%tz) in (%bsx=%Tg,%bsy=%c1,%bsz=%c1) {
      %c0=arith.constant 0:index   %ci1=arith.constant 1:index
      %K=arith.constant K:index    %T=arith.constant T:index
      %z=arith.constant 0.0:f32    %eps=arith.constant 0.000001:f32
      %n=arith.constant <float(C)>:f32
      %cache = memref.alloca() : memref<Kxf32>            // private per-thread row cache
      // PASS 1: read x ONCE into cache, accumulate sum-of-squares (fp32)
      %ss = scf.for %k=%c0 to %K step %ci1 iter_args(%acc=%z)->(f32){
        %kt=arith.muli %k,%T:index    %col=arith.addi %tx,%kt:index  // coalesced: tx + k*T
        %v=memref.load %X[%bx,%col]:memref<RxCxf32>
        memref.store %v,%cache[%k]:memref<Kxf32>
        %sq=arith.mulf %v,%v:f32      %a=arith.addf %acc,%sq:f32      scf.yield %a:f32
      }
      %total = gpu.all_reduce add %ss uniform {} : (f32)->(f32)        // block-cooperative
      %mean=arith.divf %total,%n:f32  %meps=arith.addf %mean,%eps:f32
      %rstd = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%meps):(f32)->f32
      %is0=arith.cmpi eq,%tx,%c0:index
      scf.if %is0 { memref.store %rstd,%RSTD[%bx]:memref<Rxf32> }      // match liger RSTD output
      // PASS 2: write y from CACHE (no second global load of x)
      scf.for %k=%c0 to %K step %ci1 {
        %kt=arith.muli %k,%T:index    %col=arith.addi %tx,%kt:index
        %xv=memref.load %cache[%k]:memref<Kxf32>
        %xn=arith.mulf %xv,%rstd:f32
        %wv=memref.load %W[%col]:memref<Cxf32>
        %yv=arith.mulf %xn,%wv:f32                                     // offset=0 (llama)
        memref.store %yv,%Y[%bx,%col]:memref<RxCxf32>
      }
      gpu.terminator
    }
    return
  }
}
```

Lower via `compile.lower_memref_module_to_kernels(mlir)`; launch via
`backend/launch.py` with `(R,1,1)` grid, `(T,1,1)` block.

## CRITICAL gotchas (cost real debug time — encode in the emitter)

1. **`math.sqrt` / `math.rsqrt` are DEAD ENDS** — they lower to libdevice
   `__nv_sqrtf` / `__nv_rsqrtf` externs → `ptxas fatal: Unresolved extern`.
   Use the NVVM intrinsic directly: `llvm.call_intrinsic
   "llvm.nvvm.rsqrt.approx.f"(%x) : (f32)->f32`. Despite "approx", measured rel
   vs fp32-ref AND vs liger is **1.3e-7** (rstd is one value/row, error negligible).
   This is the same class of fix as the exp→ex2.approx peephole.
2. **Outliner capture order is FIRST-USE order, NOT func-arg order.** For this
   kernel the captured order is **`[X, RSTD, W, Y]`** (X used in pass1, RSTD in
   the scf.if, W+Y in pass2). The launch arg list MUST match. Productionized code
   must NOT hardcode — parse the `gpu.launch_func ... args(%argN ...)` order from
   the lowered module (the func block-arg names %X/%W/%Y/%RSTD appear in the
   pre-pipeline `front` MLIR), or arrange first-use to a known order.
3. **Read-once is the whole game.** A two-global-pass version (reload x for the
   write) pays read-x-twice + write-y = 1.5× traffic → ~1.5× SLOWER than liger.
   The `memref.alloca<Kxf32>` per-thread cache (K = C/T small, e.g. 4–16) keeps
   HBM traffic at read-x-once + read-w + write-y = parity with liger.
4. **Static shapes only** (bare-ptr calling convention) — bake R,C,T,K into the
   emitted text per config, like `bench/flash_emit.py`.

## Schedule (mirror liger calculate_settings — Layer-C hint)

threads T = num_warps*32, num_warps from next_pow2(C): C<2048→4 (128), 2048–8191→8
(256), 8192–32767→16 (512), ≥32768→32 (1024). K = next_pow2(C)/T. For non-pow2 C,
BLOCK_SIZE=next_pow2(C) with a tail mask (col<C) on load(other=0)/store; the two
PK shapes are pow2 so mask is a no-op.

## PK numbers — CORRECTED (the initial 0.77× was a clock artifact; see below)

**Primary metric = KERNEL-ONLY** (both sides preallocate Y+RSTD once; the timed
thunk only launches the kernel; ~1s sustained-clock warmup; interleaved
`triton.testing.do_bench` windows; per-side median). This is the codegen-quality
comparison the user's bar ("后端生成的 mlir 足够高性能") actually asks for.

| shape (rows,cols) | T | ours/liger (kernel-only) | verdict | rel vs liger |
|---|---|---|---|---|
| (32768,1024) | 128 | ~0.97–1.12× across runs (variance-bound) | PARITY | Y/RSTD ~1e-7 |
| (16384,4096) | 256 | ~1.00× (stable) | PARITY | Y/RSTD ~1e-7 |

Both sit in the same ~337–367 GB/s band (this throttled "D" card; sub-ms kernels
⇒ ±~8% run-to-run). **PARITY is the honest, expected result** for a memory-bound
op on identical 128-thread geometry — and parity IS the backend win.

### Retraction (honesty)
An earlier single-shot probe recorded ours at **0.771× / 461 GB/s** at (32768,1024)
with a "liger num_warps=4 BW-suboptimal" explanation. **This did NOT reproduce**
and is RETRACTED:
- Kernel-only, clocks verified pinned (2985/3150 MHz, no throttle), ours is at
  parity (~0.97–1.12×), not 0.77×. The 461 GB/s was a transient high-clock window.
- Ours and liger use the **identical 128-thread geometry** at this shape
  (`calculate_settings(1024) = (1024, 4)` → 128 threads, which ours mirrors), so
  there is **no geometry edge to exploit** — the win narrative was unfounded.
- An alloc-inclusive harness (fresh `torch.empty` Y+RSTD per timed call) shows
  apparent "parity" too, but it flatters us by dragging liger down to our level
  via allocator churn — that's why kernel-only is the primary metric.

### Vectorization (v4) — tried, lowers, no gain
Emitting `vector.load`/`vector.store` as `vector<4xf32>` with
`memref.assume_alignment %X, 16` placed INSIDE the gpu.launch region (Triton
layout: thread t, group g loads contiguous `[g·4T+4t, +4)`) **does lower to
`ld.global.v4.f32` / `st.global.v4.f32`** (verified: 4 v4 loads + 2 v4 stores, 0
scalar global loads). But kernel-only it is **neutral-to-slightly-slower** than
the scalar kernel — the kernel is **HBM-bandwidth-bound, not issue-bound**, so
relieving instruction pressure buys nothing. **Scalar kernel retained.**
(Note: v4 needs `assume_alignment` 16B inside the region + K%4==0; the bare
`vector.load` without the in-region assume_alignment scalarizes.)

### From-intent gap (the real "backend generates" requirement — follow-on)
The current `-convert-intent-to-linalg` on the rms_norm intent emits **8 separate
`linalg.generic` → ~8 naive GPU kernels** (one HBM round-trip each) — NOT this
single high-perf kernel. The hand-emit above is the *target spec*, not "the
backend". Bridging intent → this kernel needs a row-reduction codegen path
(recognize elementwise → reduce-over-last-axis → broadcast → elementwise, emit
one block-cooperative read-once kernel). Tracked as a follow-on task; this
generalizes to softmax (max+sum) and layer_norm (mean+var).
