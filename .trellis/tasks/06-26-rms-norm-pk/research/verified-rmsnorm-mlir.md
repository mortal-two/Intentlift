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

## VERIFIED PK numbers (fair: both alloc fresh Y+RSTD each call; ours writes RSTD)

| shape (rows,cols) | T | ours ms (GB/s) | liger ms (GB/s) | ours/liger | rel vs liger |
|---|---|---|---|---|---|
| (32768,1024) | 128 | 0.582 (461) | 0.755 (356) | **0.771× (win 1.30×)** | Y 1.3e-7, RSTD 1.6e-7 |
| (16384,4096) | 256 | 1.471 (365) | 1.468 (366) | **1.002× (parity)** | Y 1.6e-7, RSTD 1.1e-7 |

Timed with `triton.testing.do_bench` (warmup + L2 flush + median) for BOTH sides.
GB/s = (x + y read/write + w) / time. The win at (32768,1024) is HBM-saturation
(461 GB/s ≈ this card's real copy ceiling; liger's num_warps=4 geometry is
BW-suboptimal there on sm_120), NOT a weakened baseline — it's the real liger
kernel, same op, same outputs (Y+RSTD certified equal).
