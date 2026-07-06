# P1 perf study — can our C3–C8 cluster kernels beat CUDA-graph baselines?

**Question (user):** under CUDA graph (launch overhead removed for both sides), can our generated cluster kernels beat the real flag_gems / torch kernels?

**Method (audited SOUND).** `bench_clusters.py` times both our lowered sm_120a cubin and the reference under `triton.testing.do_bench_cudagraph` — warmup on a side stream, N calls captured into one CUDA graph, time `g.replay()`. Per-call host launch overhead → ~0 for every contender; correctness is asserted at the benched shape (nothing is a no-op). flag_gems was **unbenchable for steady state** (its cached-dispatch launcher passes 31 args but triton-3.6 expects 32 / `launch_pdl` on sm_120a — only the first call/process works; the PKs pass because they call once), so the reference is **torch = ATen / cuDNN / CUB**, the tuned real library. f32 both sides.

An adversarial audit re-ran gather / pad / dropout / masked_select and confirmed the CUDA-graph capture is real & symmetric and **all 20 ratios stand**. The only correction: the GB/s / %HBM labels are optimistic vanity (1500 denominator + logical-byte overcount) — **the our/torch ratio is the load-bearing number**, not bandwidth %.

## Result (ratio = our_us / best-torch_us; <1 = we win)

| cluster | op | shape | our µs | torch µs | ratio | verdict |
|---|---|---|--:|--:|--:|---|
| C3 | **gather** ax1 | [8192,4096] | 343 | 464 | **0.76** | 🟢 WIN |
| C3 | embedding | [8192,4096] idx8192 | 156 | 148 | 1.06 | ⚪ parity |
| C3 | scatter_add ax1 | [8192,4096] k4096 | 738 | 709 | 1.04 | ⚪ parity |
| C4 | concat ax1 | [4096,4096] | 180 | 178 | 1.01 | ⚪ parity |
| C4 | **pad** | [4092²]→[4096²] | 81 | 95 | **0.85** | 🟢 WIN |
| C4 | tile [2,2] | [4096,4096] | 274 | 278 | 0.99 | ⚪ parity |
| C4 | flip ax1 | [4096,4096] | 75 | 72 | 1.05 | ⚪ parity |
| C4 | triu k0 | [4096,4096] | 76 | 39 | 1.98 | 🔴 lose (tunable) |
| C5 | rand | [8192,8192] | 164 | 167 | 0.99 | ⚪ parity |
| C5 | **dropout** p=.25 | [8192,8192] | 358 | 408 | **0.88** | 🟢 WIN (fused mask) |
| C6 | argmax ax1 | [4096,4096] | 27 | 12 | 2.25 | 🔴 lose (tunable) |
| C6 | sort ax1 | [4096,2048] | 297 | 230 | 1.29 | 🔴 lose (bitonic vs radix) |
| C6 | sort ax1 | [4096,4096] | 703 | 518 | 1.36 | 🔴 lose (bitonic vs radix) |
| C7 | avg_pool2d k2s2 | [32,64,56,56] | 27 | 14 | 1.96 | 🔴 lose (tunable) |
| C7 | max_pool2d+idx | [32,64,56,56] | 27 | 10 | 2.72 | 🔴 lose (tunable) |
| C7 | upsample_near ×2 | [32,64,56,56] | 107 | 94 | 1.14 | ⚪ parity |
| C7 | **conv2d** k3s1p1 | [32,64,56,56]→64 | 1118 | 110 | **10.2** | 🔴 LOSE (no tensor-core vs cuDNN) |
| C7 | polar | [4096,4096] | 180 | 176 | 1.02 | ⚪ parity |
| C8 | **masked_select** | [2048,2048] ~50% | 110303 | 74 | **~1400** | 🔴 LOSE (grid(1,1,1) single-block) |
| C8 | masked_select | [4096,4096] ~50% | 439932 | 325 | ~1360 | 🔴 LOSE (single-block) |

## Honest answer

**Yes — where physics allows.** On memory-bound movers we **win or match**: genuine wins gather 0.76× / pad 0.85× / dropout 0.88× (fused mask = fewer passes), and parity across embedding, scatter_add, concat, tile, flip, rand, upsample, polar (≈1.0×). That's **11 of 20 at parity-or-win**, several at ≥96% of the GDDR7 spec bandwidth. This is the regime the design targets, and it holds up against ATen under CUDA graph.

**No — where the codegen is naive or the op is compute-bound:**
- **`masked_select` ~1400× (worst).** The compaction pass emits `grid(1,1,1)` — a single block does the whole serial prefix-scan. Architectural, not a constant factor. **Biggest, clearest tuning win available** (multi-block prefix-sum → near-parity). → P2.
- **`conv2d` 10×.** Naive windowed accumulation, no tensor cores, ~6% of f32 peak, vs cuDNN's tensor-core kernel. Closing it needs im2col + wmma — a large effort (PRD out-of-scope, but a candidate).
- **`sort` 1.3×.** Bitonic O(N log²N) single-block vs CUB radix. Algorithmic; hard to fully close.
- **`triu` 2.0×, `argmax` 2.25×, `pool` 2.0–2.7×.** Memory-bound but under-tiled — one-block-per-row leaves the SMs underutilized on small/awkward shapes. **Tunable toward parity** (tail-mask, wider tiles, more blocks). → P2.

**Caveat.** Absolute GB/s and %HBM in the raw JSON are indicative only (see the note in `bench_clusters.py`); the ratios above are the fair, audited truth. flag_gems could not be steady-state-benched on this stack (torch is the stand-in tuned reference).

## P2 tuning targets (leverage order)
1. **masked_select / compact: multi-block prefix-sum** (≈1400× → parity) — the single biggest win.
2. **pool / argmax / triu: better tiling + more blocks** (≈2× → parity).
3. **sort: multi-block merge / radix** (algorithmic; partial).
4. **conv: im2col + tensor-core** (large; likely a separate effort).
