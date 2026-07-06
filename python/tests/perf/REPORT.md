# M0 perf study — 3-way: ours vs torch vs **flag_gems** (the real Triton target)

**Question:** under CUDA graph (launch overhead removed for every side), can our generated cluster kernels beat the real flag_gems / torch kernels?

**M0 change (this task).** flag_gems is now benched **in-process**, on the SAME `triton.testing.do_bench_cudagraph` steady-state path as ours + torch. The triton-3.6 launcher bug that made flag_gems unbenchable is fixed by a runtime monkeypatch shim in our code — `intentir.backend.flaggems_shim` (`apply_flaggems_launch_shim()`): flag_gems' cached `libentry` launcher has a version gate (`major==3 and 3<=minor<=5`) that on triton 3.6 falls through and DROPS every constexpr placeholder from the launch arg list, so the launcher got 31 args where 32 were wanted ("function takes exactly 32 arguments (31 given)") on every call after the first. The shim sets `flag_gems.utils.libentry.minor_version = 5`, routing triton 3.6 through flag_gems' own constexpr-inclusive launch branch. Only the imported module object is mutated; site-packages stay read-only. Validated: 120 in-process calls + `do_bench_cudagraph` on gather/softmax/argmax with correct output. The old `FG_BENCH` subprocess escape hatch (`fg_ref.py`) is deleted.

**Method (audited SOUND).** `bench_clusters.py` times our lowered sm_120a cubin, `torch.<op>` (ATen/cuDNN/CUB), and `flag_gems.<op>` — all under `do_bench_cudagraph`. Per-call host launch overhead → ~0 for every contender; correctness is asserted at the benched shape for **all three** (nothing is a no-op). Ops with dynamic output extent / in-place mutation (scatter_add, masked_select, nonzero) fall back to plain `do_bench`, tagged. f32 all sides. **ratio = our_us / faster-of(torch, flag_gems)** — the honest "beat the better library" number. GB/s / %HBM are indicative vanity (1500 denominator + logical-byte overcount); the ratio is the load-bearing metric.

**flag_gems dtype note:** flag_gems follows torch's default dtype for these ops, so at these f32 shapes it moves the SAME bytes as ours/torch — no free fp16/bf16 halving. A real fp16/bf16 deployment (norm/softmax/attention epilogues) would move ~2× fewer bytes; that lever is not exercised here. flag_gems `dropout` returns `(out, mask)`; sort/argmax/pool match torch semantics.

## Result (3-way; ratio = our_us / faster reference; <1 = we win)

| cluster | op | shape | our µs | torch µs | **fg µs** | best-ref | ratio | verdict |
|---|---|---|--:|--:|--:|:--|--:|---|
| C3 | gather ax1 | [8192,4096] | 343 | 477 | **339** | flag_gems | 1.01 | ⚪ parity |
| C3 | embedding | [8192,4096] idx8192 | 156 | 147 | 156 | torch | 1.06 | ⚪ parity |
| C3 | scatter_add ax1 | [8192,4096] k4096 | 737 | 712 | 872 | torch | 1.04 | ⚪ parity |
| C4 | concat ax1 | [4096,4096] | 180 | 178 | **177** | flag_gems | 1.02 | ⚪ parity |
| C4 | pad | [4092²]→[4096²] | 79 | 95 | — | torch | **0.83** | 🟢 WIN |
| C4 | tile [2,2] | [4096,4096] | 274 | 278 | **258** | flag_gems | 1.06 | ⚪ parity |
| C4 | flip ax1 | [4096,4096] | 76 | 72 | **70** | flag_gems | 1.09 | ⚪ parity |
| C4 | triu k0 | [4096,4096] | 75 | 39 | 70 | torch | 1.95 | 🔴 lose (tunable) |
| C5 | rand | [8192,8192] | 165 | 168 | — | torch | 0.98 | ⚪ parity |
| C5 | dropout p=.25 | [8192,8192] | 358 | 408 | 401 | flag_gems | **0.89** | 🟢 WIN (fused mask) |
| C6 | argmax ax1 | [4096,4096] | 27 | 12 | **11** | flag_gems | 2.45 | 🔴 lose (tunable) |
| C6 | sort ax1 | [4096,2048] | 299 | **231** | 2586 | torch | 1.29 | 🔴 lose (bitonic vs radix) |
| C6 | sort ax1 | [4096,4096] | 708 | **519** | 4976 | torch | 1.37 | 🔴 lose (bitonic vs radix) |
| C7 | avg_pool2d k2s2 | [32,64,56,56] | 27 | 14 | **9** | flag_gems | 2.90 | 🔴 LOSE (tunable) |
| C7 | max_pool2d+idx | [32,64,56,56] | 27 | **10** | 18 | torch | 2.70 | 🔴 LOSE (tunable) |
| C7 | upsample_near ×2 | [32,64,56,56] | 108 | 93 | **91** | flag_gems | 1.18 | 🔴 lose |
| C7 | conv2d k3s1p1 | [32,64,56,56]→64 | 1124 | **111** | 276 | torch | 10.14 | 🔴 LOSE (no tensor-core) |
| C7 | polar | [4096,4096] | 180 | 176 | **176** | flag_gems | 1.02 | ⚪ parity |
| C8 | masked_select | [2048,2048] ~50% | 41 | 71 | **59** | flag_gems | 0.69 | 🟢 WIN (block-coop M2) |
| C8 | masked_select | [4096,4096] ~50% | 145 | 325 | **126** | flag_gems | 1.15 | 🟡 ≤1.3× (block-coop M2) |
| C8 | nonzero | [2048,2048] ~50% | 43 | **60** | 111 | torch | 0.72 | 🟢 WIN (block-coop M2) |
| C8 | nonzero | [4096,4096] ~50% | 188 | **273** | 551 | torch | 0.69 | 🟢 WIN (block-coop M2) |

best-ref split: flag_gems is the faster reference on 10/22 rows, torch on 12/22.

## What changed vs the torch-only report

Adding flag_gems as a real contender **tightens several gates** — the ratio is now vs the *faster* of the two libraries, not torch alone:
- **avg_pool2d 1.96×→2.90×**: flag_gems' Triton pool (9 µs) is faster than torch (14 µs), so the true gap is larger than the old torch-only number showed.
- **argmax 2.25×→2.45×**; **flip/tile/gather/concat/polar** best-ref flips to flag_gems (all still parity).
- **masked_select [4096²] 3.1×→7.8×**: flag_gems compaction (129 µs) is ~2.5× faster than torch (325 µs) — our single-thread-per-block chunk read is the real loser, and flag_gems is the bar to beat, not torch.
- Where flag_gems is a naive/immature kernel it **loses to torch** and does NOT tighten the gate: **sort** (flag_gems 2586/4976 µs ≈10× slower than torch's CUB radix — torch stays the target), **scatter_add** (flag_gems 872 vs torch 712 µs).

## Honest gaps vs flag_gems (may differ from the torch gaps)

| op | vs torch | vs flag_gems | real target |
|---|--:|--:|---|
| avg_pool2d | 1.96× | **2.90×** | flag_gems (Triton 9 µs) — tighter |
| masked_select[4096²] | 3.1× → **1.16×** (M2) | 7.8× → **1.16×** (M2) | flag_gems (compaction 125 µs); block-coop coalesced closed it |
| argmax | 2.25× | **2.45×** | flag_gems (11 µs) |
| upsample | 1.14× | **1.18×** | flag_gems (91 µs) |
| conv2d | **10.1×** | 4.1× | torch/cuDNN (flag_gems conv itself 2.5× off cuDNN) |
| max_pool2d+idx | **2.70×** | 1.55× | torch |
| nonzero | **2.24–2.57×** | ~1.3× | torch (flag_gems nonzero slower) |
| sort | **1.29–1.37×** | 0.11–0.14× | torch (flag_gems sort ~10× slower) |

**Parity-or-win holds on the memory-bound movers** even against the faster library: pad 0.83× and dropout 0.89× are genuine wins; gather, embedding, scatter_add, concat, tile, flip, rand, polar sit at parity (≤1.09×) — **10 of 22 rows at parity-or-win**, most near the GDDR7 spec. The naive-codegen losses (conv 10×, compaction 2.2–7.8×, pool/argmax/triu 1.95–2.90×, sort 1.3×) are now measured against the real Triton target and feed the M1–M7 milestones.

**Gate 0 (4.1) MET:** flag_gems runs ≥100× in-process under `do_bench_cudagraph` with the shim; the 3-way table is produced with correctness asserted per contender; `FG_BENCH`/subprocess path removed; results in `bench_clusters_results.json`.

## Downstream targets (unchanged milestone order)
1. **M1** pool/argmax/triu: flat grid-stride + vector<4> + tail-mask → parity.
2. **M2** compaction: block-cooperative coalesced mover → **DONE**. Rewrote `ConvertIntentCompactToGPU.cpp` from block=1-thread to block=T-threads (T=256, coalesced flag loads + block Hillis-Steele scan of the 0/1 predicate + cooperative scatter — the CUB `DeviceSelect::Flagged` pattern; exact row-major preserved). HBM 6–12% → 54–62%. masked_select [2048²] 3.12→**0.69×**, [4096²] 7.99→**1.16×**; nonzero [2048²] 2.30→**0.72×**, [4096²] 2.59→**0.69×**. 3/4 now beat flag_gems outright; masked_select[4096²] within the ≤1.3× target. All exact (torch.equal).
3. **M3** conv2d: implicit-GEMM on mma.sync + TF32/fp16 → ≤2× cuDNN.
4. **M7** sort: W-aware top-k (torch/CUB stays the target).
