# Deferred mechanisms + perf study (can we beat CUDA-graph baselines?)

## Goal

Two follow-ons to the finished coverage-expansion task (07-02):
1. **Perf study (the sharp question)**: benchmark the ~10 new cluster kernels (C3–C8: gather/scatter, layout, RNG, arg-reduce, bitonic sort, pool/upsample/polar, conv, compact) vs the REAL flag_gems + torch under **CUDA-graph capture** (removes launch overhead → pure kernel-time PK). Answer honestly per cluster: **win / parity / lose**, and why (roofline: memory-bound → parity achievable + win via fusion; compute-bound → sm_120a hardware ceiling; naive-untuned → tuning gap).
2. **Deferred mechanisms**: build the coverage-task's classified-miss set — backward/grad ops, multi-op-graph compositions (the general op-graph dispatch: quantile/unique/topk_softmax/cross_entropy), native complex dtype, order-dependent RNG (multinomial/randperm), bicubic/aa interp, KV-cache/MoE/RWKV stateful compositions.

## What I already know

- Prior perf verdict (HONEST, prior task): memory-bound 15 kernels at parity/beat @ real ~1500 GB/s HBM; compute-bound 1.0–1.44× slower than cuBLAS = **sm_120a hardware ceiling** (Blackwell lacks Hopper warpgroup-MMA/TMA). Fusion wins (jsd 1.8× via not materializing the loss buffer).
- The NEW cluster kernels are **correctness-only** (one-block-per-row, no tail-mask, grid-stride, no bank-conflict/vectorization tuning) → likely BELOW tuned Triton until tuned. conv is naive windowed-accumulation (no tensor-core) → far off cuDNN.
- "CUDA graph PK" ≈ steady-state kernel-time comparison; do_bench already amortizes launch, CUDA-graph capture makes it explicit + fair.
- The general **op-graph dispatch** (multi-op fusion) is the keystone: unblocks the multi-op-graph deferred cluster AND is the fusion path that lets memory-bound ops WIN.

## Requirements

- [x] **Perf harness** (P1): `bench_clusters.py` — CUDA-graph capture (do_bench_cudagraph) for our cubin AND torch/flag_gems, 22 rows, realistic f32 shapes, ratio + roofline. Adversarially audited SOUND.
- [x] **Honest per-cluster verdict** (P1, REPORT.md): WIN gather 0.78 / pad 0.83 / dropout 0.88; PARITY embedding/concat/tile/flip/rand/upsample/polar/scatter_add; LOSE conv 10.2× (no tensor-core), sort 1.3× (bitonic vs radix), triu/argmax/pool ~2× (under-tiled), masked_select. flag_gems unbenchable for steady-state (triton-3.6 launcher ABI) → torch is the tuned ref.
- [x] **Tune** the worst loser (P2a): masked_select/nonzero single-block → 3-phase multi-block stream compaction, ~1400× → 2.5–3.1×. (The remaining ~2× memory-bound tunables — triu/argmax/pool, small absolute times, L2-residency-dependent — left documented, not rabbit-holed.)
- [x] **General op-graph dispatch** (P3): quantile / topk_softmax / cross_entropy_loss end-to-end (unique landed in P7). No new C++ — cluster passes compose over a multi-func module; `opgraph.py` threads device buffers across outlined stage cubins.
- [x] **Backward/grad ops** (P4): 14 backward ops (activation/softmax/norm dx via map+reduce, embedding/gather/nll via scatter-grad) vs torch autograd + flag_gems backward.
- [x] **Complex dtype** (P5): angle (libdevice-safe atan2 poly) / resolve_conj / resolve_neg / complex_add / complex_mul via real-imag planes.
- [x] **Remaining deferred** (P6/P7/P8): rotary_embedding / KV-cache scatter (flash+MLA+non-flash) / moe_sum / moe_align_block_size / apply_repetition_penalties / per_token_group_quant_fp8 / multinomial / randperm / bicubic upsample — all vs real flag_gems.
- [x] Every new op verified vs the real kernel; coverage.json/COVERAGE.md updated (verified-e2e 129→**162/231**); no per-kernel recognizer (trellis-check confirmed); Dataset/refs/key read-only (confirmed).

## Milestone plan (difficulty/leverage-ordered)

| P | work | why first |
|---|---|---|
| **P1** | Perf study: CUDA-graph PK harness + measure all clusters + honest verdict + tuning targets | answers "能获胜吗"; needs a quiet GPU (before build workflows) |
| **P2** | Tune the losing memory-bound kernels (tail-mask/vectorize/occupancy) toward parity | banks the perf wins the study identifies |
| **P3** | General op-graph dispatch (multi-op fusion) → quantile/unique/topk_softmax/cross_entropy | keystone; fusion is the memory-bound WIN path |
| **P4** | Backward/grad ops | biggest deferred class |
| **P5** | Complex dtype + multinomial/randperm + bicubic + KV-cache/MoE | the tail |

## Decision (ADR-lite)

**Context**: coverage-expansion shipped correctness for all static-shape forward mechanisms; the deferred set + perf remained. User (2026-07-06, ultracode): "延后的机制也做了吧,而且性能呢?和 cuda graph pk 能获胜吗?"
**Decision**: perf FIRST (sharp question, quiet-GPU-sensitive), then the deferred mechanisms in leverage order. Perf verdict must be HONEST (roofline-grounded, real baselines, CUDA-graph-fair) — winning where physics allows (memory-bound + fusion), losing where the sm_120a ceiling / tuning gap is real, and saying which.

## Out of Scope

- Beating cuDNN on conv via a full tensor-core im2col pipeline (report the gap; tensor-core conv is a large separate effort).
- Multi-GPU / non-GPU backends.

## Technical Notes

- Perf recipe + hardware ceiling: memory `tianchenir-backend-recipe`, `tianchenir-coverage-mechanisms`.
- Cluster passes: `lib/Intent/Conversion/ConvertIntent{Gather,Layout,Rng,ArgReduce,Sort,Pool,Upsample,Polar,Conv,Compact}ToGPU.cpp`. Coverage: `corpus/COVERAGE.md`.

## Outcome (2026-07-06 — DONE)

Delivered, all committed (`4f72aaf..9d4f155`), trellis-check PASS (0 blockers/majors).

**Perf — 和 CUDA graph PK 能获胜吗? Honest answer: yes where physics allows.** (`python/tests/perf/REPORT.md`, audited SOUND.) Under CUDA-graph capture vs torch (ATen/cuDNN/CUB — flag_gems is unbenchable for steady state on this triton-3.6/sm_120a stack): **WIN** on memory-bound movers (gather 0.78×, pad 0.83×, dropout 0.88× via fused mask), **PARITY** across embedding/concat/tile/flip/rand/upsample/polar/scatter_add (~1.0×). **LOSE** where the codegen is naive or the op is compute-bound: conv2d 10× (no tensor-core vs cuDNN), sort 1.3× (bitonic vs CUB radix), triu/argmax/pool ~2× (under-tiled, small absolute times). P2a fixed the one catastrophe — masked_select/nonzero's single-block compaction **~1400× → 2.5–3.1×** via 3-phase multi-block parallel stream compaction. Absolute GB/s labels are indicative only; the our/torch ratios are the audited truth.

**Deferred mechanisms — all tractable ones done** (P3–P8), each verified vs torch AND real flag_gems, composed from existing op-type-matched passes (no new recognizer): op-graph dispatch (quantile/topk_softmax/cross_entropy), 14 backward ops, complex dtype (angle via libdevice-safe atan2), rotary/KV-cache/moe_sum/repetition, unique/randperm/multinomial/bicubic, fp8-group-quant/moe_align/reshape_and_cache. **Coverage verified-e2e 129 → 162 / 231.**

**Residual classified-miss (14, genuinely irreducible):** flash/attention (loop-kernel authorship — PRD out-of-scope; the actual flash fwd/bwd kernels shipped in the 29-kernel task), rwkv ×2 (recurrent carried state — needs a sequential-scan-with-state mechanism), DSA ×3 (experimental sparse sub-package, not top-level ops), + Triton bicubic-aa-downsample / cross-correlation. Each named in `corpus/coverage.json`.

**Not pursued (diminishing returns):** the ~2× memory-bound tunables (triu/argmax/pool) — small absolute times (10–27 µs), losses partly from torch's L2 residency on these shapes; documented, not rabbit-holed. Tensor-core conv (im2col+wmma) — a large separate effort, explicitly out of scope.
