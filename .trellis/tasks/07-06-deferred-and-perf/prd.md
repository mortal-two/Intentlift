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

- [ ] **Perf harness**: CUDA-graph capture + do_bench for each cluster kernel vs flag_gems/torch at realistic shapes; report our-time / ref-time / ratio + roofline classification + GB/s vs the ~1500 peak.
- [ ] **Honest per-cluster verdict**: win/parity/lose with the reason; identify tuning targets (tail-mask, vectorized ld/st, occupancy, tensor-core for conv).
- [ ] **Tune** the losing memory-bound kernels toward parity where feasible; report residual gaps honestly.
- [ ] **General op-graph dispatch** (multi-op fusion): quantile, unique, topk_softmax, cross_entropy_loss end-to-end.
- [ ] **Backward/grad ops**: the _backward cluster (conv/pool/softmax/norm/… gradients).
- [ ] **Native complex dtype**: angle/resolve_conj/resolve_neg + complex arithmetic.
- [ ] **Remaining deferred**: multinomial/randperm, bicubic/aa interp, KV-cache/MoE/RWKV compositions.
- [ ] Every new op verified vs the real kernel; coverage.json/COVERAGE.md updated; no per-kernel recognizer; Dataset/refs/key read-only.

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
