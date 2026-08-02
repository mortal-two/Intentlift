# vs-Triton performance (measured on RTX 5090 D)

## FlashAttention latency (RTX 5090 D, single head, f32)

IntentIR = the kernel we compile from intent + launch (cached plan).

| shape (Sq×Sk×D) | IntentIR ms | Triton ms | torch SDPA ms | IntentIR TFLOP/s | Triton TFLOP/s | IntentIR/Triton |
|---|---|---|---|---|---|---|
| 128x128x64 | 7.1643 | 0.0082 | 0.0144 | 0.0006 | 0.512 | 875× |
| 256x256x64 | 8.1896 | 0.0144 | 0.0267 | 0.0020 | 1.165 | 569× |
| 512x512x64 | 34.2116 | 0.0348 | 0.1839 | 0.0020 | 1.927 | 983× |

## Corpus kernels: Triton vs torch native (4096×4096, f32)

| kernel | Triton ms | torch ms |
|---|---|---|
| softmax | 0.2041 | 0.2079 |
| silu | 0.2056 | 0.2046 |

## Honest reading

- IntentIR's fused kernel is correctness-first (one thread per query, scalar online-softmax carries), so it trails Triton by the factor above. What closes it: a warp-tiled schedule (shared-memory K/V tiles, register-blocked accumulators) emitted under the SAME Layer-A intent — a Layer-C retune, not a semantic change.
- Correctness parity IS established: IntentIR == Triton == torch to rel ~1e-7 (see corpus + flash tests). This table is the latency half.
