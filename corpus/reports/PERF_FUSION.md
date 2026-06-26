# Beat-Triton-via-fusion (elementwise front) — RTX 5090 D, f32

IntentIR recovers the *algebra* of a pointwise chain and emits the whole
chain as ONE fused GPU kernel (`-linalg-fuse-elementwise-ops` + a tiled
256-thread schedule, generated automatically from the lifted intent). A
high-performance library ships the chain as >=2 Triton kernels, each paying a
full HBM round-trip + a launch. The fused kernel makes 4 HBM passes (read
a,b,c + write z) / 1 launch vs the library's 6 passes / 2 launches — a 1.5x
traffic ceiling on a memory-bound chain.

Method: clean inline Triton baselines (strongest+fairest Triton, and they
launch under Triton 3.6 where flag_gems' pointwise_dynamic wrapper does not);
every kernel warmed; median of 5 windows; ONE sync per window (never
per-iter); working sets > the 101 MB L2 so timing is genuinely HBM-bound.

| chain | shape | IIR us | Triton unfused us | Triton fused us | **x vs unfused** | x vs hand-fused | IIR GB/s | cert |
|---|---|---|---|---|---|---|---|---|
| add_mul | 32768x512 | 659 | 929 | 660 | **1.41x** | 1.00x | 407 | True |
| add_mul | 65536x512 | 1405 | 2146 | 1404 | **1.53x** | 1.00x | 382 | True |
| add_mul | 32768x1024 | 1409 | 2147 | 1404 | **1.52x** | 1.00x | 381 | True |
| silu_mul_add | 32768x512 | 660 | 928 | 659 | **1.41x** | 1.00x | 407 | True |
| silu_mul_add | 65536x512 | 1406 | 2147 | 1405 | **1.53x** | 1.00x | 382 | True |
| silu_mul_add | 32768x1024 | 1408 | 2147 | 1405 | **1.52x** | 1.00x | 381 | True |
| gelu_mul_add | 32768x512 | 660 | 928 | 660 | **1.41x** | 1.00x | 407 | True |
| gelu_mul_add | 65536x512 | 1407 | 2147 | 1405 | **1.53x** | 1.00x | 382 | True |
| gelu_mul_add | 32768x1024 | 1409 | 2147 | 1405 | **1.52x** | 1.00x | 381 | True |

## Honest reading

- **IntentIR's auto-fused kernel beats the unfused Triton composition by 1.41-1.53x** — at the 1.5x traffic ceiling, i.e. it captures the
  full fusion advantage the library left on the table.
- **It ties a hand-written single fused Triton kernel (~1.0x)**: both move the
  same 4 HBM passes. That tie is the point — the win over the library is the
  fusion the library did not do, not a codegen trick — and it proves the
  auto-generated kernel is competent (GB/s matches a hand-written Triton copy).
- `silu(a)*b+c`'s unfused baseline already uses the *fused* `silu_and_mul`
  (the SwiGLU FFN kernel from production inference libraries) + a separate
  `add`: the best the library can do, so the win is not a strawman.
- **Scope:** this is the ELEMENTWISE front. It does NOT touch the
  FlashAttention latency gap (PERF.md), which remains a separate open front.
