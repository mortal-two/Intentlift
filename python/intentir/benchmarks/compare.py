"""vs-Triton harness (M11): run the IntentIR attention kernel, the Triton flash
golden, and torch as references; report correctness and wall-clock latency.

Correctness is the headline: the intent-compiled kernel and the Triton flash
kernel compute the *same* attention output, so agreement across the three is the
end-to-end proof that semantic lifting -> lowering -> execution preserved the
algorithm. Latency is reported for context (the intent bring-up kernels are
naive one-thread-per-element; closing the perf gap is what retune addresses).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CompareResult:
    rel_vs_reference: float          # intent kernel vs Layer-A reference
    rel_intent_vs_triton: float      # intent kernel vs Triton flash golden
    rel_triton_vs_reference: float   # Triton flash golden vs Layer-A reference
    t_intent_ms: float
    t_triton_ms: float
    t_torch_ms: float

    def ok(self, atol_rel: float = 1e-3) -> bool:
        return (self.rel_vs_reference < atol_rel
                and self.rel_intent_vs_triton < atol_rel
                and self.rel_triton_vs_reference < atol_rel)


def _time_ms(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs() / (b.abs() + 1e-6)).max().item()


def compare_attention(intent_mlir: str, sq: int = 4, sk: int = 8, d: int = 16,
                      scale: float = 1.0, seed: int = 0,
                      block_n: int = 16) -> CompareResult:
    """Build Q/K/V, run all three attention paths, return correctness + latency.

    The intent graph computes softmax(Q·Kᵀ)·V (scale folded into Q on the host),
    so we pre-scale Q for the intent run and pass scale=1 there; Triton/torch get
    the raw Q with the explicit scale.
    """
    from intentir.backend.orchestrate import run_program
    from intentir.benchmarks.triton_flash import triton_flash_attention

    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(sq, d, device="cuda", generator=g).contiguous()
    k = torch.randn(sk, d, device="cuda", generator=g).contiguous()
    v = torch.randn(sk, d, device="cuda", generator=g).contiguous()

    qs = (q * scale).contiguous()  # scale folded into Q for the intent graph

    ref = torch.softmax((q @ k.T) * scale, dim=1) @ v  # Layer-A reference

    intent_out = next(iter(run_program(intent_mlir, {"q": qs, "k": k, "v": v}).values()))
    triton_out = triton_flash_attention(q, k, v, scale=scale, block_n=block_n)

    res = CompareResult(
        rel_vs_reference=_rel(intent_out, ref),
        rel_intent_vs_triton=_rel(intent_out, triton_out),
        rel_triton_vs_reference=_rel(triton_out, ref),
        t_intent_ms=_time_ms(lambda: run_program(intent_mlir, {"q": qs, "k": k, "v": v})),
        t_triton_ms=_time_ms(lambda: triton_flash_attention(q, k, v, scale, block_n)),
        t_torch_ms=_time_ms(lambda: torch.softmax((q @ k.T) * scale, dim=1) @ v),
    )
    return res
