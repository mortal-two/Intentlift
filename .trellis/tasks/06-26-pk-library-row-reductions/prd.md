# PK the library: high-perf MLIR row-reduction codegen vs real Triton kernels

## Why (the bar the user actually holds)

The Forge paper's E1 metric is the real target: **generate CUDA for the SAME
kernel and match/beat the source/provider Triton on THAT op**. The previous
"fusion" work (`06-26-beat-triton-fusion`) is now judged **off-target / toy**:
it invents a *different* comparison (a fusion gap the library left open) instead
of competing head-to-head on the library's own kernel. It is deprioritized — not
deleted, but explicitly relabeled as "elementwise fusion front", NOT the PK bar.

The user's directive (verbatim intent):
- 我们是 pk 人家的算子库 — compete head-to-head with the library's OWN kernels.
- 人家不都已经是高性能库了 — the baselines are already high-performance.
- forge 导入了 ~20 个高性能 kernel — target that dataset.
- 我们的后端生成（mlir）的时候要足够高性能 — OUR MLIR codegen must itself be
  high-performance.
- lower 是真实 mlir 工具链；对 cuda 要做的尽可能像 triton — real MLIR toolchain,
  Triton-like CUDA (one block/row, block-cooperative reduction, coalesced).

## The dataset (what Forge imported)

20 kernels: liger 15, xformers 4, flash_attn 1. Families: normalization 7,
loss/divergence 4, position_embedding 3, activation 2, attention 2, softmax 1,
linear 1. **Most are ROW-WISE REDUCTIONS** (rms_norm, softmax, layer_norm,
losses) — the canonical Triton high-perf pattern: ONE block per row, threads
cooperate on a block-level reduction over the feature dim.

Real baselines are INSTALLED and runnable on sm_120: `pip install --no-deps
liger_kernel` (0.8.0). The autograd `Function.apply` wrappers hit a
torch.distributed.tensor version mismatch, but the **forward launchers**
(`liger_kernel.ops.rms_norm.rms_norm_forward`, etc.) call the real `@triton.jit`
kernels directly and run correctly (rel ~1e-7). PK those.

## Feasibility — both gates PASSED (2026-06-26)

1. Real liger rms_norm runs on 5090D via `rms_norm_forward(...)`, rel 1.4e-7.
2. **Block-cooperative reduction lowers through MLIR-20 → sm_120a**: a
   hand-emitted `gpu.all_reduce add` row-sum kernel lowers cleanly (11.5KB
   cubin, non-empty `.text`) and computes correctly (rel 1.49e-7). This is the
   primitive that gates every row-reduction kernel and it works on the real
   toolchain. (The flash path used only scalar carries — this is new capability.)

## Honest performance expectation (anti-toy in the OTHER direction)

rms_norm / softmax / layer_norm are **memory-bound**: a competent kernel of
either origin saturates HBM, so the achievable backend-codegen result is
**parity (~1.0×)** and *parity IS the backend win* — "our MLIR-generated kernel
matches the hand-tuned library Triton on the same op." The paper's 1.6× comes
from OptGuide schedule SEARCH, which we are explicitly NOT building here. Target
HBM saturation; report the honest `ours/Triton` ratio; if we beat Triton it must
be *explainable* (e.g. the source schedule is suboptimal on the brand-new
sm_120), never manufactured by weakening the baseline.

## Scope

- IN: real high-perf MLIR row-reduction codegen, lowered via the real toolchain
  (mlir-opt → nvvm → ptxas → sm_120a), Triton-like; PK head-to-head vs the real
  liger kernels; honest `ours/Triton` + GB/s vs HBM peak.
- OUT (now): OptGuide schedule search / contract / mapping-axis machinery (that
  is the search apparatus, not the backend). xformers/flash_attn compiled bits.
  Fusion-across-boundaries (relabeled, not the PK bar).

## Children
- `06-26-rms-norm-pk` — first end-to-end PK number (simplest reduction).
- (next) softmax, layer_norm, then the rest of the reductions.

## End state (the user's actual ask)
The **backend generates** the high-perf row-reduction MLIR from intent — a
hand-emitted kernel is the *target spec* (same relationship flash_emit's
hand-emit has to the loop lowering), not "the backend". Hand-emit first to retire
risk + get the number; then wire it into the codegen path from intent.
