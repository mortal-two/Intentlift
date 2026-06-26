"""layer_norm head-to-head PK vs the real liger kernel.

Strict correctness: our MLIR-generated read-once layer_norm kernel matches the
fp32 reference AND the real liger Y/Mean/RSTD at rel < 1e-3 (expect ~1e-7),
including a non-power-of-2 cols case that exercises the variance tail-mask re-zero
(the #1 non-pow2 correctness trap: ``x - mean`` is nonzero on masked lanes -- it
equals ``-mean`` -- so it must be re-zeroed before squaring, matching liger's
``tl.where(mask, X-mean, 0)``; without it the padding lanes' ``(-mean)^2`` inflate
the variance). Lenient perf: ours/liger <= 1.25 at the HBM-bound (32768,1024)
shape (KERNEL-ONLY) -- parity is the legitimate result for a single memory-bound op
on identical block geometry, NOT a strict beat. GPU-gated, deterministic.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
# liger's layer_norm ops submodule; we call layer_norm_forward / the raw kernel
# directly (forward-only, autograd-free -- the .apply path touches
# torch.distributed.tensor which is absent on this build). Skip if unavailable.
pytest.importorskip("liger_kernel.ops.layer_norm")

_HAS_GPU = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="needs CUDA GPU")

# layer_norm's conventional eps; MUST match the value baked into the emitter so the
# bit-faithful certification (rel ~1e-7) holds, not merely rel < 1e-3.
_EPS = 1e-5


def _fp32_ref(x, w, b, eps=_EPS):
    """fp32 reference: mean = x.mean; var = x.var(unbiased=False);
    rstd = rsqrt(var+eps); y = (x-mean)*rstd*w + b."""
    import torch.nn.functional as F

    y = F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), eps)
    mean = x.float().mean(dim=-1)
    rstd = torch.rsqrt(x.float().var(dim=-1, unbiased=False) + eps)
    return y, mean, rstd


@pytest.mark.parametrize("rows,cols", [(4096, 1024), (4096, 1000)])
def test_layernorm_correct_vs_ref_and_liger(rows, cols):
    """Ours matches the fp32 reference AND liger's Y/Mean/RSTD (rel < 1e-3).

    (4096,1000) is non-power-of-2, exercising the BLOCK_SIZE=next_pow2 variance
    tail-mask re-zero -- the padded lanes hold 0 in the cache so ``x - mean`` is
    ``-mean`` there; without re-zeroing before the square, ``(-mean)^2`` per tail
    lane would inflate the variance.
    """
    from liger_kernel.ops.layer_norm import layer_norm_forward

    from intentir.bench.layernorm_emit import run_layernorm

    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32).contiguous()
    w = torch.randn(cols, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(cols, device="cuda", dtype=torch.float32).contiguous()

    y_ours, mean_ours, rstd_ours = run_layernorm(x, w, b)
    y_lig, _x, mean_lig, rstd_lig, _bs, _nw = layer_norm_forward(x, w, b, _EPS)
    torch.cuda.synchronize()

    y_ref, mean_ref, rstd_ref = _fp32_ref(x, w, b)

    def rel(a, bb):
        return (a - bb).abs().max().item() / (bb.abs().max().item() + 1e-12)

    rel_y_ref = rel(y_ours, y_ref)
    rel_y_lig = rel(y_ours, y_lig)
    rel_mean_ref = rel(mean_ours, mean_ref)
    rel_mean_lig = rel(mean_ours, mean_lig.squeeze())
    rel_rstd_ref = rel(rstd_ours, rstd_ref)
    rel_rstd_lig = rel(rstd_ours, rstd_lig.squeeze())

    assert rel_y_ref < 1e-3, f"({rows},{cols}) Y vs fp32 ref rel={rel_y_ref}"
    assert rel_y_lig < 1e-3, f"({rows},{cols}) Y vs liger rel={rel_y_lig}"
    assert rel_mean_ref < 1e-3, f"({rows},{cols}) Mean vs ref rel={rel_mean_ref}"
    assert rel_mean_lig < 1e-3, f"({rows},{cols}) Mean vs liger rel={rel_mean_lig}"
    assert rel_rstd_ref < 1e-3, f"({rows},{cols}) RSTD vs ref rel={rel_rstd_ref}"
    assert rel_rstd_lig < 1e-3, f"({rows},{cols}) RSTD vs liger rel={rel_rstd_lig}"


def test_layernorm_pk_parity_or_better():
    """At HBM-bound (32768,1024), ours is parity-or-better vs liger (KERNEL-ONLY).

    layer_norm is a single memory-bound op (two sequential reductions but
    read-x-once / write-y-once + read w,b) and ours uses liger's own block geometry,
    so parity (~1.0x) is the expected legitimate result -- we assert a lenient 1.25x
    window on the KERNEL-ONLY metric (the headroom covers the run-to-run variance on
    this throttling card), NOT a strict beat. Short windows for test speed.
    """
    from intentir.bench.layernorm_emit import run_layernorm_pk

    res = run_layernorm_pk(shapes=[(32768, 1024)], warmup=10, rep=40, windows=5)
    r = res[0]

    # Correctness must also hold under the PK harness.
    assert r["rel_y_ref"] < 1e-3, f"PK Y vs ref rel={r['rel_y_ref']}"
    assert r["rel_y_lig"] < 1e-3, f"PK Y vs liger rel={r['rel_y_lig']}"
    assert r["rel_mean_lig"] < 1e-3, f"PK Mean vs liger rel={r['rel_mean_lig']}"
    assert r["rel_rstd_lig"] < 1e-3, f"PK RSTD vs liger rel={r['rel_rstd_lig']}"

    # Parity-or-better (lenient, KERNEL-ONLY): do NOT assert a strict beat for a
    # memory-bound op. r["ratio"] is the kernel-only ratio (the primary metric).
    assert r["ratio"] <= 1.25, (
        f"expected kernel-only ours/liger <= 1.25 (parity-or-better), got "
        f"{r['ratio']:.3f}x (ours {r['ours_ms']:.3f}ms / liger {r['liger_ms']:.3f}ms)"
    )
