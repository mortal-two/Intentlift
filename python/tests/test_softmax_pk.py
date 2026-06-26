"""softmax head-to-head PK vs the real liger kernel.

Strict correctness: our MLIR-generated read-once softmax kernel matches the fp32
reference AND the real liger Y at rel < 1e-3 (expect ~1e-6/1e-7), every row sums
to ~1.0, including a non-power-of-2 cols case that exercises the -inf tail mask
(the #1 non-pow2 correctness trap: masked lanes load -inf so they don't win the
max and exp->0, contributing 0 to the sum). Lenient perf: ours/liger <= 1.25 at
the HBM-bound (32768,1024) shape (KERNEL-ONLY) -- parity is the legitimate result
for a single memory-bound op on identical block geometry, NOT a strict beat.
GPU-gated, deterministic.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
# liger's softmax ops submodule; we call the raw kernel / _softmax_forward
# directly (forward-only, autograd-free) -- skip if unavailable.
pytest.importorskip("liger_kernel.ops.softmax")

_HAS_GPU = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="needs CUDA GPU")


@pytest.mark.parametrize("rows,cols", [(4096, 1024), (4096, 1000)])
def test_softmax_correct_vs_ref_and_liger(rows, cols):
    """Ours matches the fp32 reference AND liger's Y (rel < 1e-3); rows sum to ~1.

    (4096,1000) is non-power-of-2, exercising the BLOCK_SIZE=next_pow2 -inf tail
    mask -- the padded lanes must load -inf (not 0), so they never win the row max
    even when the true max is negative, and exp(-inf - m) = 0 keeps the denominator
    exact.
    """
    from liger_kernel.ops.softmax import _softmax_forward

    from intentir.bench.softmax_emit import run_softmax

    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32).contiguous()

    y_ours = run_softmax(x)
    y_lig, _bs, _nw, _mb = _softmax_forward(x)
    torch.cuda.synchronize()

    y_ref = torch.softmax(x.float(), dim=-1)

    def rel(a, b):
        return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)

    rel_y_ref = rel(y_ours, y_ref)
    rel_y_lig = rel(y_ours, y_lig)
    rowsum_err = (y_ours.sum(dim=-1) - 1.0).abs().max().item()

    assert rel_y_ref < 1e-3, f"({rows},{cols}) Y vs fp32 ref rel={rel_y_ref}"
    assert rel_y_lig < 1e-3, f"({rows},{cols}) Y vs liger rel={rel_y_lig}"
    assert rowsum_err < 1e-3, f"({rows},{cols}) row-sum err={rowsum_err}"


def test_softmax_all_negative_row_tail_mask():
    """An all-negative non-pow2 row: -inf tail fill is mandatory (0 would be wrong).

    If the tail were filled with 0 (the rms_norm trick), a 0 would win the max of an
    all-negative row and corrupt the result. -inf is the correct fill; this row
    must still match the fp32 reference.
    """
    from intentir.bench.softmax_emit import run_softmax

    rows, cols = 64, 1000  # non-pow2 -> 24 tail lanes per (BLOCK_SIZE=1024) block
    torch.manual_seed(1)
    x = -torch.rand(rows, cols, device="cuda", dtype=torch.float32) - 5.0  # all < -5
    y = run_softmax(x)
    ref = torch.softmax(x.float(), dim=-1)
    rel = (y - ref).abs().max().item() / (ref.abs().max().item() + 1e-12)
    rowsum_err = (y.sum(dim=-1) - 1.0).abs().max().item()
    assert rel < 1e-3, f"all-negative non-pow2 row rel={rel}"
    assert rowsum_err < 1e-3, f"all-negative row-sum err={rowsum_err}"


def test_softmax_pk_parity_or_better():
    """At HBM-bound (32768,1024), ours is parity-or-better vs liger (KERNEL-ONLY).

    softmax is a single memory-bound op (two reductions but read-x-once /
    write-y-once) and ours uses liger's own block geometry, so parity (~1.0x) is the
    expected legitimate result -- we assert a lenient 1.25x window on the
    KERNEL-ONLY metric (the headroom covers the run-to-run variance on this
    throttling card), NOT a strict beat. Short windows for test speed.
    """
    from intentir.bench.softmax_emit import run_softmax_pk

    res = run_softmax_pk(shapes=[(32768, 1024)], warmup=10, rep=40, windows=5)
    r = res[0]

    # Correctness must also hold under the PK harness.
    assert r["rel_y_ref"] < 1e-3, f"PK Y vs ref rel={r['rel_y_ref']}"
    assert r["rel_y_lig"] < 1e-3, f"PK Y vs liger rel={r['rel_y_lig']}"
    assert r["rowsum_err"] < 1e-3, f"PK row-sum err={r['rowsum_err']}"

    # Parity-or-better (lenient, KERNEL-ONLY): do NOT assert a strict beat for a
    # memory-bound op. r["ratio"] is the kernel-only ratio (the primary metric).
    assert r["ratio"] <= 1.25, (
        f"expected kernel-only ours/liger <= 1.25 (parity-or-better), got "
        f"{r['ratio']:.3f}x (ours {r['ours_ms']:.3f}ms / liger {r['liger_ms']:.3f}ms)"
    )
